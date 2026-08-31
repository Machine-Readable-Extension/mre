from __future__ import annotations

"""
HTML MRE generation: dispatch to per-site (per-domain) parsing logic.

A web page's body structure differs by site (Wikipedia's mw-heading div
structure is a good example). A parsing adapter is registered per domain,
and the one matching a document's source URL netloc gets picked. The
Wikipedia adapter's implementation (_wiki_build_structure_tree, etc., see
below) is the Wikipedia-specific parsing logic originally in
data_utils/mre_generator.py (v1), ported into this library's distribution
boundary.

So a site owner can distribute their own adapter as a separate package,
adapters installed via entry points (the "mre.site_adapters" group) are
auto-discovered at import time. See the "plugin auto-discovery" section below.
"""

import hashlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Callable
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from mre.appendix import _strip_appendix_sections, _consolidate_short_sections
from mre.nodes import strip_to_text_nodes

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HTMLSiteAdapter:
    """The bundle of HTML parsing/embedding logic for a single site.

    preprocess : (optional) cleans the BeautifulSoup document in-place before
                 extract() runs (e.g. removing appendix sections, merging short
                 section paragraphs). The return value (the list of removed
                 appendix headings) is unused by this library (paragraph-
                 granularity only, no <section> nodes) — the signature is kept
                 for a future section-granularity port. None (default) means
                 extract() runs on the soup with no preprocessing.
                 **Caution**: transforming the document here means paragraph
                 ids (pN) are assigned based on the transformed order — this
                 id must later share the exact same preprocessing rules with
                 whatever fetch-side parser re-walks the original (untransformed)
                 document (e.g. core/pipeline.py's _fetch_blocks_v3 re-applies
                 _strip_appendix_sections + _consolidate_short_sections
                 identically to generation time). If the rules diverge, a pid
                 ends up pointing at the wrong block.
    extract : BeautifulSoup-parsed document -> list of heading/paragraph nodes
              ({"type": "heading", "level", "text"} | {"type": "paragraph", "id", "text"})
    assign_ids : (optional) rewrites the ids of extract()'s result (paragraph
                 nodes only) in-place, based on title. None (default) keeps the
                 ids extract() assigned (usually "p1", "p2", ...). Matching
                 mre_generator3.py, the Wikipedia adapter uses this hook to
                 prepend the title's first letter (mitigating cross-document
                 id collisions).
    strip   : extract()'s result -> node list cleaned up for sending to the LLM
    embed   : (original html, assembled mre xml) -> html with mre inserted.
              Always inserts into the "original" html string, never the soup
              mutated by preprocess — appendix sections etc. must remain intact
              in the actual rendered document.
    fetch   : (optional) (html with mre embedded, node id) -> that paragraph's
              full (untruncated) text. The callable a RAG agent uses to actually
              fetch paragraph content — unlike the short preview text extract()
              produces at generation time for showing the LLM, this must return
              the raw text with no length limit. If id is "full", returns the
              whole document's text (all paragraphs concatenated) — for
              workflows that decide whether the whole document is needed after
              seeing just one paragraph. Must re-apply the same preprocessing
              used at generation time and walk in the same order, or the id
              mapping breaks — even as a separate implementation from extract(),
              it must share the exact same rules (see the preprocess note
              above). None (default) means this adapter doesn't support fetch
              (generation-only adapter).
    domains : the list of domains this adapter handles. Subdomains match too —
              domains=("wikipedia.org",) matches en.wikipedia.org,
              ko.wikipedia.org, etc.
    """
    name: str
    extract: Callable[[BeautifulSoup], list[dict]]
    strip: Callable[[list[dict]], list[dict]]
    embed: Callable[[str, str], str]
    domains: tuple[str, ...] = field(default_factory=tuple)
    preprocess: Callable[[BeautifulSoup], list[tuple[int, str]]] | None = None
    assign_ids: Callable[[list[dict], str], None] | None = None
    fetch: Callable[[str, str], str] | None = None


def compute_adapter_fingerprint(adapter: HTMLSiteAdapter) -> str:
    """Compute an adapter fingerprint by hashing the source code of the four
    functions extract/preprocess/assign_ids/fetch.

    These four functions (together or separately) determine "which paragraph
    does this id point to" — extract/preprocess/assign_ids assign ids at
    generation time, and fetch locates the paragraph again by that id. If any
    one of the four changes, the fingerprint automatically changes too — so an
    adapter author who forgets to bump a version by hand doesn't slip through
    (this scheme was chosen specifically to avoid the weakness of manual semver
    bumping). generate_mre() stamps this value into the document at generation
    time as <mre generator-fingerprint="...">, and fetch_block() recomputes and
    compares it at fetch time — a mismatch means the adapter's logic changed in
    between.

    A known limitation: even a purely cosmetic change (variable names,
    comments, formatting) changes the value. That false positive is accepted
    as a much safer failure mode than the false negative of "an id silently
    points to the wrong paragraph."

    Functions whose source can't be read (compiled extensions, C
    implementations, etc.) fall back to repr() — in that case the fingerprint
    may miss real code changes, but at least it won't crash.
    """
    parts: list[str] = []
    for fn in (adapter.extract, adapter.preprocess, adapter.assign_ids, adapter.fetch):
        if fn is None:
            parts.append("None")
            continue
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):
            parts.append(repr(fn))
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


class UnknownSiteError(LookupError):
    """Raised when no HTMLSiteAdapter is registered for the URL's domain and no fallback was given."""


# name -> adapter (domains live on adapter.domains)
_REGISTRY: dict[str, HTMLSiteAdapter] = {}


def register_site(adapter: HTMLSiteAdapter) -> None:
    """Register the adapter under its adapter.domains list. Re-registering with
    the same adapter.name overwrites the previous entry — this lets a plugin
    deliberately replace a builtin adapter (auto-discovery runs plugins after
    _register_builtin_sites(), so a plugin using the same name wins over the
    builtin one)."""
    if not adapter.domains:
        raise ValueError(f"adapter.domains is empty: {adapter.name!r}")
    if adapter.name in _REGISTRY:
        log.info("Re-registering site adapter %r (overwriting)", adapter.name)
    _REGISTRY[adapter.name] = adapter


def _domain_matches(netloc: str, domain: str) -> bool:
    netloc = netloc.lower().split(":")[0]  # strip the port
    return netloc == domain or netloc.endswith("." + domain)


def detect_site(url: str) -> str | None:
    """Find the registered site name matching url's netloc. Returns None if no match."""
    netloc = urlparse(url).netloc
    if not netloc:
        return None
    for name, adapter in _REGISTRY.items():
        if any(_domain_matches(netloc, d.lower()) for d in adapter.domains):
            return name
    return None


def get_site_adapter(url: str, *, fallback: HTMLSiteAdapter | None = None) -> HTMLSiteAdapter:
    """Return the HTMLSiteAdapter matching url.

    If no registered site matches, uses fallback when one is given; otherwise
    raises UnknownSiteError (since there's no generic HTML adapter yet, this
    chooses to fail explicitly rather than silently parsing with the wrong
    structure).
    """
    name = detect_site(url)
    if name is not None:
        return _REGISTRY[name]
    if fallback is not None:
        return fallback
    raise UnknownSiteError(f"No registered site adapter matches this domain: {url!r}")


def registered_sites() -> dict[str, tuple[str, ...]]:
    """The current {site name: domains} listing — builtins plus everything auto-discovered from plugins."""
    return {name: adapter.domains for name, adapter in _REGISTRY.items()}


def parse_html(
    url: str, html: str, title: str, *, fallback: HTMLSiteAdapter | None = None
) -> list[dict]:
    """Detect the site from url, parse html with the matching adapter, and
    return the node list cleaned up for the LLM (the strip() result).

    If adapter.preprocess exists it's applied before extract(); if
    adapter.assign_ids exists it's applied right after extract() (title-based
    id rewriting) — this keeps the paragraph ids this function returns
    identical to what the real generation path (generate_mre) produces."""
    adapter = get_site_adapter(url, fallback=fallback)
    soup = BeautifulSoup(html, "lxml")
    if adapter.preprocess is not None:
        adapter.preprocess(soup)
    nodes = adapter.extract(soup)
    if adapter.assign_ids is not None:
        adapter.assign_ids(nodes, title)
    return adapter.strip(nodes)


class FetchNotSupportedError(NotImplementedError):
    """Raised when the matched HTMLSiteAdapter doesn't implement fetch (a generation-only adapter)."""


class GeneratorFingerprintMismatch(RuntimeError):
    """Raised under strict=True when the generator-fingerprint stamped in the
    document differs from the currently installed adapter's fingerprint —
    meaning the adapter's parsing logic changed since this document was
    generated, and the id-to-paragraph mapping may have drifted."""


_MRE_ROOT_TAG_RE = re.compile(r"<mre\b([^>]*)>")
_XML_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def _extract_mre_root_attrs(html: str) -> dict[str, str]:
    """Extract attributes from the <mre ...> root tag inside html's embedded
    <script type="application/mre+xml">. Doesn't parse the whole document as
    XML: just regex-matches the <mre ...> opening tag, the same lightweight
    approach MREParser.extract_mre already uses."""
    m = _MRE_ROOT_TAG_RE.search(html)
    if not m:
        return {}
    return dict(_XML_ATTR_RE.findall(m.group(1)))


def _check_generator_fingerprint(adapter: HTMLSiteAdapter, html: str, *, strict: bool) -> None:
    attrs = _extract_mre_root_attrs(html)
    embedded_name = attrs.get("generator")
    embedded_fp = attrs.get("generator-fingerprint")
    if embedded_name is None or embedded_fp is None:
        return  # e.g. a document generated before this feature existed: nothing to compare, pass
    if embedded_name != adapter.name:
        return  # not a document this adapter produced, so there's nothing to compare fingerprints against
    current_fp = compute_adapter_fingerprint(adapter)
    if current_fp == embedded_fp:
        return
    msg = (
        f"Adapter {adapter.name!r}'s parsing logic appears to have changed "
        f"since this document was generated (fingerprint at generation time="
        f"{embedded_fp!r}, currently installed adapter={current_fp!r}). The "
        f"id-to-paragraph mapping may have drifted."
    )
    if strict:
        raise GeneratorFingerprintMismatch(msg)
    log.warning(msg)


def fetch_block(
    url: str, html: str, node_id: str, *, fallback: HTMLSiteAdapter | None = None, strict: bool = False
) -> str:
    """Detect the site from url and fetch node_id's full paragraph text via the matching adapter's fetch().

    Lets a RAG agent that consumes an MRE produced by generate_mre() fetch a
    paragraph the same way for any site through this one function — the agent
    code never needs to know "is this document Wikipedia?". Raises
    FetchNotSupportedError if the matched adapter doesn't implement fetch.

    Before fetching, compares the generator-fingerprint stamped in the
    document (if any) against the currently installed adapter's fingerprint
    (see compute_adapter_fingerprint) — a mismatch logs a warning that the
    adapter changed since this document was generated and the id mapping may
    have drifted. Under strict=True, raises GeneratorFingerprintMismatch
    instead of warning. If the document has no fingerprint at all (e.g.
    generated before this feature existed), the comparison is skipped."""
    adapter = get_site_adapter(url, fallback=fallback)
    if adapter.fetch is None:
        raise FetchNotSupportedError(
            f"Site adapter {adapter.name!r} does not support fetch (generation-only)"
        )
    _check_generator_fingerprint(adapter, html, strict=strict)
    return adapter.fetch(html, node_id)


# ─────────────────────────────────────────────
# Wikipedia adapter implementation
# ─────────────────────────────────────────────
# Ported from the same-named functions in data_utils/mre_generator.py (v1)
# into this library's distribution boundary.

_WIKI_NODE_TEXT_LIMIT = 400
# Appendix sections get filtered a second time here inside
# build_structure_tree itself, as a backup to appendix.py's
# _strip_appendix_sections (which removes them wholesale from the soup at
# the preprocess stage, before extract runs). This preserves the original
# behavior of leaving at least a minimal filter in place for call paths that
# preprocess doesn't cover, e.g. calling build_structure_tree directly
# without preprocess.
_WIKI_APPENDIX_HEADINGS = {"References", "See also", "External links"}


def _wiki_extract_node_text(element: Tag, limit: int = _WIKI_NODE_TEXT_LIMIT) -> str:
    """Extract visible text from a tag, stripped and truncated."""
    text = element.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _wiki_build_structure_tree(soup: BeautifulSoup) -> list[dict]:
    """
    Extract mw-heading section titles and <p> paragraphs from Wikipedia HTML, in document order.

    Returns: nodes, each item one of two types:
      - {"type": "heading", "level": int, "text": str}
      - {"type": "paragraph", "id": str, "text": str}
    """
    container = (
        soup.find(id="bodyContent")
        or soup.find("div", class_="mw-parser-output")
        or soup.find("body")
        or soup
    )

    nodes: list[dict] = []
    p_counter = [0]

    def process_element(el: Tag) -> None:
        if not isinstance(el, Tag):
            return

        # div.mw-heading.mw-heading{k} -> a heading node
        if el.name == "div":
            raw_classes: str | list[str] = el.get("class") or []
            classes: list[str] = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
            for cls in classes:
                m = re.match(r"mw-heading(\d)", cls)
                if m:
                    level = int(m.group(1))
                    h_tag = el.find(f"h{level}")
                    if h_tag:
                        htext = h_tag.get_text(strip=True)
                        if htext in _WIKI_APPENDIX_HEADINGS:
                            return  # appendix sections aren't included in the structure tree
                        nodes.append({
                            "type": "heading",
                            "level": level,
                            "text": htext,
                        })
                    return  # don't recurse into an mw-heading's contents

        # <p>, <ul>, <ol> -> a paragraph node
        # MediaWiki renders *(bullet) / #(number) wiki syntax as <ul><li> /
        # <ol><li>, so these are treated like body paragraphs and their <li>
        # text is joined into the paragraph text. <ol class="references">
        # (the appendix-style source list) is excluded.
        #
        # pid counting policy: only increment the counter and add a node
        # when the text is *non-empty*. This keeps MRE's <node id="pN">
        # sequence continuous as p1, p2, p3, ... (preventing the LLM from
        # hallucinating a paragraph sitting in an invisible gap).
        # MREParser.fetch_blocks's indexing (core/mre.py) follows the same
        # rule: count only non-empty paragraphs.
        if el.name in ("p", "ul", "ol"):
            if el.name == "ol" and "references" in (el.get("class") or []):
                return
            text = _wiki_extract_node_text(el)
            if text:
                p_counter[0] += 1
                pid = el.get("id") or f"p{p_counter[0]}"
                nodes.append({
                    "type": "paragraph",
                    "id": pid,
                    "text": text,
                })
            return  # don't recurse into a matched element's contents

        # Recurse into the children of any other element
        for child in el.children:
            if isinstance(child, Tag):
                process_element(child)

    for child in container.children:
        if isinstance(child, Tag):
            process_element(child)

    return nodes


_MRE_SCRIPT_TAG_RE = re.compile(
    r'\s*<script\s+type="application/mre\+xml"\s*>.*?</script>\s*',
    re.IGNORECASE | re.DOTALL,
)


def _wiki_inject_mre_into_html(html: str, mre_xml: str) -> str:
    """
    Inject the MRE block as a <script type="application/mre+xml"> tag
    inside <head>. Falls back to prepending if no <head> is found.

    Strips any existing MRE script tag(s) first, so calling this again on
    already-embedded html replaces the block instead of leaving the old one
    in place -- extract_mre_xml() finds the first match in document order,
    so without this the OLDEST embed would keep winning on re-embed, the
    opposite of insert_mre_into_zip()/embed_mre_pdf(), which both replace.
    """
    html = _MRE_SCRIPT_TAG_RE.sub("", html)
    mre_tag = f'\n<script type="application/mre+xml">\n{mre_xml}\n</script>\n'

    head_end = re.search(r"</head\s*>", html, re.IGNORECASE)
    if head_end:
        pos = head_end.start()
        return html[:pos] + mre_tag + html[pos:]

    html_start = re.search(r"<html[^>]*>", html, re.IGNORECASE)
    if html_start:
        pos = html_start.end()
        return html[:pos] + "\n<head>" + mre_tag + "</head>" + html[pos:]

    return mre_tag + html


def _wiki_title_letter_prefix(title: str) -> str:
    """Use the document title's first letter (uppercased) as the node id prefix.

    Mitigates cross-document id collisions: when several candidate
    documents' headers appear together in one prompt, a purely numeric id
    like "p3" overlaps across documents, which can lead an agent to
    hallucinate a pid belonging to the wrong one. A per-document prefix at
    least makes "which document does this pid belong to" visually
    distinguishable (same scheme as the same-named function in
    data_utils/mre_generator3.py, carried over as-is for exact
    reproduction). A title with no letters at all falls back to 'X'.
    """
    m = re.search(r"[A-Za-z]", title)
    return m.group(0).upper() if m else "X"


def _wiki_apply_title_letter_ids(nodes: list[dict], title: str) -> None:
    """Rewrite the id of paragraph-type nodes in place, from "p{N}" to "{letter}{N}".

    Keeps the original appearance order (N) assigned by
    _wiki_build_structure_tree() and only changes the prefix. It only
    matches the "p{N}" fallback pattern, so it leaves alone any id the HTML
    actually had (e.g. a real <p id="...">, which is rare). This must run
    before the LLM call (after strip()) so the LLM sees and echoes the new
    ids, keeping the new id scheme consistent all the way through XML
    assembly. It's applied via HTMLSiteAdapter's assign_ids hook, right
    after extract() and before strip().
    """
    letter = _wiki_title_letter_prefix(title)
    for node in nodes:
        if node.get("type") != "paragraph":
            continue
        m = re.match(r"^p(\d+)$", node.get("id", ""))
        if m:
            node["id"] = f"{letter}{m.group(1)}"


# ─────────────────────────────────────────────
# Built-in site adapter registration
# ─────────────────────────────────────────────

def _wiki_preprocess(soup: BeautifulSoup) -> list[tuple[int, str]]:
    # v3-only preprocessing: strips appendix sections (References/See
    # also/External links, etc.) entirely, since leaving them in would give
    # appendix content like IMDb links or stub notices their own desc/keys
    # too, then merges short subsections' paragraphs into one (avoiding
    # noisy per-paragraph desc/keys). Same order as
    # data_utils/mre_generator3.py's process_db_document_async2. Both
    # extract() and fetch() reapply this exact function as their single
    # source of truth, so changing the order or the functions here breaks
    # the id mapping between generation and fetch.
    appendix_sections = _strip_appendix_sections(soup)
    _consolidate_short_sections(soup)
    return appendix_sections


def _wiki_fetch(html: str, node_id: str) -> str:
    """Fetch node_id's full (untruncated) paragraph text from a Wikipedia document's html with MRE embedded.

    To match the ordering _wiki_build_structure_tree() assigned, this must
    reapply the exact same generation-time preprocessing (_wiki_preprocess:
    stripping appendices and consolidating short sections) and then walk
    with the same rules (<p>/<ul>/<ol>, excluding ol.references, no
    recursion into a matched element's contents, skipping empty
    paragraphs). extract()'s _wiki_extract_node_text() truncates to 400
    characters for the LLM prompt, but this function returns the untruncated
    text an agent will actually read.

    If node_id is "full", returns every paragraph's text concatenated
    (for workflows that decide, after seeing just one paragraph, that they
    need the whole document). Otherwise, the leading letter prefix is
    ignored and only the trailing number is used to find the paragraph's
    position, so any prefix scheme works the same way regardless of what
    assign_ids applied ("p1"/"B1"/"P15", etc.). Returns an empty string if
    not found, rather than raising, matching core/pipeline.py's
    _fetch_blocks_v3 contract.

    If this document's <script type="application/mre+xml"> contains a
    legacy-schema <resources><target id="..."> block, its content is
    appended after the paragraph text. mre's own generation schema
    (build_mre_xml) never produces <resources>, but this stays compatible
    with documents made by other generators.
    """
    soup = BeautifulSoup(html, "lxml")
    _wiki_preprocess(soup)

    container = (
        soup.find(id="bodyContent")
        or soup.find("div", class_="mw-parser-output")
        or soup.find("body")
        or soup
    )

    blocks: list[Tag] = []

    def _walk(node) -> None:
        if not isinstance(node, Tag):
            return
        if node.name in ("p", "ul", "ol"):
            if node.name == "ol" and "references" in (node.get("class") or []):
                return
            text = node.get_text(separator=" ", strip=True)
            if text:
                blocks.append(node)
            return  # don't descend into a matched element's contents
        for child in node.children:
            _walk(child)

    _walk(container)

    mre_script = soup.find("script", {"type": "application/mre+xml"})

    def _resources_text(target_id: str | None) -> str:
        if not (mre_script and mre_script.string):
            return ""
        if target_id is None:
            all_targets = re.findall(
                r"<target\s+id=[\"'][^\"']*[\"'][^>]*>(.*?)</target>",
                mre_script.string, re.DOTALL,
            )
            return " ".join(
                " ".join(re.sub(r"<[^>]+>", " ", t).split()) for t in all_targets
            ).strip()
        target_pat = re.compile(
            r"<target\s+id=[\"']" + re.escape(target_id) + r"[\"'][^>]*>(.*?)</target>",
            re.DOTALL,
        )
        tm = target_pat.search(mre_script.string)
        if not tm:
            return ""
        return " ".join(re.sub(r"<[^>]+>", " ", tm.group(1)).split())

    if node_id == "full":
        if not blocks:
            return ""
        block_text = "\n\n".join(b.get_text(separator=" ", strip=True) for b in blocks)
        items_text = _resources_text(None)
        return f"{block_text}\n\n[resources]\n{items_text}" if items_text else block_text

    el = None
    m = re.match(r"^[A-Za-z]*(\d+)$", node_id)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(blocks):
            el = blocks[idx - 1]
    if el is None:
        return ""

    block_text = el.get_text(separator=" ", strip=True)
    items_text = _resources_text(node_id)
    return f"{block_text}\n\n[resources]\n{items_text}" if items_text else block_text


# ─────────────────────────────────────────────
# Built-in site adapter registration
# ─────────────────────────────────────────────

def _register_builtin_sites() -> None:
    register_site(
        HTMLSiteAdapter(
            name="wikipedia",
            domains=("wikipedia.org",),
            extract=_wiki_build_structure_tree,
            strip=strip_to_text_nodes,
            embed=_wiki_inject_mre_into_html,
            preprocess=_wiki_preprocess,
            assign_ids=_wiki_apply_title_letter_ids,
            fetch=_wiki_fetch,
        ),
    )


# ─────────────────────────────────────────────
# Plugin auto-discovery (entry points)
# ─────────────────────────────────────────────
# A site owner distributes their own adapter as a separate pip package and
# declares it in that package's pyproject.toml like this:
#
#   [project.entry-points."mre.site_adapters"]
#   my-site = "my_mre_adapter:ADAPTER"
#
# The value must be either an HTMLSiteAdapter instance, or a no-argument
# factory that returns one. Every such installed package is scanned and
# auto-registered when mre is imported, so a new site can be added without
# touching the core library code.
#
# discover_plugin_adapters() is called from the end of mre/__init__.py, not
# from this module (don't call it directly here): plugins conventionally
# import `from mre import HTMLSiteAdapter`, but this module is loaded while
# mre.__init__ is still in the middle of importing itself, so the mre
# package isn't fully initialized yet at that point and a circular import
# would result. It's safe to call again if a new plugin gets installed
# after import (a runtime reinstall) — register_site() overwrites
# idempotently.
_ENTRY_POINT_GROUP = "mre.site_adapters"


def discover_plugin_adapters() -> None:
    """Scan installed packages for the 'mre.site_adapters' entry point and auto-register them.
    mre/__init__.py calls this automatically once when mre is imported — you usually don't need
    to call it directly, only when you've installed a new adapter package mid-process and want
    to re-discover it."""
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            adapter = obj() if not isinstance(obj, HTMLSiteAdapter) else obj
        except Exception as e:  # noqa: BLE001 — one plugin's failure must not block discovering the rest
            log.warning("Failed to load mre.site_adapters entry point [%s]: %s", ep.name, e)
            continue
        if not isinstance(adapter, HTMLSiteAdapter):
            log.warning(
                "mre.site_adapters entry point %r did not return an HTMLSiteAdapter: %r",
                ep.name, adapter,
            )
            continue
        register_site(adapter)
        log.info("Registered plugin site adapter: %s (entry point %r)", adapter.name, ep.name)


_register_builtin_sites()
