"""HTML MRE generation — dispatching parsing logic per site (domain).

A web page's body structure differs from site to site (Wikipedia's
``mw-heading`` div structure is a typical example). A parsing adapter is
registered per domain, and the right one is picked by the document's
source URL's netloc.

So that a site owner can ship their own adapter as a separate package,
adapters installed under an entry point (the ``"mre.site_adapters"``
group) are auto-discovered at import time — see the "plugin
auto-discovery" section below.
"""

from __future__ import annotations

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
    """Parsing/embedding logic bundled for one site.

    Attributes:
        preprocess: (Optional) Cleans up a BeautifulSoup document in-place
            before ``extract()`` runs (e.g. removing appendix sections,
            merging short-section paragraphs). Its return value (the list
            of removed appendix headings) isn't consumed by this library
            (paragraph-granularity only, no ``<section>`` is generated) —
            the signature is kept for a possible future section-granularity
            mode. ``None`` (default) means ``extract()`` runs on the soup
            unmodified.
            **Warning**: transforming the document here determines the
            order paragraph `id`s (``pN``) get assigned in at generation
            time. The fetch-side parser later re-walks the *original*
            (untransformed) document, so it must apply the exact same
            preprocessing rule (e.g. re-applying ``_strip_appendix_sections``
            + ``_consolidate_short_sections`` in the same order as
            generation). If the two ever diverge, a `pid` ends up pointing
            at the wrong block.
        extract: A parsed BeautifulSoup document -> list of heading/paragraph
            nodes (``{"type": "heading", "level", "text"}`` |
            ``{"type": "paragraph", "id", "text"}``).
        assign_ids: (Optional) Rewrites the `id` of ``extract()``'s
            paragraph-type nodes in-place, based on ``title``. ``None``
            (default) keeps whatever id ``extract()`` assigned (typically
            "p1", "p2", ...). The Wikipedia adapter uses this hook to
            prefix ids with the title's first letter, reducing
            cross-document id collisions.
        strip: ``extract()`` output -> node list cleaned up for the LLM.
        embed: ``(original html, assembled mre xml) -> html with mre
            inserted``. Always inserts into the *original* html string,
            never the soup mutated by preprocessing — appendix sections
            etc. must remain in the actually-rendered document.
        fetch: (Optional) ``(mre-embedded html, node id) -> that
            paragraph's full, untruncated text``. This is what a RAG agent
            calls to actually retrieve paragraph content — unlike the
            short preview text ``extract()`` produces for the LLM at
            generation time, this must return the full text with no
            length limit. ``id="full"`` returns the whole document's text
            (every paragraph concatenated), for a workflow that decides
            from one paragraph that it needs the whole document. Must
            re-apply the exact same preprocessing as generation time and
            walk in the same order, or the id mapping breaks — even
            though it's a separate implementation from ``extract()``, it
            must share the same rules (see ``preprocess`` above). ``None``
            (default) means this adapter doesn't support fetch
            (generation-only adapter).
        domains: Domains this adapter handles. Subdomains match too —
            ``domains=("wikipedia.org",)`` matches ``en.wikipedia.org``,
            ``ko.wikipedia.org``, etc.
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
    """Hash the source of ``extract``/``preprocess``/``assign_ids``/``fetch`` into an adapter fingerprint.

    These four functions, together, determine which paragraph an `id`
    points at — ``extract``/``preprocess``/``assign_ids`` assign ids at
    generation time, ``fetch`` locates a paragraph by that id again later.
    If any one of the four changes, the fingerprint changes automatically
    — an adapter author can't forget to bump a version number, because
    there's no version number to bump (this sidesteps the usual weakness
    of manual semver bumps). ``generate_mre()`` stamps this value into the
    document as ``<mre generator-fingerprint="...">`` at generation time,
    and ``fetch_block()`` recomputes it at fetch time to compare — a
    mismatch means the adapter's logic changed in between.

    A known limitation: even a purely cosmetic change (a renamed variable,
    a reworded comment, reformatting) changes the value too. That false
    positive is accepted as the far safer failure direction, compared to
    the false negative of an `id` silently pointing at the wrong
    paragraph.

    A function whose source can't be read (a compiled extension, a C
    implementation, ...) falls back to ``repr()`` — the fingerprint may
    then miss a real code change, but at least this doesn't crash.
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
    """Raised when a URL's domain has no registered HTMLSiteAdapter and no fallback was given."""


# name -> adapter (domains live on adapter.domains)
_REGISTRY: dict[str, HTMLSiteAdapter] = {}


def register_site(adapter: HTMLSiteAdapter) -> None:
    """Register ``adapter`` under its ``adapter.domains``.

    Registering again under the same ``adapter.name`` overwrites the
    previous entry — this lets a plugin deliberately replace a built-in
    adapter (since auto-discovery registers plugins after
    ``_register_builtin_sites()``, a plugin using the same name wins).
    """
    if not adapter.domains:
        raise ValueError(f"adapter.domains is empty: {adapter.name!r}")
    if adapter.name in _REGISTRY:
        log.info("Re-registering (overwriting) site adapter %r", adapter.name)
    _REGISTRY[adapter.name] = adapter


def _domain_matches(netloc: str, domain: str) -> bool:
    netloc = netloc.lower().split(":")[0]  # strip port
    return netloc == domain or netloc.endswith("." + domain)


def detect_site(url: str) -> str | None:
    """Find the registered site name matching ``url``'s netloc, or ``None``."""
    netloc = urlparse(url).netloc
    if not netloc:
        return None
    for name, adapter in _REGISTRY.items():
        if any(_domain_matches(netloc, d.lower()) for d in adapter.domains):
            return name
    return None


def get_site_adapter(url: str, *, fallback: HTMLSiteAdapter | None = None) -> HTMLSiteAdapter:
    """Return the HTMLSiteAdapter matching ``url``.

    Uses ``fallback`` if given and no registered site matches; otherwise
    raises ``UnknownSiteError`` — since there's no generic HTML adapter
    yet, failing explicitly is preferred over silently parsing with the
    wrong structure.
    """
    name = detect_site(url)
    if name is not None:
        return _REGISTRY[name]
    if fallback is not None:
        return fallback
    raise UnknownSiteError(f"No registered site adapter matches this domain: {url!r}")


def registered_sites() -> dict[str, tuple[str, ...]]:
    """Return ``{site name: domains}`` for every currently registered site — built-in plus discovered plugins."""
    return {name: adapter.domains for name, adapter in _REGISTRY.items()}


def parse_html(
    url: str, html: str, title: str, *, fallback: HTMLSiteAdapter | None = None
) -> list[dict]:
    """Detect the site from ``url``, parse ``html`` with the matching adapter,
    and return the node list cleaned up for the LLM (the ``strip()`` result).

    ``adapter.preprocess`` (if present) runs before ``extract()``, and
    ``adapter.assign_ids`` (if present) runs right after it — so the
    paragraph ids this function returns match what ``generate_mre()``
    itself would produce.
    """
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
    """Raised under ``strict=True`` when a document's embedded
    generator-fingerprint doesn't match the currently installed adapter's
    fingerprint — meaning the adapter's parsing logic changed since this
    document was generated, and the id-to-paragraph mapping may now be wrong.
    """


_MRE_ROOT_TAG_RE = re.compile(r"<mre\b([^>]*)>")
_XML_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def _extract_mre_root_attrs(html: str) -> dict[str, str]:
    """Extract the attributes of the ``<mre ...>`` root tag embedded in
    html's ``<script type="application/mre+xml">``.

    Reads only the opening ``<mre ...>`` tag via regex, without parsing
    the whole document as XML — the same lightweight approach
    ``extract_mre_xml`` uses.
    """
    m = _MRE_ROOT_TAG_RE.search(html)
    if not m:
        return {}
    return dict(_XML_ATTR_RE.findall(m.group(1)))


def _check_generator_fingerprint(adapter: HTMLSiteAdapter, html: str, *, strict: bool) -> None:
    attrs = _extract_mre_root_attrs(html)
    embedded_name = attrs.get("generator")
    embedded_fp = attrs.get("generator-fingerprint")
    if embedded_name is None or embedded_fp is None:
        return  # e.g. a document generated before this feature existed — nothing to compare, pass
    if embedded_name != adapter.name:
        return  # not generated by this adapter — fingerprint comparison doesn't apply
    current_fp = compute_adapter_fingerprint(adapter)
    if current_fp == embedded_fp:
        return
    msg = (
        f"Adapter {adapter.name!r}'s parsing logic appears to have changed "
        f"since this document was generated (fingerprint at generation "
        f"time={embedded_fp!r}, currently installed adapter={current_fp!r}). "
        "The id-to-paragraph mapping may be wrong."
    )
    if strict:
        raise GeneratorFingerprintMismatch(msg)
    log.warning(msg)


def fetch_block(
    url: str, html: str, node_id: str, *, fallback: HTMLSiteAdapter | None = None, strict: bool = False
) -> str:
    """Detect the site from ``url`` and fetch ``node_id``'s full paragraph text via the matching adapter's ``fetch()``.

    Lets a RAG agent consuming MRE documents fetch paragraphs from any
    supported site through this one function, without the agent needing
    to know what site produced the document. Raises
    ``FetchNotSupportedError`` if the matched adapter doesn't implement
    fetch.

    Before fetching, compares the document's embedded
    generator-fingerprint (if any) against the currently installed
    adapter's fingerprint (see ``compute_adapter_fingerprint``) — a
    mismatch means the adapter changed since this document was generated,
    so the id mapping may be wrong; that's logged as a warning. Under
    ``strict=True``, a mismatch raises ``GeneratorFingerprintMismatch``
    instead of just logging. If the document has no fingerprint at all
    (e.g. generated before this feature existed), the comparison is
    skipped entirely.
    """
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

_WIKI_NODE_TEXT_LIMIT = 400
# Appendix-like sections get filtered again here, on top of
# appendix.py's _strip_appendix_sections (the preprocess step, which
# removes them from the soup entirely before extract() runs) — a second
# line of defense. This keeps at least a minimal filter in place for call
# paths preprocess doesn't cover (e.g. calling build_structure_tree
# directly, without preprocess).
_WIKI_APPENDIX_HEADINGS = {"References", "See also", "External links"}


def _wiki_extract_node_text(element: Tag, limit: int = _WIKI_NODE_TEXT_LIMIT) -> str:
    """Extract visible text from a tag, stripped and truncated."""
    text = element.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _wiki_build_structure_tree(soup: BeautifulSoup) -> list[dict]:
    """Extract mw-heading section titles and `<p>` paragraphs from
    Wikipedia HTML, in document order.

    Returns:
        Nodes, each one of two shapes:
        ``{"type": "heading", "level": int, "text": str}`` or
        ``{"type": "paragraph", "id": str, "text": str}``.
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

        # div.mw-heading.mw-heading{k} -> heading node
        if el.name == "div":
            classes = el.get("class", [])
            for cls in classes:
                m = re.match(r"mw-heading(\d)", cls)
                if m:
                    level = int(m.group(1))
                    h_tag = el.find(f"h{level}")
                    if h_tag:
                        htext = h_tag.get_text(strip=True)
                        if htext in _WIKI_APPENDIX_HEADINGS:
                            return  # appendix-like sections are excluded from the structure tree
                        nodes.append({
                            "type": "heading",
                            "level": level,
                            "text": htext,
                        })
                    return  # don't recurse inside an mw-heading div

        # <p>, <ul>, <ol> -> paragraph node
        # MediaWiki renders *(bullet)/#(numbered) wiki syntax as
        # <ul><li>/<ol><li>, so these are treated like body paragraphs,
        # joining <li> text together as the body. <ol class="references">
        # (an appendix-like source list) is excluded.
        #
        # pid counting policy: the counter only advances, and a node is
        # only added, when the text is *non-empty*. This keeps MRE's
        # <node id="pN"> sequence contiguous (p1, p2, p3, ...) — a gap
        # would make an LLM liable to hallucinate a paragraph existing in
        # it. Fetch-side indexing uses this same rule (only non-empty
        # paragraphs are counted).
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
            return  # don't recurse inside a matched element

        # Anything else: recurse into children
        for child in el.children:
            if isinstance(child, Tag):
                process_element(child)

    for child in container.children:
        if isinstance(child, Tag):
            process_element(child)

    return nodes


def _wiki_inject_mre_into_html(html: str, mre_xml: str) -> str:
    """
    Inject the MRE block as a <script type="application/mre+xml"> tag
    inside <head>. Falls back to prepending if no <head> is found.
    """
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
    """Use the document title's first letter (uppercased) as a node id prefix.

    This reduces cross-document id collisions: when several candidate
    documents' headers are shown in the same prompt, purely numeric ids
    like "p3" collide across documents, which can make an agent
    hallucinate a `pid` request against the wrong document. Prefixing each
    document with a distinct letter at least makes "which document does
    this pid belong to" visually distinguishable. Falls back to 'X' for a
    title with no letters at all.
    """
    m = re.search(r"[A-Za-z]", title)
    return m.group(0).upper() if m else "X"


def _wiki_apply_title_letter_ids(nodes: list[dict], title: str) -> None:
    """Rewrite paragraph-type nodes' ids in-place, from "p{N}" to "{letter}{N}".

    Keeps the original appearance order (N) that
    ``_wiki_build_structure_tree()`` assigned, changing only the prefix.
    Only ids matching the "p{N}" fallback pattern are touched, so an id
    HTML actually carried on its own (rare, e.g. `<p id="...">`) is left
    alone. Must run before the LLM call, so the LLM sees and echoes the
    new ids, keeping everything downstream — including final XML assembly
    — consistent with the new id scheme. Applied via ``HTMLSiteAdapter``'s
    ``assign_ids`` hook, right after ``extract()`` and before ``strip()``.
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
    # Removes appendix sections (References/See also/External links/etc.)
    # wholesale — left in place, they'd otherwise get their own desc/keys
    # generated from appendix content like IMDb links or stub notices —
    # and merges short subsections' paragraphs into one (avoiding
    # low-signal, individually-generated desc/keys). Both ``extract()``
    # and ``fetch()`` re-apply this exact function, in this order — it's
    # the single source of truth for paragraph ordering, so changing the
    # order or the functions here breaks the generation-to-fetch id mapping.
    appendix_sections = _strip_appendix_sections(soup)
    _consolidate_short_sections(soup)
    return appendix_sections


def _wiki_fetch(html: str, node_id: str) -> str:
    """Fetch ``node_id``'s full, untruncated paragraph text from MRE-embedded Wikipedia document HTML.

    To line up with the numbering ``_wiki_build_structure_tree()``
    assigns, this re-applies ``_wiki_preprocess`` (appendix removal + short-
    section merging) exactly as generation did, then walks with the same
    rules (`<p>`/`<ul>`/`<ol>`, excluding `ol.references`, no recursion
    inside a matched element, skipping empty paragraphs) — while
    ``extract()``'s ``_wiki_extract_node_text()`` truncates to 400
    characters for the LLM prompt, this returns the untruncated text an
    agent will actually read.

    ``node_id="full"`` returns every paragraph's text concatenated — for a
    workflow that decides a single paragraph isn't enough and the whole
    document is needed. Otherwise, the id's leading letter prefix is
    ignored and only its trailing number is used to find the paragraph's
    position — so any prefix scheme ("p1"/"B1"/"P15", whatever
    ``assign_ids`` applied) works the same way. Returns an empty string if
    not found (no exception raised — the same contract as
    ``fetch_opc()``).

    If this document has a legacy-schema ``<resources><target id="...">``
    inside its ``<script type="application/mre+xml">``, its text is
    appended after the paragraph text — mre's own generator
    (``build_mre_xml``) never produces ``<resources>``, but this keeps
    compatibility with documents produced by other generators.
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
            return  # don't descend inside a matched element
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
# A site owner ships their own adapter as a separate pip package, and
# declares it in that package's pyproject.toml:
#
#   [project.entry-points."mre.site_adapters"]
#   my-site = "my_mre_adapter:ADAPTER"
#
# The value must be an HTMLSiteAdapter instance, or a no-argument factory
# returning one. Every such installed package is scanned and
# auto-registered at the time mre is imported — a new site can be added
# without touching this library's core code.
#
# discover_plugin_adapters() is called at the very end of mre/__init__.py,
# not from this module (must not be called here directly) — a plugin
# conventionally imports via `from mre import HTMLSiteAdapter`, but this
# module is loaded partway through mre.__init__ importing itself, so the
# mre package isn't fully initialized yet at that point and a circular
# import would result. It's safe to call again later (e.g. after
# installing a new plugin package at runtime) — register_site() overwrites
# idempotently.
_ENTRY_POINT_GROUP = "mre.site_adapters"


def discover_plugin_adapters() -> None:
    """Scan installed packages for the ``'mre.site_adapters'`` entry point and auto-register them.

    Called once automatically by ``mre/__init__.py`` when ``mre`` is
    imported — there's usually no need to call this directly, except to
    re-discover adapters after installing a new adapter package mid-process.
    """
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            adapter = obj() if not isinstance(obj, HTMLSiteAdapter) else obj
        except Exception as e:  # noqa: BLE001 — one plugin's failure must not block discovery of the rest
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
