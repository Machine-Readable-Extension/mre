from __future__ import annotations

"""
Wikipedia appendix section filtering and short-section consolidation, used
to clean up an HTML document before it's sent to the LLM.

Ported from the same-named section of data_utils/mre_generator3.py into this
library's distribution boundary.

**Single source of truth**: paragraph ids (pN) are assigned at generation
time based on the order in which these functions transform the soup
(stripping appendices, merging short sections). The final document embeds
the original, untransformed HTML, so a later pid-based fetch must reapply
these same two functions in the same order before walking paragraphs, for
the generation-time pid sequence to line up with the fetch-time walk
sequence (core/pipeline.py's _fetch_blocks_v3 does exactly this). Breaking
this rule makes a pid point at the wrong block.
"""

import re

from bs4 import BeautifulSoup, PageElement, Tag

_APPENDIX_HEADING_IDS: set[str] = {
    "External_links",
    "References",
    "Reference",
    "See_also",
    "Notes",
    "Further_reading",
    "Bibliography",
    "Sources",
}
_APPENDIX_HEADING_TEXT: set[str] = {
    "External links",
    "References",
    "Reference",
    "See also",
    "Notes",
    "Further reading",
    "Bibliography",
    "Sources",
}
_HEADING_TAG_RE = re.compile(r"^h[1-6]$")


def _heading_level_from_div(div: Tag) -> int | None:
    """Extract k from `<div class="mw-heading mw-heading{k}">`. Also checks for a nested h1-h6."""
    for cls in div.get("class") or []:
        if cls.startswith("mw-heading") and cls != "mw-heading":
            try:
                return int(cls[len("mw-heading"):])
            except ValueError:
                continue
    h = div.find(_HEADING_TAG_RE)
    if h:
        try:
            return int(h.name[1])
        except (ValueError, IndexError):
            pass
    return None


def _is_appendix_heading_div(div: Tag) -> tuple[bool, str, int]:
    """Check whether div is an appendix heading container (an appendix h# inside `<div class="mw-heading...">`).

    Returns (is_appendix, text, level).
    """
    raw_cls: str | list[str] = div.get("class") or []
    cls: list[str] = [raw_cls] if isinstance(raw_cls, str) else list(raw_cls)
    if not any(c.startswith("mw-heading") for c in cls):
        return False, "", 0
    h = div.find(_HEADING_TAG_RE)
    if not h:
        return False, "", 0
    hid = str(h.get("id") or "").strip()
    htext = h.get_text(strip=True)
    if hid in _APPENDIX_HEADING_IDS or htext in _APPENDIX_HEADING_TEXT:
        level = _heading_level_from_div(div) or 2
        return True, (htext or hid.replace("_", " ")), level
    return False, "", 0


def _strip_appendix_sections(soup: BeautifulSoup) -> list[tuple[int, str]]:
    """Remove appendix sections (`External links` / `References` / `See also`, etc.) in place.

    Returns the appendix headings found, in document order, as
    [(level, text), ...]. Neither this library nor the original
    data_utils/mre_generator3.py actually consumes this return value today
    (both only rely on the in-place removal and discard it at the call
    site). It's a leftover from an earlier "leave appendices as empty
    <section> markers" design that was never implemented. The signature
    stays as is so a future caller can pick it up if needed.

    Supports two HTML shapes:
      1. Parsoid: the top-level section is wrapped in
         `<section aria-labelledby="...">`, so the whole section gets
         decomposed.
      2. non-Parsoid (2wiki/older dumps): no `<section>` tag; `<div
         class="mw-heading">` appears only as a sibling. Walks the body
         container and removes everything from an appendix mw-heading div
         (whose h# id is in the appendix set) up to just before the next
         mw-heading div. asbox/navbox/stub divs disappear naturally in this
         same sibling scan.

    Heading level comes from the `mw-heading{k}` class or the h# tag name.
    """
    appendix: list[tuple[int, str]] = []

    # 1) Parsoid: decompose the whole <section aria-labelledby="External_links">.
    # Real Wikipedia HTML nests <section> tags (a parent section containing child
    # subsections). find_all("section") returns every <section> in one flat list
    # regardless of nesting, so decomposing an outer section in an earlier
    # iteration can leave stale references to its now-decomposed descendants
    # later in that same list (bs4's decompose() recursively clears the whole
    # subtree). Use .decomposed to skip those dead references.
    for sec in list(soup.find_all("section")):
        if getattr(sec, "decomposed", False):
            continue
        labelled_by = str(sec.get("aria-labelledby") or "").strip()
        first_heading = sec.find(_HEADING_TAG_RE)
        htext = first_heading.get_text(strip=True) if first_heading else ""
        is_appendix = (
            labelled_by in _APPENDIX_HEADING_IDS
            or htext in _APPENDIX_HEADING_TEXT
        )
        if not is_appendix:
            continue
        try:
            level = int(first_heading.name[1]) if first_heading else 2
        except (ValueError, IndexError):
            level = 2
        text = htext or labelled_by.replace("_", " ") or "Appendix"
        appendix.append((level, text))
        sec.decompose()

    # 2) non-Parsoid: scan mw-heading div siblings inside the body container
    container = (
        soup.find(id="bodyContent")
        or soup.find("div", class_="mw-parser-output")
        or soup.find("body")
        or soup
    )
    if container is None:
        return appendix

    # Collect mw-heading divs in document order (same walk convention as
    # build_structure_tree). Looking only at top-level headings without
    # recursing into container's children would cover most cases, but we
    # scan descendants too for safety, and delete via a sibling chain.
    for div in list(container.find_all("div", class_=lambda c: c and "mw-heading" in c)):
        # Same reason as (1) above: an earlier iteration's sibling-chain
        # decompose may already have wiped out this div along with its
        # subtree (e.g. the rare case of an mw-heading div nested inside a
        # wrapper between two heading divs).
        if getattr(div, "decomposed", False):
            continue
        is_app, text, level = _is_appendix_heading_div(div)
        if not is_app:
            continue
        appendix.append((level, text))
        # Remove every sibling from this div up to just before the next mw-heading div.
        cur: PageElement | None = div
        while cur is not None:
            nxt = cur.next_sibling
            # Stop once we hit the next mw-heading div; that's where the next section starts.
            if isinstance(nxt, Tag) and nxt.name == "div":
                raw_nxt_cls: str | list[str] = nxt.get("class") or []
                nxt_cls: list[str] = [raw_nxt_cls] if isinstance(raw_nxt_cls, str) else list(raw_nxt_cls)
                if any(c.startswith("mw-heading") for c in nxt_cls):
                    cur.decompose()
                    break
            cur.decompose()
            cur = nxt

    return appendix


# Short-section merge threshold, in characters. Below this, every
# <p>/<ul>/<ol> in a section gets merged into a single <p>, producing one
# summary node instead of many. This keeps Wikipedia stubs and short
# subsections from generating a desc/keys pair per paragraph and adding
# noise. 500 chars is roughly 3-4 paragraphs; below that, splitting into
# individual paragraphs produces near-identical heading/keywords with no
# real information gain.
_SHORT_SECTION_MERGE_THRESHOLD_CHARS = 500


def _consolidate_short_sections(
    soup: BeautifulSoup,
    threshold_chars: int = _SHORT_SECTION_MERGE_THRESHOLD_CHARS,
) -> int:
    """Merge the direct <p>/<ul>/<ol> children of a short <section> into a single <p>, in place.

    If a <section>'s direct <p>/<ul>/<ol> children (excluding references)
    total under threshold_chars of text, they all get removed and replaced
    with one new <p> holding the combined text. Content inside nested
    <section>s is evaluated in its own section's scope, not recursively:
    a parent section doesn't inherit its sub-sections' text. Headings
    (<div class="mw-heading">) are left untouched, so section titles remain.

    Returns the number of sections merged.
    """
    n_merged = 0
    for sec in soup.find_all("section"):
        content_els: list[Tag] = []
        for child in sec.children:
            if not isinstance(child, Tag):
                continue
            if child.name not in ("p", "ul", "ol"):
                continue
            if child.name == "ol" and "references" in (child.get("class") or []):
                continue
            content_els.append(child)
        if len(content_els) < 2:
            continue   # nothing to merge with 0 or 1 element
        texts: list[str] = []
        for el in content_els:
            t = el.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
        if not texts:
            continue
        combined = " ".join(texts)
        if len(combined) >= threshold_chars:
            continue
        new_p = soup.new_tag("p")
        new_p.string = combined
        content_els[0].replace_with(new_p)
        for el in content_els[1:]:
            el.decompose()
        n_merged += 1
    return n_merged
