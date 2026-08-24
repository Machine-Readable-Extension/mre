"""Wikipedia appendix-section filtering and short-section merging.

Cleans up an HTML document before it's sent to the LLM.

**Warning (single source of truth)**: paragraph `id`s (``pN``) are
assigned at generation time based on the order in which this module's
transforms (appendix removal, short-section merging) leave the soup. The
final document embeds the original, *untransformed* HTML, so fetching by
`id` later must re-apply these same two functions, in the same order,
before walking paragraphs — only then does the walk order at fetch time
match the `id` order assigned at generation time. Break this and a `pid`
will point at the wrong block.
"""

from __future__ import annotations

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
}
_APPENDIX_HEADING_TEXT: set[str] = {
    "External links",
    "References",
    "Reference",
    "See also",
    "Notes",
    "Further reading",
    "Bibliography",
}
_HEADING_TAG_RE = re.compile(r"^h[1-6]$")


def _heading_level_from_div(div: Tag) -> int | None:
    """Extract k from `<div class="mw-heading mw-heading{k}">`; also checks for a nested h1-h6."""
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
    """Determine whether ``div`` is an appendix heading container
    (`<div class="mw-heading...">` wrapping an appendix h#).

    Returns:
        A ``(is_appendix, text, level)`` tuple.
    """
    raw_cls: str | list[str] = div.get("class") or []
    cls = raw_cls if isinstance(raw_cls, list) else [raw_cls]
    if not any(c.startswith("mw-heading") for c in cls):
        return False, "", 0
    h = div.find(_HEADING_TAG_RE)
    if not h:
        return False, "", 0
    raw_id = h.get("id")
    hid = (raw_id if isinstance(raw_id, str) else "").strip()
    htext = h.get_text(strip=True)
    if hid in _APPENDIX_HEADING_IDS or htext in _APPENDIX_HEADING_TEXT:
        level = _heading_level_from_div(div) or 2
        return True, (htext or hid.replace("_", " ")), level
    return False, "", 0


def _strip_appendix_sections(soup: BeautifulSoup) -> list[tuple[int, str]]:
    """Remove appendix sections (``External links``/``References``/``See also``/etc.) in-place.

    Supports two HTML shapes:
      1. Parsoid: top-level sections are wrapped in
         `<section aria-labelledby="…">` — the whole section is decomposed.
      2. non-Parsoid (older dumps): no `<section>` tag; `<div class="mw-heading">`
         appears as a sibling instead. The body container is walked, and
         starting from an appendix `mw-heading` div, every node up to (but
         not including) the next `mw-heading` div is removed. Stray
         asbox/navbox/stub divs disappear naturally as part of this same
         sibling scan.

    Heading level is read from the `mw-heading{k}` class or the h# tag name.

    Returns:
        The appendix headings found, in document order, as
        ``[(level, text), ...]``. Not currently consumed by any caller in
        this library — kept for a possible future use (e.g. representing
        removed appendix sections as placeholder markers).
    """
    appendix: list[tuple[int, str]] = []

    # 1) Parsoid: delete <section aria-labelledby="External_links"> wholesale
    for sec in list(soup.find_all("section")):
        raw_labelled_by = sec.get("aria-labelledby")
        labelled_by = (raw_labelled_by if isinstance(raw_labelled_by, str) else "").strip()
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
    # build_structure_tree). Scanning descendants rather than only
    # top-level children is a bit more than most cases need, but it's the
    # safe way to locate every heading before removing its sibling chain.
    for div in list(container.find_all("div", class_=lambda c: c and "mw-heading" in c)):
        is_app, text, level = _is_appendix_heading_div(div)
        if not is_app:
            continue
        appendix.append((level, text))
        # Remove every sibling from this div up to (not including) the next mw-heading div.
        cur: PageElement | None = div
        while cur is not None:
            nxt = cur.next_sibling
            # Stop at the next mw-heading div — that's where the next section starts.
            if isinstance(nxt, Tag) and nxt.name == "div":
                raw_nxt_cls: str | list[str] = nxt.get("class") or []
                nxt_cls = raw_nxt_cls if isinstance(raw_nxt_cls, list) else [raw_nxt_cls]
                if any(c.startswith("mw-heading") for c in nxt_cls):
                    cur.decompose()
                    break
            cur.decompose()
            cur = nxt

    return appendix


# Short-section merge threshold, in characters. Sections under this size
# have all their <p>/<ul>/<ol> elements merged into a single <p>, so only
# one summary node gets generated for them — this keeps short Wikipedia
# stubs/subsections from producing one near-identical, low-signal
# desc/keys pair per paragraph. 500 chars is roughly 3-4 paragraphs' worth
# of text; below that, splitting into separate paragraphs would produce
# headings/keywords too similar to each other to carry real information.
_SHORT_SECTION_MERGE_THRESHOLD_CHARS = 500


def _consolidate_short_sections(
    soup: BeautifulSoup,
    threshold_chars: int = _SHORT_SECTION_MERGE_THRESHOLD_CHARS,
) -> int:
    """Merge a short `<section>`'s direct `<p>`/`<ul>`/`<ol>` children into one `<p>` (in-place).

    For each `<section>`, if the combined text of its direct `<p>`/`<ul>`/`<ol>`
    children (excluding a references list) is under ``threshold_chars``,
    those elements are removed and replaced with a single new `<p>`
    holding their combined text. Nested `<section>` content is judged
    separately, in its own section's scope — this isn't recursive, and a
    section's text is never inherited from its sub-sections. Headings
    (`<div class="mw-heading">`) are left untouched, so section titles
    remain in place.

    Args:
        soup: The document to modify in-place.
        threshold_chars: Character threshold below which a section's
            paragraphs get merged.

    Returns:
        The number of sections merged.
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
            continue   # already 0 or 1 — nothing to merge
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
