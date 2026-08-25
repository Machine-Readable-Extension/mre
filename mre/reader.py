"""Read a whole ``<mre>`` header back out of MRE-embedded HTML.

This is the counterpart to ``fetch_block()`` (which returns a single
paragraph): it needs no site-adapter dispatch, since the
``<script type="application/mre+xml">`` tag's location and format is the
same regardless of which site adapter produced it. That's why it lives
here rather than in the per-site registry in ``html_site_adapter.py``.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

# Strips a legacy <resources> section. mre's own generator
# (xml_builder.build_mre_xml) never emits one, but stripping it keeps this
# reader compatible with documents produced by other generators too (see the
# same handling in html_site_adapter._wiki_fetch).
_RESOURCES_RE = re.compile(r"\s*<resources>.*?</resources>", re.DOTALL)


def extract_mre_xml(html: str) -> str | None:
    """Return the raw ``<mre>...</mre>`` block embedded in ``html``.

    Used anywhere an agent needs the whole header (title, summary, tree)
    before it starts exploring a document — e.g. ``mre.agent``'s
    progressive loop building a metadata-only view of candidate documents,
    or showing the full tree once a document has been picked via
    ``expand_document``.

    Args:
        html: MRE-embedded HTML.

    Returns:
        The header's raw XML text, or ``None`` if the document has no
        embedded MRE header.
    """
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"type": "application/mre+xml"})
    if script is None:
        return None
    raw = script.string
    if raw is None:
        return None
    mre_text = _RESOURCES_RE.sub("", raw.strip())
    return mre_text.strip()
