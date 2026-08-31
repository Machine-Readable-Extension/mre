from __future__ import annotations

"""
A utility for reading the raw <mre> header back out of embedded HTML in one
piece. The counterpart to fetch_block(), which reads only a single
paragraph. No site-adapter dispatch is needed here: the
<script type="application/mre+xml"> tag's location and format are the same
regardless of which site produced the document, so this lives separately
from html_site_adapter.py's per-site registry.

Ported from core/mre.py's MREParser.extract_mre into this library's
distribution boundary.
"""

import re

from bs4 import BeautifulSoup

# Strips a legacy <resources> section. mre's own generation schema
# (xml_builder.build_mre_xml) never produces <resources>, but stripping it
# keeps this compatible with documents made by other generators (same
# principle as html_site_adapter._wiki_fetch).
_RESOURCES_RE = re.compile(r"\s*<resources>.*?</resources>", re.DOTALL)


def extract_mre_xml(html: str) -> str | None:
    """Return the raw <mre>...</mre> content from embedded html. None if there's no MRE.

    Used anywhere an agent needs to read the full header (title/summary/tree)
    before navigating a document — e.g. before mre.agent's progressive loop
    builds a metadata-only view of candidate documents, or before showing the
    full tree of a document picked out by expand_document."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"type": "application/mre+xml"})
    if script is None:
        return None
    raw = script.string
    if raw is None:
        return None
    mre_text = _RESOURCES_RE.sub("", raw.strip())
    return mre_text.strip()
