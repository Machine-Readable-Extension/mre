"""Builds the metadata-only view used by stage one of progressive disclosure.

Stage two (showing a document's full tree once it's been picked) needs no
separate transform — MRE's ``<mre>`` schema has no ``<section>`` nesting to
begin with, so the original ``mre_xml`` can be shown as-is.
"""

from __future__ import annotations

import re

_TREE_RE = re.compile(r"\s*<tree\b[^>]*>.*?</tree>", re.DOTALL)


def metadata_view(mre_xml: str) -> str:
    """Strip ``<tree>`` (the paragraph map), leaving only ``<metadata>``.

    This is stage one of progressive disclosure: every candidate document
    is shown through this cheap view first, and only once the agent picks
    a specific document (``expand_document``) does that document's full
    ``mre_xml`` — tree included — get shown.

    Args:
        mre_xml: A full ``<mre>`` document.

    Returns:
        The same document with its ``<tree>`` element removed.
    """
    return _TREE_RE.sub("", mre_xml).strip()
