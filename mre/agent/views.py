from __future__ import annotations

"""
The transform that builds the metadata-only view: stage 1 of progressive two-stage disclosure.

Of core/ablations.py's ten ablation-experiment modes
(desc_only/keys_only/no_metadata/...), only this one, which strips out the
entire <tree>, is actually used by the progressive loop, so only this one
was ported; the rest are paper-ablation-experiment-only and out of scope
for this subpackage. Stage 2 (full disclosure once a document is picked)
needs no separate transform: mre's <mre> schema is already a flat tree with
no <section> nesting, so showing the original mre_xml as is suffices.
"""

import re

_TREE_RE = re.compile(r"\s*<tree\b[^>]*>.*?</tree>", re.DOTALL)


def metadata_view(mre_xml: str) -> str:
    """Strip out the entire <tree> (paragraph map), leaving only <metadata> (title+summary).

    The first of the two disclosure stages — every candidate document is shown
    cheaply through this view first, and once the agent picks out a specific
    document (expand_document), only that document is shown as the original
    mre_xml (full <tree> included)."""
    return _TREE_RE.sub("", mre_xml).strip()
