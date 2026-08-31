from __future__ import annotations

"""
Format-agnostic node normalization: pieces shared by the html/hwpx/docx/pdf adapters.

Ported from the same-named functions in data_utils/mre_generator.py (v1)
into this library's distribution boundary.
"""

import re

_PID_RE = re.compile(r"^[A-Za-z]*(\d+)$")


def strip_to_text_nodes(nodes: list[dict]) -> list[dict]:
    """
    Extract heading and paragraph nodes, in order, for sending to the LLM.
    - heading: type, level, text
    - paragraph: type, id, text
    """
    result = []
    for node in nodes:
        ntype = node.get("type", "paragraph")
        if ntype == "heading":
            result.append({"type": "heading", "level": node["level"], "text": node["text"]})
        else:
            result.append({"type": "paragraph", "id": node["id"], "text": node["text"]})
    return result


def fetch_paragraph_by_id(nodes: list[dict], node_id: str) -> str:
    """Fetch the full text of the node_id paragraph from the node list produced by extract().

    id="full" returns the whole document's text, paragraphs separated by a
    blank line. An id that isn't found returns an empty string rather than
    raising, matching html_site_adapter.fetch_block()'s contract.

    Shared by hwpx/docx (mre.opc_adapter) and pdf (mre.pdf_adapter): both
    formats implement fetch as "re-run extract() on the path-based document,
    then index into it" (the single-source-of-truth principle where
    generation time and fetch time use the exact same function, see the
    mre.appendix module docstring), so the id parsing/lookup logic itself is
    format-agnostic."""
    para_nodes = [n for n in nodes if n.get("type") == "paragraph"]
    if node_id == "full":
        return "\n\n".join(n["text"] for n in para_nodes)
    m = _PID_RE.match(node_id)
    if not m:
        return ""
    idx = int(m.group(1))
    if 1 <= idx <= len(para_nodes):
        return para_nodes[idx - 1]["text"]
    return ""
