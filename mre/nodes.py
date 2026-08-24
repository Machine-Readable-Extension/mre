"""Format-agnostic node normalization shared by the html/hwpx/docx adapters."""

from __future__ import annotations


def strip_to_text_nodes(nodes: list[dict]) -> list[dict]:
    """Extract heading and paragraph nodes for the LLM, preserving order.

    Args:
        nodes: Raw nodes as produced by an adapter's ``extract()``.

    Returns:
        A list of ``{"type": "heading", "level", "text"}`` and
        ``{"type": "paragraph", "id", "text"}`` dicts, in the input order.
    """
    result = []
    for node in nodes:
        ntype = node.get("type", "paragraph")
        if ntype == "heading":
            result.append({"type": "heading", "level": node["level"], "text": node["text"]})
        else:
            result.append({"type": "paragraph", "id": node["id"], "text": node["text"]})
    return result
