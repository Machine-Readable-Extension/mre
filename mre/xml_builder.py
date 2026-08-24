"""Pure string assembly of the final MRE XML.

Combines the LLM output (summary + parallel headings/keywords arrays)
with the original node tree into an ``<mre>`` document. Makes no LLM
calls of its own.
"""

from __future__ import annotations

from xml.sax.saxutils import escape as xml_escape


def build_mre_xml(
    llm_data: dict,
    original_nodes: list[dict],
    title: str = "Untitled",
    *,
    generator: str | None = None,
    generator_fingerprint: str | None = None,
) -> str:
    """Assemble minimal MRE XML — no <tags>, no lang attribute, no <resources>.

    Structure:
        <mre version="1.0" generator="wikipedia" generator-fingerprint="a3f9c21b">
          <metadata>
            <title>...</title>
            <summary>...</summary>
          </metadata>
          <tree>
            <node id="pN">
              <desc>LLM heading</desc>
              <keys>LLM keywords</keys>
            </node>
          </tree>
        </mre>

    Paragraph granularity — the only granularity this library supports —
    is always flat; no ``<section>`` elements are produced. Heading-type
    nodes are exposed as context to the LLM prompt (``build_user_prompt``)
    but are skipped when assembling the final XML tree.

    ``generator``/``generator_fingerprint`` identify which site adapter
    produced this document, so that ``fetch_block()`` can later detect
    whether the adapter's parsing logic has changed since generation (see
    ``compute_adapter_fingerprint()`` in ``html_site_adapter.py``). Both
    attributes are recorded on the ``<mre>`` root tag only if both
    arguments are given; if either is ``None`` (formats with no adapter
    concept, like hwpx/docx, or a caller that omits them), both attributes
    are left out.

    Args:
        llm_data: LLM output — ``summary``, ``headings``, ``keywords``.
        original_nodes: The document's heading/paragraph nodes, in order.
        title: Document title.
        generator: Name of the site adapter that produced this document.
        generator_fingerprint: Fingerprint of that adapter's parsing logic.

    Returns:
        The assembled ``<mre>`` XML as a string.
    """
    title_esc = xml_escape(title)
    # Maps the headings/keywords arrays to input paragraphs by position.
    # On a length mismatch, paragraphs past the shorter array fall back
    # to an empty value.
    summary_esc  = xml_escape((llm_data.get("summary") or "").strip())
    llm_headings: list[str] = list(llm_data.get("headings", []))
    llm_keywords: list[str] = list(llm_data.get("keywords", []))

    pad = "    "
    tree_lines: list[str] = []

    para_idx = 0  # paragraph counter — index into the LLM arrays
    for node in original_nodes:
        ntype = node.get("type", "paragraph")
        if ntype == "heading":
            continue  # flat tree — headings are prompt context only, not reflected in the XML
        safe_id = xml_escape(node["id"])
        heading = llm_headings[para_idx] if para_idx < len(llm_headings) else ""
        kws     = llm_keywords[para_idx] if para_idx < len(llm_keywords) else ""
        desc = xml_escape(heading)
        keys = xml_escape(kws)
        para_idx += 1
        tree_lines.append(f'{pad}<node id="{safe_id}">')
        if desc:
            tree_lines.append(f"{pad}  <desc>{desc}</desc>")
        if keys:
            tree_lines.append(f"{pad}  <keys>{keys}</keys>")
        tree_lines.append(f"{pad}</node>")

    tree_body = "\n".join(tree_lines)

    # Assembled explicitly (no textwrap.dedent): combining dedent with a
    # multiline {tree_body} would only strip the source's leading
    # whitespace from tree_body's first line, breaking its indentation.
    # tree_body already carries its own pad+nesting indentation, so it's
    # inserted as-is at column 0.
    summary_xml = (
        f'    <summary>\n      {summary_esc}\n    </summary>\n'
        if summary_esc else ''
    )
    root_attrs = '<mre version="1.0"'
    if generator is not None and generator_fingerprint is not None:
        root_attrs += f' generator="{xml_escape(generator)}"'
        root_attrs += f' generator-fingerprint="{xml_escape(generator_fingerprint)}"'
    root_attrs += '>'
    mre = (
        f'{root_attrs}\n'
        f'  <metadata>\n'
        f'    <title>{title_esc}</title>\n'
        f'{summary_xml}'
        f'  </metadata>\n'
        f'  <tree>\n'
        f'{tree_body}\n'
        f'  </tree>\n'
        f'</mre>'
    )
    return mre
