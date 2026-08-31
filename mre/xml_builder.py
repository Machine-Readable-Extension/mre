from __future__ import annotations

"""
Minimal MRE XML assembly. Combines the LLM output (a summary plus parallel
headings/keywords arrays) with the original node tree to build an <mre>
document. Pure string assembly, no LLM calls here.

Ported from data_utils/mre_generator3.py's build_mre_xml2 into this
library's distribution boundary (renamed to build_mre_xml: the "2" suffix
was mre_generator3.py's way of distinguishing it from v1, and since this
library has no v1, the suffix is just noise here).
"""

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

    Paragraph granularity (the only granularity this library supports) is
    always flat: no <section> elements are generated. Heading-type nodes
    are exposed as context to the LLM prompt input (build_user_prompt) but
    skipped when assembling the final XML tree. This function is a direct
    port of data_utils/mre_generator3.py's build_mre_xml2
    (paragraph-granularity only), renamed to build_mre_xml; mre_generator3.py
    is the reference implementation. The generator/generator_fingerprint
    parameters, however, are an extension unique to this library and have
    no counterpart in mre_generator3.py, which has no adapter-plugin concept
    at all (see html_site_adapter.py's compute_adapter_fingerprint(): it
    lets a later fetch identify which adapter produced this document, to
    detect whether the adapter's parsing logic changed in between).

    When both generator and generator_fingerprint are given, they're
    recorded as generator/generator-fingerprint attributes on the <mre>
    root tag. If either is None (formats with no adapter concept, like
    hwpx/docx, or a caller that simply didn't pass them), both attributes
    are omitted.
    """
    title_esc = xml_escape(title)
    # Map the headings/keywords parallel arrays to input paragraphs by position.
    # On a length mismatch, map as far as the shorter array reaches and fall
    # back to an empty value for the rest.
    summary_esc  = xml_escape((llm_data.get("summary") or "").strip())
    llm_headings: list[str] = list(llm_data.get("headings", []))
    llm_keywords: list[str] = list(llm_data.get("keywords", []))

    pad = "    "
    tree_lines: list[str] = []

    para_idx = 0  # paragraph counter, indexes into the LLM arrays
    for node in original_nodes:
        ntype = node.get("type", "paragraph")
        if ntype == "heading":
            continue  # flat tree: headings are prompt context only, not reflected in the XML
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

    # Assembled explicitly; do not use textwrap.dedent. Combining dedent with
    # a multi-line {tree_body} misaligned the indent: only tree's first line
    # got the source's leading whitespace stripped. tree_body already carries
    # its own pad+nesting indent, so it's inserted straight at column 0.
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
