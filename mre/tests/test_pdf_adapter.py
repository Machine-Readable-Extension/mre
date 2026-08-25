import pypdf
import pytest

from mre.pdf_adapter import (
    _remove_existing_mre_attachment,
    _split_paragraphs,
    build_structure_tree_pdf,
    embed_mre_pdf,
    extract_mre_xml_pdf,
    fetch_pdf,
    mre_xml_exists_pdf,
    parse_pdf,
)

_SAMPLE_MRE_XML = '<mre version="1.0"><metadata><title>T</title></metadata><tree></tree></mre>'

# ─────────────────────────────────────────────
# _split_paragraphs (pure string logic, no PDF involved)
# ─────────────────────────────────────────────


def test_split_paragraphs_blank_line_separates():
    text = "line one\nline two\n\nline three"
    assert _split_paragraphs(text) == ["line one line two", "line three"]


def test_split_paragraphs_wrapped_lines_joined_with_space():
    text = "wrapped\nacross\nthree lines"
    assert _split_paragraphs(text) == ["wrapped across three lines"]


def test_split_paragraphs_multiple_blank_lines_still_one_break():
    text = "first\n\n\n\nsecond"
    assert _split_paragraphs(text) == ["first", "second"]


def test_split_paragraphs_whitespace_only_lines_dont_create_empty_paragraphs():
    text = "\n\n  \nfirst\n\n   \n\nsecond\n\n"
    assert _split_paragraphs(text) == ["first", "second"]


def test_split_paragraphs_empty_input():
    assert _split_paragraphs("") == []


def test_split_paragraphs_strips_leading_trailing_whitespace_per_line():
    text = "  padded line  \n  another  "
    assert _split_paragraphs(text) == ["padded line another"]


def test_split_paragraphs_drops_private_use_area_bullet_glyphs():
    # observed on a real Korean corporate PDF: a bullet drawn through a custom
    # symbol font with no ToUnicode mapping extracts as a raw PUA codepoint
    # (U+F000) instead of any recoverable bullet character.
    text = " 작업장 안전분야\n-1 (조직인력충원) 안전인력 충원"
    assert _split_paragraphs(text) == ["작업장 안전분야 -1 (조직인력충원) 안전인력 충원"]


def test_split_paragraphs_drops_stray_control_characters():
    # \x01 (SOH) isn't a line-boundary character to str.splitlines(), unlike
    # \x0c (form feed) -- so this exercises the mid-line Cc-stripping path
    # specifically, rather than splitlines() already separating it out.
    text = "before\x01after"
    assert _split_paragraphs(text) == ["beforeafter"]


# ─────────────────────────────────────────────
# build_structure_tree_pdf / parse_pdf (full pypdf round-trip)
# ─────────────────────────────────────────────


def test_sample_prose_pdf_splits_into_expected_paragraphs(sample_prose_pdf):
    nodes = build_structure_tree_pdf(sample_prose_pdf)

    assert [n["text"] for n in nodes] == [
        "Quarterly Safety Report",
        "This report summarizes site inspection findings for the third "
        "quarter. All inspected sites passed the baseline safety checklist "
        "with no critical violations recorded during the review period.",
        "Two minor issues were noted at the north entrance: a missing "
        "warning sign and a partially blocked fire exit corridor.",
        "Corrective action has already been taken for both issues, and "
        "a follow-up inspection is scheduled for next month.",
        "Recommendations",
        "Site managers should review the updated signage checklist before "
        "the next quarterly audit, and confirm all fire exits remain clear "
        "throughout the shift, not only during scheduled inspections.",
    ]


def test_sample_prose_pdf_ids_are_sequential_across_pages(sample_prose_pdf):
    nodes = build_structure_tree_pdf(sample_prose_pdf)

    assert [n["id"] for n in nodes] == [f"p{i}" for i in range(1, len(nodes) + 1)]
    assert all(n["type"] == "paragraph" for n in nodes)
    # 4 paragraphs on page 1, 2 on page 2 -- numbering must not restart at the page break.
    assert len(nodes) == 6


def test_no_paragraph_breaks_pdf_falls_back_to_one_paragraph(no_paragraph_breaks_pdf):
    nodes = build_structure_tree_pdf(no_paragraph_breaks_pdf)

    assert len(nodes) == 1
    assert nodes[0] == {
        "type": "paragraph",
        "id": "p1",
        "text": (
            "This document never inserts extra vertical space between lines. "
            "Every line sits at the same fixed leading as every other line. "
            "There is no visual cue anywhere on this page for a paragraph break. "
            "The adapter should therefore treat the whole page as one paragraph."
        ),
    }


def test_parse_pdf_returns_stripped_paragraph_nodes(sample_prose_pdf):
    nodes = parse_pdf(sample_prose_pdf)

    assert nodes
    assert all(set(n.keys()) == {"type", "id", "text"} for n in nodes)
    assert [n["id"] for n in nodes] == [f"p{i}" for i in range(1, len(nodes) + 1)]


# ─────────────────────────────────────────────
# embed / exists / extract_mre_xml_pdf
# ─────────────────────────────────────────────


def test_embed_then_exists_and_reparse_still_works(sample_prose_pdf):
    before = build_structure_tree_pdf(sample_prose_pdf)
    assert not mre_xml_exists_pdf(sample_prose_pdf)

    embed_mre_pdf(sample_prose_pdf, _SAMPLE_MRE_XML)

    assert mre_xml_exists_pdf(sample_prose_pdf)
    # embedding mre.xml must not touch the page content streams
    after = build_structure_tree_pdf(sample_prose_pdf)
    assert after == before


def test_extract_mre_xml_pdf_roundtrips_embedded_content(sample_prose_pdf):
    assert extract_mre_xml_pdf(sample_prose_pdf) is None  # nothing embedded yet

    embed_mre_pdf(sample_prose_pdf, _SAMPLE_MRE_XML)

    assert extract_mre_xml_pdf(sample_prose_pdf) == _SAMPLE_MRE_XML


def test_extract_mre_xml_pdf_missing_file_returns_none(tmp_path):
    assert extract_mre_xml_pdf(tmp_path / "does_not_exist.pdf") is None


def test_reembed_replaces_rather_than_accumulates(sample_prose_pdf):
    embed_mre_pdf(sample_prose_pdf, "<mre>version one</mre>")
    embed_mre_pdf(sample_prose_pdf, "<mre>version two</mre>")

    assert extract_mre_xml_pdf(sample_prose_pdf) == "<mre>version two</mre>"
    reader = pypdf.PdfReader(str(sample_prose_pdf))
    assert reader.attachments["mre.xml"] == [b"<mre>version two</mre>"]


def test_reembed_preserves_unrelated_attachments(sample_prose_pdf):
    writer = pypdf.PdfWriter(clone_from=str(sample_prose_pdf))
    writer.add_attachment("other_file.txt", b"unrelated, should survive")
    writer.write(str(sample_prose_pdf))

    embed_mre_pdf(sample_prose_pdf, _SAMPLE_MRE_XML)
    embed_mre_pdf(sample_prose_pdf, _SAMPLE_MRE_XML)  # a second embed to exercise pruning

    reader = pypdf.PdfReader(str(sample_prose_pdf))
    assert dict(reader.attachments) == {
        "mre.xml": [_SAMPLE_MRE_XML.encode("utf-8")],
        "other_file.txt": [b"unrelated, should survive"],
    }


def test_remove_existing_mre_attachment_noop_when_none_present():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    _remove_existing_mre_attachment(writer)  # must not raise


def test_embed_mre_pdf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        embed_mre_pdf(tmp_path / "does_not_exist.pdf", _SAMPLE_MRE_XML)


# ─────────────────────────────────────────────
# fetch_pdf
# ─────────────────────────────────────────────


def test_fetch_pdf_matches_extracted_paragraph_text(sample_prose_pdf):
    nodes = build_structure_tree_pdf(sample_prose_pdf)
    for node in nodes:
        assert fetch_pdf(sample_prose_pdf, node["id"]) == node["text"]


def test_fetch_pdf_full_concatenates_all_paragraphs(sample_prose_pdf):
    full = fetch_pdf(sample_prose_pdf, "full")
    nodes = build_structure_tree_pdf(sample_prose_pdf)
    for node in nodes:
        assert node["text"] in full


def test_fetch_pdf_unknown_id_returns_empty_string(sample_prose_pdf):
    assert fetch_pdf(sample_prose_pdf, "p999") == ""
