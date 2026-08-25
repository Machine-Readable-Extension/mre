import zipfile
from dataclasses import replace

import pytest

from mre import DocFormat, embed_mre_opc, fetch_opc, get_opc_adapter, parse_opc
from mre.html_site_adapter import FetchNotSupportedError
from mre.opc_adapter import (
    _mre_xml_exists_in_zip,
    build_structure_tree_docx,
    build_structure_tree_hwpx,
)

# sample_docx / sample_hwpx fixtures live in conftest.py (shared with test_agent.py).


# ─────────────────────────────────────────────
# docx extract
# ─────────────────────────────────────────────

def test_docx_extract_headings_and_paragraphs(sample_docx):
    nodes = build_structure_tree_docx(sample_docx)
    headings = [n for n in nodes if n["type"] == "heading"]
    paras = [n for n in nodes if n["type"] == "paragraph"]

    assert [(h["level"], h["text"]) for h in headings] == [
        (1, "Sample Report"),
        (2, "Background"),
    ]
    # table cell text and the empty trailing paragraph must NOT appear
    assert not any("table cell" in p["text"] for p in paras)
    assert all(p["text"].strip() for p in paras)
    assert [p["id"] for p in paras] == ["p1", "p2", "p3"]


def test_docx_paragraph_order_is_preserved_around_table(sample_docx):
    nodes = build_structure_tree_docx(sample_docx)
    texts = [n["text"] for n in nodes if n["type"] == "paragraph"]
    assert texts[-1].startswith("This is the third paragraph")


# ─────────────────────────────────────────────
# hwpx extract
# ─────────────────────────────────────────────

def test_hwpx_short_paragraph_coalesces_into_next(sample_hwpx):
    nodes = build_structure_tree_hwpx(sample_hwpx)
    paras = [n for n in nodes if n["type"] == "paragraph"]
    assert len(paras) == 19
    # the short "보도자료" label paragraph got merged into the "보도시점..."
    # paragraph that follows it, instead of staying a standalone node
    assert paras[0]["text"].startswith("보도자료")
    assert "보도시점" in paras[0]["text"]


def test_hwpx_nested_p_absorbed_not_double_counted(sample_hwpx):
    nodes = build_structure_tree_hwpx(sample_hwpx)
    paras = [n for n in nodes if n["type"] == "paragraph"]
    # the contact-info block is a table (<hp:tbl>/<hp:subList>/<hp:p>) -- its
    # nested <hp:p> cell text is absorbed into one outer paragraph's text
    # (concatenated with no separators, exactly like the real HWPX gives it),
    # and must not produce its own separate top-level node.
    last = paras[-1]
    assert "담당 부서" in last["text"]
    assert "전한성" in last["text"] and "문지환" in last["text"]
    assert last["id"] == "p19"


# ─────────────────────────────────────────────
# strip / parse_opc
# ─────────────────────────────────────────────

def test_parse_opc_matches_strip_to_text_nodes_shape(sample_docx):
    stripped = parse_opc(sample_docx, DocFormat.DOCX)
    for node in stripped:
        assert "type" in node and "text" in node
        if node["type"] == "paragraph":
            assert "id" in node
        else:
            assert "level" in node


# ─────────────────────────────────────────────
# embed / exists
# ─────────────────────────────────────────────

@pytest.mark.parametrize("fmt,fixture_name", [(DocFormat.DOCX, "sample_docx"), (DocFormat.HWPX, "sample_hwpx")])
def test_embed_then_exists_and_reparse_still_works(fmt, fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    assert not _mre_xml_exists_in_zip(path)

    embed_mre_opc(path, "<mre version=\"1.0\"><metadata><title>T</title></metadata><tree></tree></mre>", fmt)

    assert _mre_xml_exists_in_zip(path)
    with zipfile.ZipFile(path) as zf:
        assert "mre.xml" in zf.namelist()
    # embedding mre.xml must not corrupt the original document parts
    nodes_after = get_opc_adapter(fmt).extract(path)
    assert any(n["type"] == "paragraph" for n in nodes_after)


@pytest.mark.parametrize("fmt,fixture_name", [(DocFormat.DOCX, "sample_docx"), (DocFormat.HWPX, "sample_hwpx")])
def test_extract_mre_xml_opc_roundtrips_embedded_content(fmt, fixture_name, request):
    from mre import extract_mre_xml_opc

    path = request.getfixturevalue(fixture_name)
    assert extract_mre_xml_opc(path) is None  # nothing embedded yet

    mre_xml = "<mre version=\"1.0\"><metadata><title>T</title></metadata><tree></tree></mre>"
    embed_mre_opc(path, mre_xml, fmt)

    assert extract_mre_xml_opc(path) == mre_xml


def test_extract_mre_xml_opc_missing_file_returns_none(tmp_path):
    from mre import extract_mre_xml_opc

    assert extract_mre_xml_opc(tmp_path / "does_not_exist.docx") is None


# ─────────────────────────────────────────────
# fetch
# ─────────────────────────────────────────────

def test_docx_fetch_matches_extracted_paragraph_text(sample_docx):
    nodes = build_structure_tree_docx(sample_docx)
    for node in nodes:
        if node["type"] != "paragraph":
            continue
        assert fetch_opc(sample_docx, node["id"], DocFormat.DOCX) == node["text"]


def test_docx_fetch_full_concatenates_all_paragraphs(sample_docx):
    full = fetch_opc(sample_docx, "full", DocFormat.DOCX)
    nodes = build_structure_tree_docx(sample_docx)
    for node in nodes:
        if node["type"] == "paragraph":
            assert node["text"] in full


def test_docx_fetch_unknown_id_returns_empty_string(sample_docx):
    assert fetch_opc(sample_docx, "p999", DocFormat.DOCX) == ""


def test_hwpx_fetch_matches_extracted_paragraph_text(sample_hwpx):
    nodes = build_structure_tree_hwpx(sample_hwpx)
    for node in nodes:
        assert fetch_opc(sample_hwpx, node["id"], DocFormat.HWPX) == node["text"]


def test_hwpx_fetch_full_concatenates_all_paragraphs(sample_hwpx):
    full = fetch_opc(sample_hwpx, "full", DocFormat.HWPX)
    nodes = build_structure_tree_hwpx(sample_hwpx)
    for node in nodes:
        assert node["text"] in full


def test_fetch_opc_raises_when_adapter_has_no_fetch(sample_hwpx, monkeypatch):
    import mre.opc_adapter as opc_mod

    no_fetch_adapter = replace(opc_mod._REGISTRY[DocFormat.HWPX], fetch=None)
    monkeypatch.setitem(opc_mod._REGISTRY, DocFormat.HWPX, no_fetch_adapter)

    with pytest.raises(FetchNotSupportedError):
        fetch_opc(sample_hwpx, "p1", DocFormat.HWPX)
