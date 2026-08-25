import struct
import unicodedata
import zlib

import olefile
import pytest

from mre.hwp_adapter import (
    HWPTAG_PARA_TEXT,
    _decode_para_text,
    _is_compressed,
    _parse_records,
    build_structure_tree_hwp,
    parse_hwp,
)
from mre.tests._ole2_builder import build_ole2_hwp

# ─────────────────────────────────────────────
# record-level byte helpers (mirror what a real HWP5 BodyText section
# contains, so _parse_records/_decode_para_text can be tested without any
# OLE2 container at all)
# ─────────────────────────────────────────────


def _record(tag_id: int, payload: bytes, level: int = 0) -> bytes:
    size = len(payload)
    if size < 0xFFF:
        header = (size << 20) | (level << 10) | tag_id
        return struct.pack("<I", header) + payload
    header = (0xFFF << 20) | (level << 10) | tag_id
    return struct.pack("<I", header) + struct.pack("<I", size) + payload


def _control(code: int, extra_wchars: int) -> bytes:
    return struct.pack("<H", code) + b"\x00\x00" * extra_wchars


def _text(s: str) -> bytes:
    return s.encode("utf-16-le")


def _para_text_record(payload: bytes) -> bytes:
    return _record(HWPTAG_PARA_TEXT, payload)


# ─────────────────────────────────────────────
# _decode_para_text
# ─────────────────────────────────────────────

def test_decode_plain_text():
    assert _decode_para_text(_text("hello world")) == "hello world"


def test_decode_korean_text_untouched():
    assert _decode_para_text(_text("안녕하세요")) == "안녕하세요"


def test_decode_tab_becomes_tab_char_and_skips_its_parameter():
    payload = _text("Name:") + _control(9, 7) + _text("John")
    assert _decode_para_text(payload) == "Name:\tJohn"


def test_decode_line_break_and_paragraph_break_become_newline():
    payload = _text("line1") + _control(10, 0) + _text("line2") + _control(13, 0) + _text("line3")
    assert _decode_para_text(payload) == "line1\nline2\nline3"


def test_decode_unknown_inline_control_skips_default_seven_wchars_silently():
    # code 6 (field start) isn't in the special table -> default 7-wchar skip,
    # no character emitted for it.
    payload = _text("before") + _control(6, 7) + _text("after")
    assert _decode_para_text(payload) == "beforeafter"


def test_decode_strips_leftover_c0_control_as_safety_net():
    # simulates the extra-wchars table being wrong for some code: a stray
    # control char survives decoding. It must not corrupt/crash, just get
    # filtered by the C0 safety net.
    payload = _text("a") + struct.pack("<H", 0x01) + _text("b")
    # code 1 has no declared extra size (defaults to 7), so this payload is
    # deliberately short (no 7-wchar param actually follows) to exercise the
    # "ran past the end" -> loop just stops naturally, nothing crashes.
    result = _decode_para_text(payload)
    assert "\x01" not in result


# ─────────────────────────────────────────────
# _parse_records
# ─────────────────────────────────────────────

def test_parse_records_extracts_para_text_in_order():
    data = _para_text_record(_text("first")) + _para_text_record(_text("second"))
    assert _parse_records(data) == ["first", "second"]


def test_parse_records_ignores_non_para_text_tags():
    other = _record(tag_id=0x50, payload=b"\x00" * 8)  # arbitrary non-PARA_TEXT tag
    data = other + _para_text_record(_text("kept"))
    assert _parse_records(data) == ["kept"]


def test_parse_records_handles_extended_size_sentinel():
    long_text = "x" * 3000  # payload > 0xFFF (4095) bytes forces the extended-size path
    data = _para_text_record(_text(long_text))
    assert _parse_records(data) == [long_text]


def test_parse_records_stops_gracefully_on_truncated_trailing_bytes():
    data = _para_text_record(_text("ok")) + b"\x01\x02"  # incomplete trailing header
    assert _parse_records(data) == ["ok"]


def test_parse_records_empty_input():
    assert _parse_records(b"") == []


def test_parse_records_skips_empty_para_text_records():
    data = _para_text_record(b"") + _para_text_record(_text("real"))
    assert _parse_records(data) == ["real"]


# ─────────────────────────────────────────────
# build_structure_tree_hwp / parse_hwp (full OLE2 round-trip)
# ─────────────────────────────────────────────


def _file_header(compressed: bool) -> bytes:
    flags = 1 if compressed else 0
    return b"\x00" * 36 + struct.pack("<I", flags)


def _compress(data: bytes) -> bytes:
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return co.compress(data) + co.flush()


def test_build_structure_tree_hwp_uncompressed_single_paragraph(tmp_path):
    section = _para_text_record(_text("only paragraph"))
    path = tmp_path / "sample.hwp"
    path.write_bytes(build_ole2_hwp([section], _file_header(compressed=False)))

    nodes = build_structure_tree_hwp(path)

    assert nodes == [{"type": "paragraph", "id": "p1", "text": "only paragraph"}]


def test_build_structure_tree_hwp_multiple_paragraphs_numbered_in_order(tmp_path):
    section = (
        _para_text_record(_text("first"))
        + _para_text_record(_text("second"))
        + _para_text_record(_text("third"))
    )
    path = tmp_path / "sample.hwp"
    path.write_bytes(build_ole2_hwp([section], _file_header(compressed=False)))

    nodes = build_structure_tree_hwp(path)

    assert [n["id"] for n in nodes] == ["p1", "p2", "p3"]
    assert [n["text"] for n in nodes] == ["first", "second", "third"]
    assert all(n["type"] == "paragraph" for n in nodes)


def test_build_structure_tree_hwp_compressed_section_round_trips(tmp_path):
    section = _compress(
        _para_text_record(_text("compressed paragraph one"))
        + _para_text_record(_text("compressed paragraph two"))
    )
    path = tmp_path / "sample.hwp"
    path.write_bytes(build_ole2_hwp([section], _file_header(compressed=True)))

    nodes = build_structure_tree_hwp(path)

    assert [n["text"] for n in nodes] == ["compressed paragraph one", "compressed paragraph two"]


def test_build_structure_tree_hwp_paragraphs_continue_numbering_across_sections(tmp_path):
    section0 = _para_text_record(_text("section0 para"))
    section1 = _para_text_record(_text("section1 para"))
    path = tmp_path / "sample.hwp"
    path.write_bytes(build_ole2_hwp([section0, section1], _file_header(compressed=False)))

    nodes = build_structure_tree_hwp(path)

    assert [n["id"] for n in nodes] == ["p1", "p2"]
    assert [n["text"] for n in nodes] == ["section0 para", "section1 para"]


def test_build_structure_tree_hwp_korean_text_with_tab_and_line_break(tmp_path):
    payload = _text("담당자:") + _control(9, 7) + _text("전한성") + _control(10, 0) + _text("문의")
    section = _para_text_record(payload)
    path = tmp_path / "sample.hwp"
    path.write_bytes(build_ole2_hwp([section], _file_header(compressed=False)))

    nodes = build_structure_tree_hwp(path)

    assert nodes[0]["text"] == "담당자:\t전한성\n문의"


def test_parse_hwp_returns_stripped_paragraph_nodes(tmp_path):
    section = _para_text_record(_text("hello"))
    path = tmp_path / "sample.hwp"
    path.write_bytes(build_ole2_hwp([section], _file_header(compressed=False)))

    nodes = parse_hwp(path)

    assert nodes == [{"type": "paragraph", "id": "p1", "text": "hello"}]


# ─────────────────────────────────────────────
# real-world fixtures: genuine compressed .hwp files, not synthetic ones
# ─────────────────────────────────────────────
# _ole2_builder.py fixtures prove the record/control-char logic is right for
# whatever we hand-craft, but real documents are the only thing that exercises
# the actual compression ratio and the inline-control mix a real HWP writer
# produces -- which is exactly the case the module docstring flags as
# under-validated. Two structurally different real files, so a fix that only
# happens to work on one document shape doesn't pass unnoticed:
#   - animal_shelter_hwp: a table (county/shelter-name/address/phone cells,
#     tabs, a parenthetical aside in its own cell)
#   - construction_safety_cost_notice_hwp: prose/legal-citation text (long
#     sentences, a revision-history list, CJK bracket punctuation ｢｣「」·․)

_REAL_HWP_FIXTURES = ["animal_shelter_hwp", "construction_safety_cost_notice_hwp"]


@pytest.mark.parametrize("fixture_name", _REAL_HWP_FIXTURES)
def test_real_hwp_no_leftover_control_or_private_use_chars(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    nodes = build_structure_tree_hwp(path)

    assert nodes  # sanity: extraction actually produced something
    for n in nodes:
        for ch in n["text"]:
            category = unicodedata.category(ch)
            assert category[0] != "C" or ch in "\n\t", (n["id"], ch, hex(ord(ch)), category)


@pytest.mark.parametrize("fixture_name", _REAL_HWP_FIXTURES)
def test_real_hwp_is_actually_compressed(fixture_name, request):
    """Sanity check that these fixtures exercise the zlib decompression path
    (as opposed to accidentally testing only the uncompressed branch)."""
    path = request.getfixturevalue(fixture_name)
    with olefile.OleFileIO(str(path)) as ole:
        assert _is_compressed(ole) is True


@pytest.mark.parametrize("fixture_name", _REAL_HWP_FIXTURES)
def test_parse_hwp_on_real_file_matches_stripped_shape(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    nodes = parse_hwp(path)

    assert nodes
    assert all(set(n.keys()) == {"type", "id", "text"} for n in nodes)
    assert [n["id"] for n in nodes] == [f"p{i}" for i in range(1, len(nodes) + 1)]


def test_animal_shelter_hwp_extracts_clean_paragraphs(animal_shelter_hwp):
    nodes = build_structure_tree_hwp(animal_shelter_hwp)

    assert len(nodes) == 93
    assert nodes[0]["text"] == "동물보호센터 운영현황"
    assert nodes[1]["text"] == "□ 시군별 현황"
    # table cells: county name, then the shelter name/address/phone columns
    assert "춘천시" in [n["text"] for n in nodes[:15]]
    assert any("033-" in n["text"] for n in nodes)  # phone numbers survived intact
    # a parenthetical aside that sits in its own table cell/paragraph
    assert any("농업기술센터" in n["text"] for n in nodes)


def test_construction_safety_cost_notice_hwp_extracts_clean_paragraphs(
    construction_safety_cost_notice_hwp,
):
    nodes = build_structure_tree_hwp(construction_safety_cost_notice_hwp)

    assert len(nodes) == 641
    assert nodes[0]["text"] == "건설업 산업안전보건관리비 계상 및 사용기준"
    # a chapter heading and the article text right after it, with legal
    # citations wrapped in CJK brackets (｢｣) rather than ASCII quotes
    assert nodes[74]["text"] == "제 1 장   총    칙"
    assert "｢산업안전보건법｣" in nodes[75]["text"]
    assert nodes[75]["text"].startswith("제1조(목적)")
    # revision-history entries scattered through the front matter
    assert any(n["text"] == "고시 제88 - 13호" for n in nodes)
