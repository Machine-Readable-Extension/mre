import re

from mre.xml_builder import build_mre_xml

_NODES = [
    {"type": "heading", "level": 2, "text": "Biography"},
    {"type": "paragraph", "id": "p1", "text": "para one"},
    {"type": "paragraph", "id": "p2", "text": "para two"},
    {"type": "heading", "level": 2, "text": "See also"},
    {"type": "paragraph", "id": "p3", "text": "para three"},
]

_LLM_DATA = {
    "summary": "A short summary.",
    "headings": ["Heading 1", "Heading 2", "Heading 3"],
    "keywords": ["kw1, kw2", "kw3", "kw4, kw5"],
}


def test_no_section_tags_ever_emitted():
    xml = build_mre_xml(_LLM_DATA, _NODES, title="T")
    assert "<section" not in xml
    assert "</section>" not in xml


def test_heading_nodes_are_skipped_in_tree():
    xml = build_mre_xml(_LLM_DATA, _NODES, title="T")
    node_ids = re.findall(r'<node id="([^"]+)">', xml)
    assert node_ids == ["p1", "p2", "p3"]


def test_desc_and_keys_mapped_positionally():
    xml = build_mre_xml(_LLM_DATA, _NODES, title="T")
    assert "<desc>Heading 1</desc>" in xml
    assert "<keys>kw1, kw2</keys>" in xml
    assert "<desc>Heading 3</desc>" in xml


def test_version_is_1_0_by_default():
    xml = build_mre_xml(_LLM_DATA, _NODES, title="T")
    assert xml.startswith('<mre version="1.0">')


def test_generator_attrs_included_only_when_both_given():
    xml = build_mre_xml(_LLM_DATA, _NODES, title="T")
    assert "generator=" not in xml

    xml_with_gen = build_mre_xml(
        _LLM_DATA, _NODES, title="T",
        generator="wikipedia", generator_fingerprint="abc123",
    )
    assert 'generator="wikipedia"' in xml_with_gen
    assert 'generator-fingerprint="abc123"' in xml_with_gen


def test_title_and_special_chars_are_escaped():
    xml = build_mre_xml(_LLM_DATA, _NODES, title="A & B <weird>")
    assert "A &amp; B &lt;weird&gt;" in xml
    assert "<weird>" not in xml.split("<title>")[1]


def test_missing_llm_entries_fall_back_to_empty():
    short_llm_data = {"summary": "", "headings": ["only one"], "keywords": []}
    xml = build_mre_xml(short_llm_data, _NODES, title="T")
    # first paragraph gets the one heading, rest have neither <desc> nor <keys>
    assert xml.count("<desc>") == 1
    assert xml.count("<keys>") == 0
    assert "<summary>" not in xml
