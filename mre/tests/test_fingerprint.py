from bs4 import BeautifulSoup

from mre import HTMLSiteAdapter, fetch_block
from mre.html_site_adapter import GeneratorFingerprintMismatch, compute_adapter_fingerprint
from mre.xml_builder import build_mre_xml
import logging
import pytest


def _extract_v1(soup):
    return [{"type": "paragraph", "id": "p1", "text": "hello world"}]


def _extract_v2(soup):
    # behaviorally different (even if trivially so) -> must fingerprint differently
    return [{"type": "paragraph", "id": "p1", "text": "hello world!!"}]


def _strip(nodes):
    return nodes


def _embed(html, xml):
    return html.replace("</head>", f"<script>{xml}</script></head>")


def _fetch(html, node_id):
    return "hello world"


def test_fingerprint_is_deterministic():
    adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed)
    first = compute_adapter_fingerprint(adapter)
    second = compute_adapter_fingerprint(adapter)
    assert first == second


def test_fingerprint_changes_when_extract_changes():
    a1 = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed)
    a2 = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v2, strip=_strip, embed=_embed)
    assert compute_adapter_fingerprint(a1) != compute_adapter_fingerprint(a2)


def test_fingerprint_stable_across_absent_optional_fields():
    a1 = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed)
    a2 = HTMLSiteAdapter(
        name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed,
        preprocess=None, assign_ids=None, fetch=None,
    )
    assert compute_adapter_fingerprint(a1) == compute_adapter_fingerprint(a2)


def _make_embedded_doc(adapter: HTMLSiteAdapter) -> str:
    html = "<html><head></head><body></body></html>"
    nodes = adapter.extract(BeautifulSoup(html, "lxml"))
    llm_data = {"summary": "", "headings": ["h"], "keywords": ["k"]}
    mre_xml = build_mre_xml(
        llm_data, nodes, title="T",
        generator=adapter.name, generator_fingerprint=compute_adapter_fingerprint(adapter),
    )
    return adapter.embed(html, mre_xml)


def test_mismatch_warns_by_default_and_still_fetches(caplog):
    old_adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed, fetch=_fetch)
    embedded = _make_embedded_doc(old_adapter)

    new_adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v2, strip=_strip, embed=_embed, fetch=_fetch)

    with caplog.at_level(logging.WARNING):
        result = fetch_block("https://x.example/page", embedded, "p1", fallback=new_adapter)
    assert result == "hello world"
    assert any("fingerprint" in rec.message.lower() for rec in caplog.records)


def test_mismatch_raises_in_strict_mode():
    old_adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed, fetch=_fetch)
    embedded = _make_embedded_doc(old_adapter)

    new_adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v2, strip=_strip, embed=_embed, fetch=_fetch)

    with pytest.raises(GeneratorFingerprintMismatch):
        fetch_block("https://x.example/page", embedded, "p1", fallback=new_adapter, strict=True)


def test_no_fingerprint_in_doc_skips_check_entirely():
    adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed, fetch=_fetch)
    html = "<html><head></head><body></body></html>"
    nodes = adapter.extract(BeautifulSoup(html, "lxml"))
    # build_mre_xml without generator/generator_fingerprint -> no fingerprint stamped
    mre_xml = build_mre_xml({"summary": "", "headings": ["h"], "keywords": ["k"]}, nodes, title="T")
    embedded = adapter.embed(html, mre_xml)

    # should not raise even in strict mode -- there's nothing to compare against
    result = fetch_block("https://x.example/page", embedded, "p1", fallback=adapter, strict=True)
    assert result == "hello world"


def test_matching_fingerprint_no_warning(caplog):
    adapter = HTMLSiteAdapter(name="x", domains=("x.example",), extract=_extract_v1, strip=_strip, embed=_embed, fetch=_fetch)
    embedded = _make_embedded_doc(adapter)

    with caplog.at_level(logging.WARNING):
        fetch_block("https://x.example/page", embedded, "p1", fallback=adapter)
    assert not any("fingerprint" in rec.message.lower() for rec in caplog.records)
