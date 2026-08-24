from bs4 import BeautifulSoup

from mre import fetch_block, get_site_adapter
from mre.html_site_adapter import FetchNotSupportedError
from mre.xml_builder import build_mre_xml
import pytest

_URL = "https://en.wikipedia.org/wiki/David_Lightfoot"
_TITLE = "David Lightfoot"


def _build_embedded_doc(html: str):
    adapter = get_site_adapter(_URL)
    soup = BeautifulSoup(html, "lxml")
    adapter.preprocess(soup)
    raw_nodes = adapter.extract(soup)
    adapter.assign_ids(raw_nodes, _TITLE)
    stripped = adapter.strip(raw_nodes)

    para_nodes = [n for n in stripped if n["type"] == "paragraph"]
    llm_data = {
        "summary": "stub summary",
        "headings": [f"heading for {n['id']}" for n in para_nodes],
        "keywords": [f"kw-{n['id']}" for n in para_nodes],
    }
    mre_xml = build_mre_xml(llm_data, stripped, title=_TITLE)
    embedded_html = adapter.embed(html, mre_xml)
    return embedded_html, para_nodes


def test_fetch_block_returns_original_paragraph_text(david_lightfoot_html):
    embedded_html, para_nodes = _build_embedded_doc(david_lightfoot_html)
    assert para_nodes, "fixture must have paragraphs"
    for node in para_nodes:
        fetched = fetch_block(_URL, embedded_html, node["id"])
        assert fetched.strip() == node["text"].strip()


def test_fetch_block_full_returns_whole_document(david_lightfoot_html):
    embedded_html, para_nodes = _build_embedded_doc(david_lightfoot_html)
    full_text = fetch_block(_URL, embedded_html, "full")
    for node in para_nodes:
        assert node["text"].strip() in full_text


def test_fetch_block_unknown_id_does_not_crash(david_lightfoot_html):
    embedded_html, _ = _build_embedded_doc(david_lightfoot_html)
    # no assertion on content — just documents current behavior (empty-ish
    # result) so a future change to this contract is visible in a diff.
    fetch_block(_URL, embedded_html, "Z999")


def test_fetch_raises_when_adapter_has_no_fetch():
    from mre import HTMLSiteAdapter

    adapter = HTMLSiteAdapter(
        name="no-fetch-site",
        domains=("no-fetch.example",),
        extract=lambda soup: [],
        strip=lambda nodes: nodes,
        embed=lambda html, xml: html,
    )
    with pytest.raises(FetchNotSupportedError):
        fetch_block("https://no-fetch.example/page", "<html></html>", "p1", fallback=adapter)
