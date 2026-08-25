from bs4 import BeautifulSoup

from mre import extract_mre_xml, get_site_adapter, register_site, registered_sites
from mre.html_site_adapter import UnknownSiteError
import pytest


def _extract_ids(html: str, title: str) -> list[str]:
    adapter = get_site_adapter("https://en.wikipedia.org/wiki/David_Lightfoot")
    soup = BeautifulSoup(html, "lxml")
    if adapter.preprocess is not None:
        adapter.preprocess(soup)
    raw_nodes = adapter.extract(soup)
    if adapter.assign_ids is not None:
        adapter.assign_ids(raw_nodes, title)
    return raw_nodes


def test_wikipedia_is_registered_by_default():
    assert "wikipedia" in registered_sites()
    assert registered_sites()["wikipedia"] == ("wikipedia.org",)


def test_unknown_site_raises():
    with pytest.raises(UnknownSiteError):
        get_site_adapter("https://not-a-registered-site.example")


def test_extract_produces_headings_and_paragraphs(david_lightfoot_html):
    nodes = _extract_ids(david_lightfoot_html, "David Lightfoot")
    types = {n["type"] for n in nodes}
    assert types == {"heading", "paragraph"}
    # fixture has 6 raw <p> tags, but adapter.preprocess() (_strip_appendix_sections)
    # drops the "External links"/"See also" appendix sections' paragraphs -- 4 is
    # the real post-preprocessing count.
    assert sum(1 for n in nodes if n["type"] == "paragraph") == 4


def test_paragraph_ids_get_title_letter_prefix(david_lightfoot_html):
    # "David Lightfoot" -> first alpha char is 'D'
    nodes = _extract_ids(david_lightfoot_html, "David Lightfoot")
    para_ids = [n["id"] for n in nodes if n["type"] == "paragraph"]
    assert para_ids, "fixture should contain at least one paragraph"
    for pid in para_ids:
        assert pid[0] == "D", f"expected letter-prefixed id, got {pid!r}"
        assert pid[1:].isdigit()
    # original ordering (numeric suffix) preserved, not reshuffled
    suffixes = [int(pid[1:]) for pid in para_ids]
    assert suffixes == sorted(suffixes)


def test_title_with_no_alpha_falls_back_to_X(david_lightfoot_html):
    nodes = _extract_ids(david_lightfoot_html, "1979")
    para_ids = [n["id"] for n in nodes if n["type"] == "paragraph"]
    assert all(pid.startswith("X") for pid in para_ids)


def test_strip_drops_non_llm_fields(david_lightfoot_html):
    adapter = get_site_adapter("https://en.wikipedia.org/wiki/David_Lightfoot")
    soup = BeautifulSoup(david_lightfoot_html, "lxml")
    adapter.preprocess(soup)
    raw_nodes = adapter.extract(soup)
    adapter.assign_ids(raw_nodes, "David Lightfoot")
    stripped = adapter.strip(raw_nodes)
    assert len(stripped) == len(raw_nodes)
    for node in stripped:
        assert "type" in node and "text" in node
        if node["type"] == "paragraph":
            assert "id" in node
        else:
            assert "id" not in node and "level" in node


def test_register_site_requires_domains():
    from mre import HTMLSiteAdapter

    with pytest.raises(ValueError):
        register_site(
            HTMLSiteAdapter(
                name="no-domains",
                domains=(),
                extract=lambda soup: [],
                strip=lambda nodes: nodes,
                embed=lambda html, xml: html,
            )
        )


# ─────────────────────────────────────────────
# embed (_wiki_inject_mre_into_html) -- re-embed must replace, not accumulate
# ─────────────────────────────────────────────


def test_embed_inserts_mre_script_tag_before_head_close():
    html = "<html><head><title>T</title></head><body>x</body></html>"
    embedded = get_site_adapter("https://en.wikipedia.org/wiki/X").embed(html, "<mre>v1</mre>")

    assert embedded.count('<script type="application/mre+xml">') == 1
    assert extract_mre_xml(embedded) == "<mre>v1</mre>"


def test_reembed_replaces_rather_than_accumulates():
    html = "<html><head><title>T</title></head><body>x</body></html>"
    adapter = get_site_adapter("https://en.wikipedia.org/wiki/X")

    once = adapter.embed(html, "<mre>v1</mre>")
    twice = adapter.embed(once, "<mre>v2</mre>")

    assert twice.count('<script type="application/mre+xml">') == 1
    assert extract_mre_xml(twice) == "<mre>v2</mre>"


def test_reembed_leaves_rest_of_document_untouched():
    html = "<html><head><title>T</title></head><body><p>keep me</p></body></html>"
    adapter = get_site_adapter("https://en.wikipedia.org/wiki/X")

    once = adapter.embed(html, "<mre>v1</mre>")
    twice = adapter.embed(once, "<mre>v2</mre>")

    assert "<p>keep me</p>" in twice
    assert "<title>T</title>" in twice


def test_embed_falls_back_to_prepending_when_no_head_tag():
    html = "no html tags at all"
    embedded = get_site_adapter("https://en.wikipedia.org/wiki/X").embed(html, "<mre>v1</mre>")

    assert extract_mre_xml(embedded) == "<mre>v1</mre>"
    assert "no html tags at all" in embedded
