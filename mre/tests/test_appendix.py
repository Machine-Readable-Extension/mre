from bs4 import BeautifulSoup

from mre.appendix import _strip_appendix_sections


def test_strip_appendix_sections_removes_flat_appendix():
    soup = BeautifulSoup(
        '<body>'
        '<section aria-labelledby="Intro"><h2 id="Intro">Intro</h2><p>body text</p></section>'
        '<section aria-labelledby="References"><h2 id="References">References</h2><p>cite 1</p></section>'
        '</body>',
        "lxml",
    )
    appendix = _strip_appendix_sections(soup)

    assert appendix == [(2, "References")]
    assert soup.find(id="References") is None
    assert "body text" in soup.get_text()
    assert "cite 1" not in soup.get_text()


def test_strip_appendix_sections_survives_nested_sections_inside_appendix():
    """Regression test: real (Parsoid-style) Wikipedia HTML nests subsections
    inside their parent <section> (e.g. "Works cited" nested inside a
    "Bibliography" appendix section). find_all("section") returns every
    <section> in one flat list regardless of nesting, so decomposing an outer
    appendix section used to leave stale references to its now-decomposed
    descendants later in that same list -- bs4's Tag.decompose() recursively
    clears attrs on the whole subtree, so calling .get() on one of those
    stale descendants crashed with AttributeError: 'NoneType' object has no
    attribute 'get'. Caught against a real live Wikipedia article
    (en.wikipedia.org/wiki/Alan_Turing) whose Bibliography section nests an
    "Articles"/"Books"/"Works cited" sub-section the same way.
    """
    soup = BeautifulSoup(
        '<body>'
        '<section aria-labelledby="Intro"><h2 id="Intro">Intro</h2><p>body text</p></section>'
        '<section aria-labelledby="Bibliography">'
        '  <h2 id="Bibliography">Bibliography</h2>'
        '  <section aria-labelledby="Works_cited"><h3 id="Works_cited">Works cited</h3><p>cite 1</p></section>'
        '</section>'
        '</body>',
        "lxml",
    )

    appendix = _strip_appendix_sections(soup)  # must not raise

    assert appendix == [(2, "Bibliography")]
    assert soup.find(id="Bibliography") is None
    assert soup.find(id="Works_cited") is None
    assert "body text" in soup.get_text()
    assert "cite 1" not in soup.get_text()


def test_strip_appendix_sections_removes_sources():
    """Wikipedia biography-style articles commonly use a "Sources" heading
    for their bibliography/citation list (distinct from "References" and
    "Bibliography", which were already stripped). Real examples: "Gerard,
    Count of Rieneck", "Moritz of Limburg".
    """
    soup = BeautifulSoup(
        '<body>'
        '<section aria-labelledby="Intro"><h2 id="Intro">Intro</h2><p>body text</p></section>'
        '<section aria-labelledby="Sources"><h2 id="Sources">Sources</h2><p>cite 1</p></section>'
        '</body>',
        "lxml",
    )
    appendix = _strip_appendix_sections(soup)

    assert appendix == [(2, "Sources")]
    assert soup.find(id="Sources") is None
    assert "body text" in soup.get_text()
    assert "cite 1" not in soup.get_text()


def test_strip_appendix_sections_non_parsoid_mw_heading_divs():
    soup = BeautifulSoup(
        '<div class="mw-parser-output">'
        '<div class="mw-heading mw-heading2"><h2 id="Intro">Intro</h2></div>'
        '<p>body text</p>'
        '<div class="mw-heading mw-heading2"><h2 id="See_also">See also</h2></div>'
        '<ul><li>related link</li></ul>'
        '</div>',
        "lxml",
    )
    appendix = _strip_appendix_sections(soup)

    assert appendix == [(2, "See also")]
    assert soup.find(id="See_also") is None
    assert "body text" in soup.get_text()
    assert "related link" not in soup.get_text()
