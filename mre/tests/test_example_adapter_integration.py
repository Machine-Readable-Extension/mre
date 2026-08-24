"""End-to-end check that examples/mre-example-adapter works the way the
docs claim: install it, and `mre` picks it up purely through the
`mre.site_adapters` entry point -- no code in `mre` core names this
package or vice versa.

Every other plugin-discovery test (test_plugin_discovery.py) exercises
discover_plugin_adapters() against a *fake*, monkeypatched entry point --
useful for the discovery logic itself, but it never proves a real,
separately-packaged plugin actually gets found by importlib.metadata and
actually works end to end. This module does, which is why it needs the
example package genuinely installed rather than mocked.

Skipped entirely (not failed) if mre_example_adapter isn't installed --
`pip install -e examples/mre-example-adapter` first (the "Install example
adapter plugin" step in .github/workflows/tests.yml does this in CI).
"""

import pytest

pytest.importorskip("mre_example_adapter")

from bs4 import BeautifulSoup

from mre import fetch_block, get_site_adapter, registered_sites

_EXAMPLE_URL = "https://example.com/some-article"
_EXAMPLE_HTML = """
<html><head></head><body>
<article>
<p>First paragraph about widgets.</p>
<p>Second paragraph about gadgets.</p>
</article>
</body></html>
"""


def test_example_adapter_discovered_via_entry_point():
    sites = registered_sites()
    assert "example" in sites
    assert "example.com" in sites["example"]


def test_example_adapter_extract_embed_fetch_roundtrip():
    """Runs the installed plugin's real extract/embed/fetch functions
    through mre's own dispatch (get_site_adapter/fetch_block) -- the same
    path generate_mre()/fetch_block() use in production.
    """
    adapter = get_site_adapter(_EXAMPLE_URL)
    assert adapter.name == "example"

    soup = BeautifulSoup(_EXAMPLE_HTML, "lxml")
    nodes = adapter.strip(adapter.extract(soup))
    assert [n["text"] for n in nodes] == [
        "First paragraph about widgets.",
        "Second paragraph about gadgets.",
    ]

    # A minimal <mre> document stands in for generate_mre()'s real output --
    # this test isn't exercising generation, just that the plugin's own
    # embed/fetch round-trip through mre's dispatch.
    fake_mre_xml = '<mre version="1.0"><metadata><title>t</title></metadata><tree></tree></mre>'
    embedded_html = adapter.embed(_EXAMPLE_HTML, fake_mre_xml)
    assert "application/mre+xml" in embedded_html

    assert fetch_block(_EXAMPLE_URL, embedded_html, "p1") == "First paragraph about widgets."
    assert fetch_block(_EXAMPLE_URL, embedded_html, "p2") == "Second paragraph about gadgets."
    assert fetch_block(_EXAMPLE_URL, embedded_html, "full") == (
        "First paragraph about widgets.\n\nSecond paragraph about gadgets."
    )
    assert fetch_block(_EXAMPLE_URL, embedded_html, "p3") == ""
