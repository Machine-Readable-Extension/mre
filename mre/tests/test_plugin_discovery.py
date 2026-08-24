import logging
from types import SimpleNamespace

import pytest

from mre import HTMLSiteAdapter, discover_plugin_adapters, registered_sites
import mre.html_site_adapter as hsa


def _fake_entry_point(name, loader):
    return SimpleNamespace(name=name, load=loader)


@pytest.fixture(autouse=True)
def _restore_registry():
    """register_site() mutates the module-level adapter registry in place and
    there's no unregister(); snapshot/restore it so these plugin-registration
    tests don't leak fake adapters into the rest of the test session."""
    snapshot = dict(hsa._REGISTRY)
    yield
    hsa._REGISTRY.clear()
    hsa._REGISTRY.update(snapshot)


def test_discover_registers_a_valid_plugin_instance(monkeypatch):
    adapter = HTMLSiteAdapter(
        name="plugin-a",
        domains=("plugin-a.example",),
        extract=lambda soup: [],
        strip=lambda nodes: nodes,
        embed=lambda html, xml: html,
    )
    ep = _fake_entry_point("plugin-a", lambda: adapter)
    monkeypatch.setattr(hsa, "entry_points", lambda group=None: [ep])

    discover_plugin_adapters()

    assert "plugin-a" in registered_sites()
    assert registered_sites()["plugin-a"] == ("plugin-a.example",)


def test_discover_accepts_a_factory_that_returns_an_instance(monkeypatch):
    adapter = HTMLSiteAdapter(
        name="plugin-b", domains=("plugin-b.example",),
        extract=lambda soup: [], strip=lambda nodes: nodes, embed=lambda html, xml: html,
    )
    # ep.load() resolves to a zero-arg factory (not the instance itself) --
    # discover_plugin_adapters() must call it to get the actual adapter.
    ep = _fake_entry_point("plugin-b", lambda: (lambda: adapter))
    monkeypatch.setattr(hsa, "entry_points", lambda group=None: [ep])
    discover_plugin_adapters()
    assert "plugin-b" in registered_sites()


def test_one_broken_plugin_does_not_block_others(monkeypatch, caplog):
    good = HTMLSiteAdapter(
        name="plugin-good", domains=("good.example",),
        extract=lambda soup: [], strip=lambda nodes: nodes, embed=lambda html, xml: html,
    )

    def _broken_loader():
        raise RuntimeError("boom")

    eps = [
        _fake_entry_point("plugin-broken", _broken_loader),
        _fake_entry_point("plugin-good", lambda: good),
    ]
    monkeypatch.setattr(hsa, "entry_points", lambda group=None: eps)

    with caplog.at_level(logging.WARNING):
        discover_plugin_adapters()

    assert "plugin-good" in registered_sites()
    assert any("plugin-broken" in rec.message for rec in caplog.records)


def test_entry_point_returning_wrong_type_is_skipped_with_warning(monkeypatch, caplog):
    # ep.load() resolves to a zero-arg factory (per the "module:ADAPTER_OR_FACTORY"
    # convention) that itself returns something that isn't an HTMLSiteAdapter --
    # exercises the post-call type-check branch, distinct from a load()/call failure.
    bad_factory = lambda: "not an adapter"
    ep = _fake_entry_point("plugin-wrong-type", lambda: bad_factory)
    monkeypatch.setattr(hsa, "entry_points", lambda group=None: [ep])

    with caplog.at_level(logging.WARNING):
        discover_plugin_adapters()

    assert "plugin-wrong-type" not in registered_sites()
    assert any("plugin-wrong-type" in rec.message for rec in caplog.records)


def test_wikipedia_still_present_after_plugin_discovery(monkeypatch):
    monkeypatch.setattr(hsa, "entry_points", lambda group=None: [])
    discover_plugin_adapters()
    assert "wikipedia" in registered_sites()
