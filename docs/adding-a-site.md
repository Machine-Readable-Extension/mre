# Adding your site

HTML support is a **site-adapter registry**, not a generic scraper — a page's
usable structure differs too much site to site to parse generically. Only
`wikipedia.org` ships out of the box. Two ways to add another site:

**You don't need the site owner's involvement, and you don't need to adopt MRE
headers at all.** An adapter's `extract`/`strip` are what actually matter for
turning a page into clean, LLM-ready text; `embed` is a required field on the
`HTMLSiteAdapter` dataclass, but it can be a harmless no-op
(`embed=lambda html, xml: html`) if the site never publishes MRE headers and you
just want parsing. Anyone -- not just a site's own maintainers -- can write and
publish an adapter this way.

**In-process, for a one-off script:**

```python
from mre import HTMLSiteAdapter, register_site

register_site(
    HTMLSiteAdapter(
        name="my-site",
        domains=("example.com",),
        extract=my_extract_fn,         # soup -> [{"type": "heading"|"paragraph", ...}, ...]
        strip=my_strip_fn,             # -> LLM-ready node list
        embed=lambda html, xml: html,  # no-op if this site won't publish MRE headers;
                                        # implement for real to also support generate_mre()
    ),
)
```

**As an installable plugin package**, so anyone who `pip install`s it gets
your site supported automatically — no changes to `mre` itself, and no
`register_site()` call needed at all. This is how a site owner (or a company
managing a domain) ships their own adapter: publish a package that declares
an `mre.site_adapters` entry point pointing at an `HTMLSiteAdapter` instance.

```toml
# your_package/pyproject.toml
[project.entry-points."mre.site_adapters"]
my-site = "your_package:ADAPTER"
```

```python
# your_package/__init__.py
from mre import HTMLSiteAdapter

ADAPTER = HTMLSiteAdapter(
    name="my-site",
    domains=("example.com",),
    extract=my_extract_fn,
    strip=my_strip_fn,
    embed=lambda html, xml: html,  # or a real embed function, see the note above
)
```

`mre` scans installed packages for this entry-point group every time it's
imported (`mre.registered_sites()` shows what was found — built-ins plus
every discovered plugin) and registers each one automatically. A plugin
that fails to load only logs a warning; it never breaks discovery of the
others. See [`examples/mre-example-adapter/`](https://github.com/Machine-Readable-Extension/py-mre/tree/master/examples/mre-example-adapter)
for a complete, working reference package built exactly this way — install
it (`pip install -e examples/mre-example-adapter`) and `example.com` support
appears with no other code changes.

Or pass `html_fallback_adapter=` to `generate_mre()` for a one-off document
without registering it globally.
