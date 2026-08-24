# Supported document formats

| Format | Parsing | Embedding | Fetch |
|---|---|---|---|
| HTML (Wikipedia) | built-in site adapter | `<script type="application/mre+xml">` inside `<head>` | `fetch_block()` |
| HWPX | built-in | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| DOCX | built-in (body paragraphs only — table cells are out of scope) | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| PDF, HWP | detected (`detect_format`) | not implemented — `generate_mre()` raises `NotImplementedError` | not yet |

HWPX/DOCX use a separate function, `fetch_opc(path, node_id, fmt)`, since (unlike HTML)
there's no per-site adapter to pick — just a format:

```python
from mre import DocFormat, fetch_opc

text = fetch_opc("pleasure_cove.hwpx", "p2", DocFormat.HWPX)
```

Same `"full"` sentinel and the same empty-string-on-miss / `FetchNotSupportedError`
contract as `fetch_block()`. No generator-fingerprint check yet for this path
(see [Detecting a stale adapter](quickstart.md#detecting-a-stale-adapter) — HTML-only for now).

## Adding a new HTML site

HTML support is a **site-adapter registry**, not a generic scraper — a page's
usable structure differs too much site to site to parse generically. Only
`wikipedia.org` ships out of the box. Two ways to add another site:

**In-process, for a one-off script:**

```python
from mre import HTMLSiteAdapter, register_site

register_site(
    HTMLSiteAdapter(
        name="my-site",
        domains=("example.com",),
        extract=my_extract_fn,   # soup -> [{"type": "heading"|"paragraph", ...}, ...]
        strip=my_strip_fn,       # -> LLM-ready node list
        embed=my_embed_fn,       # (html, mre_xml) -> html with MRE injected
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
    embed=my_embed_fn,
)
```

`mre` scans installed packages for this entry-point group every time it's
imported (`mre.registered_sites()` shows what was found — built-ins plus
every discovered plugin) and registers each one automatically. A plugin
that fails to load only logs a warning; it never breaks discovery of the
others. See [`examples/mre-example-adapter/`](https://github.com/Machine-Readable-Extension/mre/tree/master/examples/mre-example-adapter)
for a complete, working reference package built exactly this way — install
it (`pip install -e examples/mre-example-adapter`) and `example.com` support
appears with no other code changes.

Or pass `html_fallback_adapter=` to `generate_mre()` for a one-off document
without registering it globally.
