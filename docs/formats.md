# Supported document formats

| Format | Parsing | Embedding | Fetch |
|---|---|---|---|
| HTML (Wikipedia) | built-in site adapter | `<script type="application/mre+xml">` inside `<head>` | `fetch_block()` |
| HWPX | built-in | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| DOCX | built-in (body paragraphs only — table cells are out of scope) | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| HWP (legacy, OLE2) | built-in, paragraph text only — see below | not implemented (see below) | not yet |
| PDF | detected (`detect_format`) | not implemented — `generate_mre()` raises `NotImplementedError` | not yet |

HWPX/DOCX use a separate function, `fetch_opc(path, node_id, fmt)`, since (unlike HTML)
there's no per-site adapter to pick — just a format:

```python
from mre import DocFormat, fetch_opc

text = fetch_opc("pleasure_cove.hwpx", "p2", DocFormat.HWPX)
```

Same `"full"` sentinel and the same empty-string-on-miss / `FetchNotSupportedError`
contract as `fetch_block()`. No generator-fingerprint check yet for this path
(see [Detecting a stale adapter](quickstart.md#detecting-a-stale-adapter) — HTML-only for now).

## Legacy HWP (parsing-only)

**Legacy HWP is parsing-only, and lives outside the `generate_mre()`/`run_agent()`
pipeline** — `mre.hwp_adapter.parse_hwp(path)` (or `build_structure_tree_hwp(path)` for
the unstripped heading/paragraph nodes) reads a `.hwp` (OLE2/Compound File Binary, not
a zip) file's `BodyText/SectionN` streams and returns the same
`[{"type": "paragraph", "id": "pN", "text": ...}, ...]` shape as `parse_opc()` — no
headings, same as HWPX. `generate_mre(fmt=DocFormat.HWP, ...)` still raises
`NotImplementedError`: embedding requires writing a new stream into the OLE2
container in-place, and there's no maintained pure-Python library that can write CFB
files (`olefile`, used for reading, is read-only). Use `parse_hwp()` directly if you
just need the text:

```python
from mre.hwp_adapter import parse_hwp

nodes = parse_hwp("report.hwp")
```

Internally this decompresses `BodyText/SectionN` (raw deflate, when `FileHeader`'s
flag bit says the document is compressed) and walks the `HWPTAG_PARA_TEXT` records in
each one, skipping the inline-control-character parameter bytes that are mixed into
the UTF-16 paragraph text (naively decoding the whole stream without skipping those
survives on simple documents but leaks garbage characters on ones with embedded
objects, hyperlinks, or tables). Tested against synthetic fixtures built by
[`mre/tests/_ole2_builder.py`](https://github.com/Machine-Readable-Extension/mre/tree/master/mre/tests/_ole2_builder.py)
(control-char/record-parsing edge cases) and two real compressed government documents
— a table layout (`animal_shelter_status.hwp`) and prose/legal-citation text
(`construction_safety_cost_notice.hwp`) — solid on tables, tabs, and parenthetical
asides, but still not exhaustively battle-tested against every real-world `.hwp`
quirk (heavy embedded objects, equations, revision marks).

!!! warning "HWP has no embed path, and the one workaround has known content loss"
    `mre.convert_hwp(path, target=DocFormat.DOCX)` shells out to an **externally
    installed** LibreOffice + [H2Orestart](https://github.com/ebandal/H2Orestart) (a
    community reverse-engineered HWP import filter, not Hancom's own converter) to
    produce a `.docx`/`.pdf` you can then run through the normal `generate_mre()`/embed
    pipeline. Measured against a real 27-table government document: opening prose
    matched `parse_hwp()` character-for-character and ~86% of total text survived, but
    a revision-history table entry was **not found anywhere** in the converted output —
    genuine loss from the filter, not just this library's own table-cell scope limit.
    Every call logs a `WARNING`. **Verify the converted output before trusting it for
    anything table-heavy or otherwise structurally complex.**

    Deliberately kept separate from `generate_mre()` — a hard external system
    dependency this library can't pip-install, plus the measured content loss above,
    mean it should never run implicitly. Call it explicitly, then feed its output into
    the existing docx/pdf pipeline yourself:

    ```python
    from mre import DocFormat, convert_hwp, generate_mre

    docx_path = convert_hwp("report.hwp", target=DocFormat.DOCX)  # needs soffice on PATH
    result = await generate_mre(docx_path, client=client, model=model,
                                 title="...", fmt=DocFormat.DOCX)
    ```

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
