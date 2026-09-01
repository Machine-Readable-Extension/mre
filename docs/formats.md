# Supported document formats

| Format | Parsing | Embedding | Fetch |
|---|---|---|---|
| HTML (Wikipedia) | built-in site adapter | `<script type="application/mre+xml">` inside `<head>` | `fetch_block()` |
| HWPX | built-in | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| DOCX | built-in (body paragraphs only — table cells are out of scope) | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| HWP (legacy, OLE2) | built-in, paragraph text only — see below | not implemented (see below) | not yet |
| PDF | built-in, paragraph text only — see below | `mre.xml` as a PDF file attachment — see below | `fetch_pdf()` |

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
[`mre/tests/_ole2_builder.py`](https://github.com/Machine-Readable-Extension/py-mre/tree/master/mre/tests/_ole2_builder.py)
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

## PDF

**Parsing** — `mre.pdf_adapter.parse_pdf(path)` (or `build_structure_tree_pdf(path)`
for the unstripped nodes, paragraph-only — no heading concept, same as HWP/HWPX)
extracts each page's text via `pypdf`'s `extraction_mode="layout"`, which — unlike the
default plain-text mode — reconstructs vertical whitespace as blank lines
proportional to the actual gap between lines on the page. A run of non-blank lines
separated from the next by a blank line becomes one paragraph; a page with no
extra spacing anywhere falls back to one paragraph for the whole page rather than
fragmenting every wrapped line. Scanned/image-only PDFs (no text layer) yield no
paragraphs — that needs OCR, out of scope here. Some PDFs draw bullets through a
custom symbol font with no `ToUnicode` mapping, which would otherwise leak a raw
private-use-area codepoint into the text (observed on a real report) — stripped
alongside stray control characters, mirroring HWP's leftover-control-char safety net.

```python
from mre.pdf_adapter import parse_pdf

nodes = parse_pdf("report.pdf")
```

**Embed/fetch, unlike HWP, is supported** — the reason HWP is parsing-only
("no maintained pure-Python CFB writer") doesn't apply to PDF: PDF has a standard
container mechanism, file attachments (`/EmbeddedFiles`, PDF32000-1:2008 §7.11.3),
that plays the same role HWPX/DOCX's "extra zip entry" does. `pypdf.PdfWriter`
supports it natively, and embedding an attachment never touches the page content
streams — `generate_mre(fmt=DocFormat.PDF, ...)` works the same way it does for
hwpx/docx (`embedded_path` is the same file, updated in place), and `docs` entries
for `run_agent()` use the identical `{"path", "fmt"}` shape with `fmt=DocFormat.PDF`.
`fetch_pdf(path, node_id)` re-parses the file the same way `fetch_opc()` does (the
single-source-of-truth principle — see
[Detecting a stale adapter](quickstart.md#detecting-a-stale-adapter)):

```python
from mre import DocFormat, embed_mre_pdf, fetch_pdf

embed_mre_pdf("report.pdf", mre_xml)
text = fetch_pdf("report.pdf", "p2")
```

Re-embedding replaces rather than accumulates — pypdf has no attachment-removal
API, so `embed_mre_pdf()` prunes any existing `mre.xml` entry out of the writer's
`Names`/`EmbeddedFiles` tree before adding the fresh one (see
`mre.pdf_adapter._remove_existing_mre_attachment` — the one place this module
reaches into pypdf's private object graph, since no public API exists for removal;
the PDF-spec-fixed tree shape makes that a reasonably safe bet). No
generator-fingerprint check for this path either, same as hwpx/docx.

## Adding a new HTML site

Moved to its own page: **[Adding your site](adding-a-site.md)**. HTML support
is a site-adapter registry (only `wikipedia.org` ships out of the box), and
that page covers both the in-process and installable-plugin ways to add
another one — no involvement from the site owner required.
