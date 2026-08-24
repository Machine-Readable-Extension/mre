# Machine-Readable-Extension(MRE)

[![Tests](https://github.com/Machine-Readable-Extension/mre/actions/workflows/tests.yml/badge.svg)](https://github.com/Machine-Readable-Extension/mre/actions/workflows/tests.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Machine-Readable-Extension_mre&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Machine-Readable-Extension_mre)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Machine-Readable-Extension_mre&metric=bugs)](https://sonarcloud.io/summary/new_code?id=Machine-Readable-Extension_mre)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Machine-Readable-Extension_mre&metric=coverage)](https://sonarcloud.io/summary/new_code?id=Machine-Readable-Extension_mre)

**Machine-Readable Extension (MRE)** — a producer-side document standard and
navigation structure that lets LLM agents read a document precisely, instead
of consuming its raw, markup-heavy source.

## Why

The web is built for human eyes: HTML pages are dominated by markup tags,
scripts, styling, and boilerplate that a browser renders away but an LLM
agent has to read in full. The same overhead shows up in XML-based formats
like DOCX, HWPX, and EPUB. That noise raises inference cost and increases the
risk of an agent missing the passage it actually needed, buried somewhere in
a long context.

MRE takes a different approach from *cleaning documents up after the fact*
(heuristic scrapers, LLM-written parsers per site). It puts a small,
standardized XML header into the document **once, at publication time** —
much like `robots.txt` or `sitemap.xml` let producers tell crawlers where to
go. The header doesn't just say what a document is about; it tells an agent
exactly which paragraph to look at, by a stable ID. An agent reads the
header, requests only the paragraphs it needs, and never has to touch the
raw markup.

This package covers both ends of that story for a registered site: it
generates and embeds MRE headers into a document (`generate_mre()`), and it
retrieves a specific paragraph's full text back out of an MRE-embedded
document by ID (`fetch_block()`) — the same per-site adapter backs both, so
a site's owner writes their parsing logic once and it stays correct on both
sides. Everything *above* that — the agent's turn-by-turn reasoning loop
deciding which IDs to request — is a separate concern, kept out of this
core package. An opt-in `mre.agent` subpackage implements one such loop for
anyone who wants it too, not just the primitives — see
[Agentic RAG](#agentic-rag), below.

## What it looks like

```xml
<html>
<head>
  ...
  <script type="application/mre+xml">
  <mre version="1.0">
    <metadata>
      <title>Pleasure Cove</title>
      <summary>Pleasure Cove is ...</summary>
    </metadata>
    <tree>
      <node id="p1">
        <desc>Released in 1979 on the ABC network.</desc>
        <keys>Pleasure Cove, 1979, ABC</keys>
      </node>
      <node id="p2">
        <desc>Protagonist is a conman at the resort.</desc>
        <keys>Raymond Gordon, conman, resort</keys>
      </node>
      <node id="p3">
        <desc>Tom Jones stars as Raymond Gordon.</desc>
        <keys>Tom Jones, Raymond Gordon</keys>
      </node>
    </tree>
  </mre>
  </script>
</head>
<body>
  ...
</body>
</html>
```

`<metadata>` carries the document's title and a short summary — it lets an
agent judge whether the *whole document* is worth exploring before reading
anything else. `<tree>` maps the document's paragraphs: each `<node>` has a
stable ID, a `<desc>` naming what the paragraph asserts, and `<keys>` —
distinctive entity names that let an agent bridge across documents when
chasing a multi-hop query. Paragraph text itself is *not* in the header —
it's fetched on demand by ID from the source document, by a separate
retrieval-side parser.

Early experiments on MRE-based agentic RAG show up to a 48.4% relative F1
improvement over baselines on multi-hop QA benchmarks, by letting the agent
target the right paragraph instead of reading (or missing) it inside a
long, noisy context. Write-up is in progress (not yet on arXiv); benchmark
and evaluation code lives at
[Lactobacillus/machine-readable-extension](https://github.com/Lactobacillus/machine-readable-extension).

## Install

Not yet published to PyPI — install from this repository:

```bash
pip install -e .
```

The keyword-grounding repair pass (see [Repair](#repair), below) has an
optional fuzzy-matching fallback:

```bash
pip install -e ".[fuzzy]"
```

## Quick start

```python
import asyncio
import openai
from mre import generate_mre

async def main():
    # Any OpenAI-compatible async client works — the official SDK, or a
    # vLLM / other OpenAI-compatible server (just point base_url at it).
    client = openai.AsyncOpenAI()

    with open("pleasure_cove.html", encoding="utf-8") as f:
        html = f.read()

    result = await generate_mre(
        html,
        client=client,
        model="gpt-4o-mini",
        title="Pleasure Cove",
        url="https://en.wikipedia.org/wiki/Pleasure_Cove",  # picks the site adapter
    )

    print(result.mre_xml)          # the generated <mre> block
    print(result.embedded_html)    # original HTML with MRE injected into <head>

asyncio.run(main())
```

`generate_mre()` auto-detects the document format from `fmt`/`url`/magic
bytes, dispatches to the right adapter, calls the LLM to fill in per-paragraph
`<desc>`/`<keys>` and a document `<summary>`, repairs any paragraphs the LLM
skipped, assembles the `<mre>` XML, and (by default) embeds it back into the
document. Model choice isn't hardcoded — you always pass your own
`(client, model)`.

### Fetching a paragraph back out

An agent reading the header decides which `id`s it needs, then calls
`fetch_block()` (HTML only, for now) to get that paragraph's full text —
untruncated, unlike the short preview `<desc>` the LLM saw while generating:

```python
from mre import fetch_block

text = fetch_block(
    "https://en.wikipedia.org/wiki/Pleasure_Cove",  # same url, to pick the adapter
    result.embedded_html,
    "p2",
)
```

Pass `"full"` as the id to get the whole document's text at once, for a
workflow where the agent decides a single paragraph isn't enough context.
This goes through the *same* site adapter as generation — for a document
whose adapter doesn't implement `fetch` (a generation-only adapter),
`fetch_block()` raises `FetchNotSupportedError` rather than guessing.

### Detecting a stale adapter

A site's adapter package can get upgraded after documents were already
generated with an older version — if the paragraph-walking logic changed,
an `id` embedded in an old document could now point at the wrong paragraph.
`generate_mre()` guards against this automatically: it hashes the adapter's
`extract`/`preprocess`/`assign_ids`/`fetch` functions and stamps the result
into the header as `generator-fingerprint`. `fetch_block()` recomputes that
hash from whatever adapter is *currently* installed and compares — a
mismatch means the parsing logic changed since this document was generated.
By default it just logs a warning and fetches anyway (best-effort); pass
`strict=True` to raise `GeneratorFingerprintMismatch` instead. Documents
generated before this feature existed (no `generator-fingerprint` in the
header) skip the check entirely rather than false-alarming.

This is a content hash, not a version number an adapter author has to
remember to bump — it changes automatically whenever the relevant code
does, so it can't go stale from a forgotten version bump. The tradeoff is
that it's *oversensitive*: renaming a variable or rewording a comment in
one of those functions also changes the fingerprint, even though nothing
about paragraph order actually changed. That's considered the safer failure
direction — an occasional unnecessary warning costs little, while a silent
wrong-paragraph fetch is a real correctness bug.

## Agentic RAG

`mre` core stops at generating and fetching — deciding which `id`s to
request, turn by turn, is a separate concern kept out of the base package.
`mre.agent` is an **opt-in subpackage** for anyone who wants that loop too:
`import mre` never pulls it in, `import mre.agent` does (no extra install —
it only needs `openai`, already a core dependency).

It implements one retrieval strategy, **progressive disclosure**: every
candidate document starts out showing only its `<metadata>` (title +
summary — cheap to show many documents at once). The agent picks specific
documents to `expand_document` (reveals the full `<tree>`) or `fetch_doc`
(pulls the whole document's text directly, when metadata alone makes clear
the whole document is needed) before drilling into individual paragraphs
with `fetch_blocks`.

```python
import asyncio
import openai
from mre.agent import run_agent

async def main():
    client = openai.AsyncOpenAI()

    docs = {
        "Pleasure Cove": {
            "html": embedded_html,  # from generate_mre()'s embedded_html
            "url": "https://en.wikipedia.org/wiki/Pleasure_Cove",  # picks the site adapter
        },
        # ... more candidate documents
    }

    result = await run_agent(
        "Who starred in Pleasure Cove?",
        docs,
        client=client,
        model="gpt-4o-mini",
    )

    print(result.answer)
    print(result.success, result.num_turns, result.stats)

asyncio.run(main())
```

Each `docs` entry needs both `"html"` (the MRE-embedded document) and
`"url"` (forwarded to `fetch_block()` to pick the right site adapter).
`run_agent()` returns an `AgentResult` — `answer`, `success`, `num_turns`,
the full `messages`/`action_log` for inspection, and `stats` (prompt/
completion tokens, call count — same shape as `generate_mre()`'s `stats`).
It raises `MRENotFoundError` if a candidate document has no embedded header,
and `BlockFetchError` if a requested paragraph can't be resolved.

Like the core package, this isn't only usable as one function — every piece
`run_agent()` is built from is independently importable, for wiring MRE
into a different agent loop (LangChain, a hand-rolled one, ...) instead:

| Piece | What it does |
|---|---|
| `mre.agent.SYSTEM_PROMPT` | static system prompt describing the four tools |
| `mre.agent.ANSWER_FORMAT` | small `.format(query=...)` template for the answer instruction |
| `mre.agent.build_progressive_action_schema(titles, has_expanded=, has_retrieved=)` | guided-decoding JSON schema for the current turn's state |
| `mre.agent.CHECK_SUFFICIENCY_SCHEMA` | schema for the one-turn "is this answer actually supported by what was retrieved?" check |
| `mre.agent.metadata_view(mre_xml)` | strips `<tree>` down to metadata-only, for the first-stage view |
| `mre.reader.extract_mre_xml(html)` | reads the raw `<mre>` block back out of embedded HTML |

`run_agent()` answers in whatever length and form the question calls for —
there's no short-answer/long-answer mode switch, since that distinction
only matters for exact-match benchmark grading, not for a real consumer of
the answer.

Only progressive disclosure is implemented so far. `mre.agent` uses an
OpenAI-compatible async client the same way `generate_mre()` does (so a
local vLLM server works too, via `base_url`), and only works with formats
whose site adapter implements `fetch` (currently HTML/Wikipedia — see
[Supported document formats](#supported-document-formats), below).

## Supported document formats

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
(see [Detecting a stale adapter](#detecting-a-stale-adapter) — HTML-only for now).

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
others. See [`examples/mre-example-adapter/`](examples/mre-example-adapter)
for a complete, working reference package built exactly this way — install
it (`pip install -e examples/mre-example-adapter`) and `example.com` support
appears with no other code changes.

Or pass `html_fallback_adapter=` to `generate_mre()` for a one-off document
without registering it globally.

## Specification

`<mre>` is a single element containing `<metadata>` and `<tree>`.

- **`<metadata>`**: `<title>` (display title) and `<summary>` (2–4 sentence
  overview of the document's core message — used for document-level
  relevance judgment before any paragraph is fetched).
- **`<tree>`**: a flat, in-document-order map of the body's paragraphs —
  `<node id="pN">` marks one paragraph (no `<section>` nesting; heading
  structure is used as context during generation but isn't reflected in the
  tree). Each node carries:
  - `<desc>` — a short heading naming the paragraph's main entity and what
    it asserts about it (e.g. "directed by", "birthplace of").
  - `<keys>` — comma-separated distinctive proper nouns appearing in that
    paragraph, used as entity anchors for multi-hop bridging.

The header travels embedded in its source document, outside the rendering
path — invisible to human readers and to any software that doesn't look for
it.

## Repair

LLM generation for long documents is chunked, and a chunk boundary or a
truncated response can leave some paragraphs without a `<desc>`/`<keys>`.
`generate_mre(..., repair=True)` (default) detects those paragraphs and
regenerates just that subset, anchored by paragraph ID so replacements land
back in the right place. A stricter check — rejecting `<keys>` phrases that
don't literally appear in their paragraph's text — is available via
`repair_misaligned=True`, but ships **off** by default: forcing literal
grounding was found to strip cross-paragraph entity signal that retrieval
actually benefits from.

## Package layout

| Module | Responsibility |
|---|---|
| `generate.py` | `generate_mre()` — the top-level entry point |
| `format_detect.py` | magic-byte format detection (html/pdf/hwp/hwpx/docx) |
| `html_site_adapter.py` | per-domain HTML adapter registry + `mre.site_adapters` entry-point plugin discovery (Wikipedia built in) |
| `opc_adapter.py` | HWPX/DOCX parsing and zip-based embedding |
| `nodes.py` | format-agnostic node normalization shared by all adapters |
| `appendix.py` | Wikipedia appendix-section stripping / short-section merging |
| `generation.py` | LLM prompt, schema, and chunked calling |
| `repair.py` | missing/misaligned paragraph detection and targeted regeneration |
| `xml_builder.py` | pure string assembly of the final `<mre>` XML |
| `reader.py` | reads a `<mre>` header back out of embedded HTML (`extract_mre_xml`) |
| `agent/` | opt-in agentic RAG loop — see [Agentic RAG](#agentic-rag), above |

## License

MIT — see [LICENSE](LICENSE).
