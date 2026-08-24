# Machine-Readable-Extension(MRE)

[![Tests](https://github.com/Machine-Readable-Extension/mre/actions/workflows/tests.yml/badge.svg)](https://github.com/Machine-Readable-Extension/mre/actions/workflows/tests.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Machine-Readable-Extension_mre&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Machine-Readable-Extension_mre)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Machine-Readable-Extension_mre&metric=bugs)](https://sonarcloud.io/summary/new_code?id=Machine-Readable-Extension_mre)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Machine-Readable-Extension_mre&metric=coverage)](https://sonarcloud.io/summary/new_code?id=Machine-Readable-Extension_mre)
[![Docs](https://img.shields.io/badge/docs-machine--readable--extension.github.io-blue)](https://machine-readable-extension.github.io/mre/)

**Machine-Readable Extension (MRE)** — a producer-side document standard and
navigation structure that lets LLM agents read a document precisely, instead
of consuming its raw, markup-heavy source.

Full docs, including the auto-generated API reference: **[machine-readable-extension.github.io/mre](https://machine-readable-extension.github.io/mre/)**

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
retrieval-side parser. Full schema: [Specification](https://machine-readable-extension.github.io/mre/spec/).

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

The keyword-grounding [repair pass](https://machine-readable-extension.github.io/mre/quickstart/#repair)
has an optional fuzzy-matching fallback:

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

`generate_mre()`/`fetch_block()` also guard against a document outliving the
adapter that generated it (a `generator-fingerprint` mismatch) — see
[Detecting a stale adapter](https://machine-readable-extension.github.io/mre/quickstart/#detecting-a-stale-adapter)
in the docs.

## Agentic RAG

`mre` core stops at generating and fetching — deciding which `id`s to
request, turn by turn, is a separate concern kept out of the base package.
The opt-in `mre.agent` subpackage implements that loop, using **progressive
disclosure**: every candidate document starts out showing only its
`<metadata>`, and the agent expands specific documents' full `<tree>` (or
fetches them whole) before drilling into individual paragraphs.

```python
from mre.agent import run_agent

result = await run_agent(
    "Who starred in Pleasure Cove?",
    {"Pleasure Cove": {"html": embedded_html, "url": "https://en.wikipedia.org/wiki/Pleasure_Cove"}},
    client=client,
    model="gpt-4o-mini",
)
print(result.answer, result.success, result.stats)
```

Every piece `run_agent()` is built from — the system prompt, the guided-
decoding schemas, the metadata view — is independently importable too, for
wiring MRE into a different agent loop instead. Full walkthrough:
[Agentic RAG](https://machine-readable-extension.github.io/mre/agentic-rag/).

## Supported document formats

| Format | Parsing | Embedding | Fetch |
|---|---|---|---|
| HTML (Wikipedia) | built-in site adapter | `<script type="application/mre+xml">` inside `<head>` | `fetch_block()` |
| HWPX | built-in | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| DOCX | built-in (body paragraphs only — table cells are out of scope) | extra `mre.xml` entry in the zip archive | `fetch_opc()` |
| PDF, HWP | detected (`detect_format`) | not implemented — `generate_mre()` raises `NotImplementedError` | not yet |

HTML support is a **site-adapter registry**, not a generic scraper — only
`wikipedia.org` ships out of the box, but a new site can be registered
in-process or shipped as an installable plugin package (an `mre.site_adapters`
entry point). See [`examples/mre-example-adapter/`](examples/mre-example-adapter)
for a working reference, and
[Document formats](https://machine-readable-extension.github.io/mre/formats/)
for the full guide to adding a site.

## License

MIT — see [LICENSE](LICENSE).
