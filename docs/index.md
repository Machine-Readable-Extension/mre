# Machine-Readable Extension (MRE)

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
anyone who wants it too, not just the primitives — see [Agentic RAG](agentic-rag.md).

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

## Where to go next

- [Install](install.md) — get the package
- [Quick start](quickstart.md) — generate an MRE header and fetch a paragraph back out
- [Agentic RAG](agentic-rag.md) — the opt-in `mre.agent` retrieval loop
- [Document formats](formats.md) — HTML/HWPX/DOCX support and adding a new site
- [Specification](spec.md) — the `<mre>` XML schema
- [API Reference](api.md) — generated from docstrings

## License

MIT — see [LICENSE](https://github.com/Machine-Readable-Extension/mre/blob/master/LICENSE).
