# Specification

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
