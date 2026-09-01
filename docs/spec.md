# Specification

`<mre>` is a single element containing `<metadata>` and `<tree>`.

## Root element attributes

- **`version`** — the MRE *schema* version (currently `"1.0"`), not the
  `py-mre` library's own version (see the [changelog](https://github.com/Machine-Readable-Extension/py-mre/blob/master/CHANGELOG.md)
  for that). The schema is meant to stay stable across many library
  releases; it only bumps on a breaking change to the header format itself
  (e.g. renaming `<desc>`/`<keys>`, or changing how `<node id="pN">` is
  addressed). A parser should key format-compatibility decisions off this
  attribute, not off which `py-mre` version generated the header.
- **`generator`** *(optional)* — name of the site adapter that produced this
  header (e.g. `"wikipedia"`). Present only when the source went through a
  registered `HTMLSiteAdapter`; absent for HWPX/DOCX/PDF/HWP and for HTML
  parsed via `html_fallback_adapter=`.
- **`generator-fingerprint`** *(optional)* — a hash of that adapter's
  `extract`/`preprocess`/`assign_ids`/`fetch` logic at generation time (see
  `compute_adapter_fingerprint()`). `fetch_block(..., strict=True)` compares
  it against the currently-installed adapter and raises
  `GeneratorFingerprintMismatch` if the adapter's parsing logic has changed
  since the header was generated, since paragraph IDs may no longer point
  at the same text otherwise.

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
