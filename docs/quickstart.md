# Quick start

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

## Fetching a paragraph back out

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

## Detecting a stale adapter

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
