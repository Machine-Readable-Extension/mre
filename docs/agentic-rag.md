# Agentic RAG

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
from mre import DocFormat
from mre.agent import run_agent

async def main():
    client = openai.AsyncOpenAI()

    docs = {
        "Pleasure Cove": {
            "html": embedded_html,  # from generate_mre()'s embedded_html
            "url": "https://en.wikipedia.org/wiki/Pleasure_Cove",  # picks the site adapter
        },
        "Pleasure Cove (press release)": {
            "path": "pleasure_cove.hwpx",  # from generate_mre()'s embedded_path
            "fmt": DocFormat.HWPX,  # or DocFormat.DOCX
        },
        # ... more candidate documents, html and hwpx/docx freely mixed
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

Each `docs` entry is one of two shapes, picked per-title — a single `docs`
dict can mix both:

- html: `"html"` (the MRE-embedded document) + `"url"` (forwarded to
  `fetch_block()` to pick the right site adapter).
- hwpx/docx: `"path"` (the MRE-embedded file, in-place-updated by
  `generate_mre()`'s `embedded_path`) + `"fmt"` (`DocFormat.HWPX` or
  `DocFormat.DOCX`, forwarded to `fetch_opc()`).

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
| `mre.extract_mre_xml_opc(path)` | reads the raw `<mre>` block back out of an embedded hwpx/docx file |

`run_agent()` answers in whatever length and form the question calls for —
there's no short-answer/long-answer mode switch, since that distinction
only matters for exact-match benchmark grading, not for a real consumer of
the answer.

Only progressive disclosure is implemented so far. `mre.agent` uses an
OpenAI-compatible async client the same way `generate_mre()` does (so a
local vLLM server works too, via `base_url`), and only works with formats
that implement `fetch` — currently HTML/Wikipedia and hwpx/docx (see
[Document formats](formats.md)); PDF/HWP raise `NotImplementedError` from
`generate_mre()` itself, so they never reach `run_agent()` as an embeddable
candidate document.
