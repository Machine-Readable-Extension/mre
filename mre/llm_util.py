from __future__ import annotations

"""
Low-level LLM call utilities: response parsing, usage/latency accounting,
context budget calculation, chunking.

Ported the same logic that lived in data_utils/mre_generator.py (v1) into
this library's distribution boundary, so the rest of the mre package
doesn't depend on data_utils/. Cost reporting (_build_cost_record and
friends, CLI-only) is not ported: generate_mre() doesn't use it.
"""

import json
import logging
import re
import textwrap

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Token budget constants
# ─────────────────────────────────────────────
MAX_TOKENS = 16384               # output token ceiling to request (whenever the model context allows it)
MODEL_CTX = 32768                # default model context length; caller overrides to match the actual model
TOKEN_SAFETY_MARGIN = 512        # chat template / rounding margin
MIN_OUTPUT_TOKENS = 2048         # below this, treat the input as too large and skip


def _parse_openai_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from OpenAI response."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _extract_content(response, label: str, requested_max: int) -> str:
    choice = response.choices[0]
    if choice.finish_reason == "length":
        log.warning("  Response truncated by max_tokens=%d [%s]", requested_max, label)
    return choice.message.content


def _new_stats() -> dict:
    """Accumulator for LLM call cost: tokens, call count, latency."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0, "llm_latency_sec": 0.0}


def _accumulate_usage(stats: dict, response, dt: float) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        stats["prompt_tokens"]     += getattr(usage, "prompt_tokens", 0) or 0
        stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    stats["llm_calls"]       += 1
    stats["llm_latency_sec"] += dt


def _merge_stats(dst: dict, src: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens", "llm_calls", "llm_latency_sec"):
        dst[k] += src[k]


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3 chars/token, a safe ratio for mixed EN/KO text)."""
    return len(text) // 3


def _compute_max_output(messages: list[dict], model_ctx: int) -> int:
    """Output token budget remaining after subtracting the estimated input, capped at MAX_TOKENS."""
    input_est = sum(_estimate_tokens(m["content"]) for m in messages) + 100  # chat template overhead
    available = model_ctx - input_est - TOKEN_SAFETY_MARGIN
    return min(MAX_TOKENS, available)


class _InputTooLarge(Exception):
    """Raised when the input nearly fills the model context, leaving no room for output."""


def _resolve_max_tokens(messages: list[dict], model_ctx: int, label: str) -> int:
    max_tok = _compute_max_output(messages, model_ctx)
    if max_tok < MIN_OUTPUT_TOKENS:
        input_est = model_ctx - max_tok - TOKEN_SAFETY_MARGIN
        raise _InputTooLarge(
            f"input ~{input_est} tokens leaves only {max_tok} for output "
            f"(< MIN {MIN_OUTPUT_TOKENS}); model_ctx={model_ctx}"
        )
    return max_tok


# ─────────────────────────────────────────────
# Reference prompt (v1), for chunk-budget estimation only. Never used for an actual generation call.
# ─────────────────────────────────────────────
# Carried over as-is from data_utils/mre_generator.py's (v1) SYSTEM_PROMPT/build_user_prompt.
# Its only purpose is estimating the fixed overhead in
# _chunk_p_nodes_by_token_budget(); see the module docstring above.

_V1_BUDGET_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert technical document analyst and structural metadata engineer. Your task is to generate a Machine-Readable Extension (MRE)—a highly structured, hierarchical JSON header—for a given HTML document.

You will be provided with:
1. The title of the document.
2. A list of paragraph nodes, each containing a unique `id` and its corresponding plain text.

Based on this input, you must analyze the document's overall narrative flow and generate a comprehensive summary. Furthermore, for every individual paragraph node, you must synthesize a concise heading and extract core keywords that accurately represent its specific content.

CRITICAL INSTRUCTION:
Return ONLY a single valid JSON object. Do NOT wrap the JSON in markdown fences (e.g., ```json ... ```) and do NOT include any introductory or concluding commentary.

Output Schema:
{
  "lang": "<ISO-639-1 language code, e.g., 'en' or 'ko'>",
  "summary": "<A clear, 2–4 sentence summary of the entire document's core message>",
  "tags": {
    "source_type": "<Select EXACTLY ONE from: encyclopedia_article, academic_paper, technical_report, blog_post, official_documentation, news_article, other>",
    "topic": ["<3–5 broad topic keywords representing the overall document>"]
  },
  "nodes": [
    {
      "id": "<Exact paragraph id from the input, e.g., 'p3'>",
      "heading": "<Heading naming the main entity of this paragraph (25-80 chars)>",
      "keywords": "<Comma-separated 4-5 distinctive proper nouns from this paragraph, no years/numbers (15-80 chars)>"
    }
  ]
}

Strict Rules & Constraints:
1. Exact Node Match: The `nodes` array MUST contain exactly one entry for every paragraph node provided in the input. You must preserve the exact same order and use the exact same `id`. Do NOT add, remove, or alter any `id`.
2. Length Limits: `heading` must be 25-80 characters; `keywords` must be 15-80 characters.
3. heading — Main Entity + Its Role/Relation Required:
   - MUST name BOTH (a) the main entity (person, place, organization, work, event, or concept) AND (b) the entity's role or relation in this paragraph — i.e. what the paragraph asserts ABOUT that entity.
   - Use a short relation phrase such as: "birthplace of", "directed by", "filming location of", "founded in", "won by", "married to", "composed by", "performer of", "located in", "headquartered in", "discovered by", "succeeded by", "based on", "owned by", "captured by", "released in", "stars", "killed", "designed", "destroyed", "rebuilt as".
   - This relation phrase is what enables a downstream RAG agent to follow multi-hop chains. Do NOT omit it even if the entity is named in keywords.
   - DO NOT use generic placeholders such as "Overview", "Introduction", "Background", "Details", "Summary", "Navigation Links", "Team Overview", "Article Status".
4. keywords — Distinctive Proper Nouns Only:
   - Include the distinctive named entities that actually appear in the paragraph: person names, place names, organization names, work titles, and any rare or distinguishing terms.
   - 4-5 comma-separated phrases. Prefer fewer, more salient entities over exhaustive lists.
   - DO NOT include years, dates, or numeric values (e.g. "1946", "2020", "No. 7", "1737"). Years and numbers belong in the paragraph itself, not in the keywords.
   - DO NOT pad with generic words such as "history", "overview", "navigation", "wikipedia stub", "v t e".
5. Specificity: The `heading` and `keywords` for each node must be highly specific to that exact paragraph, acting as a precise "semantic pointer" for a downstream RAG agent to decide whether to fetch that block."""
).strip()


def _v1_budget_user_prompt(url_hint: str, p_nodes: list[dict]) -> str:
    lines = [f"Document URL/hint: {url_hint}\n", "Nodes:"]
    para_idx = 0
    for node in p_nodes:
        if node.get("type") == "heading":
            prefix = "#" * node.get("level", 1)
            lines.append(f"\n{prefix} {node['text']}")
        else:
            para_idx += 1
            lines.append(f"\n[{para_idx}] id={node['id']}")
            lines.append(f"    text: {node['text']}")
    return "\n".join(lines)


def _chunk_p_nodes_by_token_budget(
    p_nodes: list[dict],
    title: str,
    model_ctx: int,
) -> list[list[dict]]:
    """Split p_nodes (a mixed list of headings and paragraphs) to fit the LLM input budget.

    At a chunk boundary, the currently open heading stack is replayed as the
    next chunk's prefix, so the LLM doesn't lose section context (headings
    have no id, so a duplicate is harmless).

    The fixed-overhead estimate (fixed_messages) uses
    _V1_BUDGET_SYSTEM_PROMPT/_v1_budget_user_prompt (carried over unchanged
    from v1 mre_generator.py's SYSTEM_PROMPT/build_user_prompt), not the
    actual call prompt (SYSTEM_PROMPT etc.), because
    data_utils/mre_generator3.py reuses this chunker from v1 unmodified. The
    v1 prompt is longer than v2's, so the estimate is somewhat conservative
    (chunks come out slightly smaller), but reproducing mre_generator3.py's
    exact chunk boundaries means keeping this approximation as is. In real
    operation with a large model_ctx, most documents fit in a single chunk
    anyway, so this has no practical effect.

    Returns
    -------
    list[list[dict]] : each chunk is a p_nodes slice that can be sent to the
                       LLM on its own. A single-element list [p_nodes] means
                       one call suffices.
    """
    fixed_messages = [
        {"role": "system", "content": _V1_BUDGET_SYSTEM_PROMPT},
        {"role": "user",   "content": _v1_budget_user_prompt(title, [])},
    ]
    fixed_input_tok  = sum(_estimate_tokens(m["content"]) for m in fixed_messages) + 100
    output_reserve   = MIN_OUTPUT_TOKENS + TOKEN_SAFETY_MARGIN
    nodes_budget     = model_ctx - fixed_input_tok - output_reserve

    if nodes_budget <= 0:
        # Chunking is impossible if the system prompt alone nearly fills the context.
        return [p_nodes]

    def _node_tok(n: dict) -> int:
        # Approximates structural overhead (position index, newlines, etc.)
        # with fixed padding rather than measuring it, erring conservative.
        return _estimate_tokens(n.get("text", "")) + 30

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tok = 0
    open_headings: list[dict] = []  # stack of currently open headings, levels increase monotonically

    for node in p_nodes:
        # The heading stack must be updated *before* the split decision so
        # the update is reflected in the next chunk's replay.
        if node.get("type") == "heading":
            level = node.get("level", 1)
            while open_headings and open_headings[-1].get("level", 1) >= level:
                open_headings.pop()
            open_headings.append(node)

        ntok = _node_tok(node)

        if current_tok + ntok > nodes_budget and current:
            # Close the current chunk. Drop it if it has no paragraphs (headings only).
            if any(n.get("type") == "paragraph" for n in current):
                chunks.append(current)
            # Start a new chunk, replaying open headings (excluding the node
            # about to be added, if it's itself a heading).
            if node.get("type") == "heading":
                replay = list(open_headings[:-1])
            else:
                replay = list(open_headings)
            current = replay
            current_tok = sum(_node_tok(h) for h in current)

        current.append(node)
        current_tok += ntok

    if current and any(n.get("type") == "paragraph" for n in current):
        chunks.append(current)

    return chunks if chunks else [p_nodes]
