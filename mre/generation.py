from __future__ import annotations

"""
Core v2 MRE generation logic: the LLM schema/prompts and the actual calls
(paragraph granularity only).

Ported from data_utils/mre_generator3.py's generation path into this
library's distribution boundary. That path takes a document title plus a
paragraph sequence and produces a document summary and two parallel
per-paragraph arrays (headings, keywords); ids are assigned mechanically by
code, following input paragraph order, to save LLM output tokens. The
section-granularity path (--granularity section) is not ported; add it
separately if it's ever needed.
"""

import logging
import textwrap
import time

import openai

from mre.llm_util import (
    MODEL_CTX,
    _accumulate_usage,
    _chunk_p_nodes_by_token_budget,
    _extract_content,
    _merge_stats,
    _new_stats,
    _parse_openai_response,
    _resolve_max_tokens,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Schema (v2): nodes only, no summary/tags/lang wrapper — parallel arrays instead
# ─────────────────────────────────────────────

# document-level summary + two parallel arrays (no node wrapper, no id).
# id is assigned mechanically in post-processing, following input paragraph
# order, to save on LLM generation cost. headings[i] and keywords[i] map to
# the i-th input paragraph; both arrays are the same length as the input
# paragraph count. summary is a doc-level navigation signal, used by the
# agent to judge document relevance before fetching individual paragraphs.
MRE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "maxLength": 500,
            "description": (
                "2-4 sentence summary (up to 500 chars) of the entire document's core "
                "message. Should name the main entity, its key role/relation, and a few "
                "salient facts. Used by downstream agent for document-level relevance "
                "judgment before fetching individual paragraphs."
            ),
        },
        "headings": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 25,
                "maxLength": 50,
                "description": (
                    "Heading (25-50 chars) naming BOTH (a) the main entity AND "
                    "(b) its role/relation in THIS paragraph (e.g. 'birthplace of', "
                    "'directed by', 'filming location of', 'founded in', 'won by'). "
                    "Do NOT use generic phrases like 'Overview', 'Introduction', 'Background'."
                ),
            },
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 15,
                "maxLength": 80,
                "description": (
                    "Comma-separated 4-5 distinctive proper nouns (15-80 chars) "
                    "that appear IN THIS specific paragraph. "
                    "CRITICAL: keywords[i] must correspond to the SAME paragraph as headings[i] "
                    "— do NOT list entities from adjacent paragraphs. "
                    "Do NOT include years, dates, or numeric values."
                ),
            },
        },
    },
    "required": ["summary", "headings", "keywords"],
}


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert metadata extractor for a Machine-Readable Extension (MRE) v2 index.

Input you will receive:
  1. Document title.
  2. Section headings interleaved with paragraph nodes (each paragraph numbered [1], [2], ... in input order).

Your task is to produce:
  (a) one document-level summary (2-4 sentences) that lets a downstream agent decide whether this whole document is relevant to a query;
  (b) for every paragraph node, a concise heading and core keywords that act as a precise semantic pointer for which individual paragraph(s) to fetch.

CRITICAL INSTRUCTION:
Return ONLY a single valid JSON object. Do NOT wrap the JSON in markdown fences and do NOT include any commentary.

Output Schema (summary + two parallel arrays — headings[i] and keywords[i] correspond to the SAME i-th paragraph in input order):
{
  "summary":  "<2-4 sentence summary of the whole document (up to 500 chars)>",
  "headings": ["<heading for paragraph 1>", "<heading for paragraph 2>", ...],
  "keywords": ["<keywords for paragraph 1>", "<keywords for paragraph 2>", ...]
}

Strict Rules & Constraints:
1. Exact Count & Order: BOTH arrays MUST contain EXACTLY one entry per paragraph node provided, in the same order as input. headings[i] and keywords[i] BOTH describe the SAME i-th paragraph. Do NOT generate entries for section headings — only paragraphs.
2. CRITICAL Index Alignment: For each index i, headings[i] and keywords[i] MUST describe the SAME paragraph. Do NOT list entities from adjacent paragraphs in keywords[i]. If a paragraph mentions Merrilee Rush, keywords[i] must be about Merrilee Rush — NOT about Juice Newton's version in a different paragraph.
3. Length Limits: summary must be at most 500 characters; each heading must be 25-50 characters (concise); each keywords string must be 15-80 characters.
4. summary — Document-Level:
   - Name the main entity (person, place, organization, work, event, or concept) that the document is about.
   - State the entity's primary role/significance and 1-2 key facts from the document.
   - 2-4 sentences. Do NOT summarize section by section — produce a single coherent document-level overview.
5. heading — Main Entity + Role/Relation Required:
   - Name BOTH (a) the main entity (person, place, organization, work, event, or concept) AND (b) the entity's role or relation in THIS paragraph (what the paragraph asserts about that entity).
   - Use a short relation phrase such as: "birthplace of", "directed by", "filming location of", "founded in", "won by", "married to", "composed by", "performer of", "located in", "headquartered in", "discovered by", "succeeded by", "based on", "owned by", "captured by", "released in", "stars", "killed", "designed", "destroyed", "rebuilt as".
   - Do NOT use generic placeholders such as "Overview", "Introduction", "Background", "Details", "Summary", "Navigation Links", "Team Overview", "Article Status".
6. keywords — Distinctive Proper Nouns Only, MATCHING the heading field:
   - Distinctive named entities appearing IN THIS paragraph (the same paragraph that heading describes): person/place/organization names, work titles, rare or distinguishing terms.
   - 4-5 comma-separated phrases. Prefer fewer, more salient entities over exhaustive lists.
   - Do NOT include years, dates, or numeric values. Do NOT pad with generic words.
7. Specificity: Each heading and keywords pair (same index) must be highly specific to that exact paragraph; the summary must be document-level (not paragraph-specific).
""").strip()


def build_user_prompt(title: str, p_nodes: list[dict]) -> str:
    """v2 input: document title + section headings + paragraphs (text only, no id)."""
    lines = [f"Document title: {title}", "", "Nodes:"]
    para_idx = 0
    for node in p_nodes:
        if node.get("type") == "heading":
            prefix = "#" * node.get("level", 1)
            lines.append(f"\n{prefix} {node['text']}")
        else:
            para_idx += 1
            # v2 doesn't output id (the LLM never generates one, so the input
            # doesn't need it either). The [N] number maps 1:1 to the output
            # array's index.
            lines.append(f"\n[{para_idx}] {node['text']}")
    return "\n".join(lines)


def _build_response_format(guided: bool) -> dict | None:
    if not guided:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": "mre", "schema": MRE_JSON_SCHEMA, "strict": True},
    }


# ─────────────────────────────────────────────
# LLM calls (async only)
# ─────────────────────────────────────────────

async def call_llm_async(
    client: openai.AsyncOpenAI,
    title: str,
    p_nodes: list[dict],
    *,
    model: str,
    guided: bool = False,
    model_ctx: int = MODEL_CTX,
) -> tuple[dict, dict]:
    user_prompt = build_user_prompt(title, p_nodes)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    label = title[:50]
    max_tok = _resolve_max_tokens(messages, model_ctx, label)
    t0 = time.perf_counter()
    # messages is a plain list[dict] and response_format a plain dict, so this stays
    # backend-portable (vLLM/other OpenAI-compatible servers don't share the official
    # SDK's typed message/schema unions) -- mypy can't match that against the strict
    # overloaded signature.
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tok,
        messages=messages,
        temperature=0.0,
        response_format=_build_response_format(guided),
    )  # type: ignore[call-overload]
    stats = _new_stats()
    _accumulate_usage(stats, response, time.perf_counter() - t0)
    return _parse_openai_response(_extract_content(response, label, max_tok)), stats


def _merge_llm_chunks(chunk_results: list[dict]) -> dict:
    """Merge v2 chunk responses: summaries are joined with "\\n\\n" per chunk,
    and the headings/keywords arrays are concatenated in chunk order. Since
    each chunk generates its array in its own paragraph order, a plain
    concat preserves the whole document's order (chunk_results is already
    in original chunk order).
    """
    merged_headings: list[str] = []
    merged_keywords: list[str] = []
    summaries: list[str] = []
    for r in chunk_results:
        merged_headings.extend(r.get("headings", []))
        merged_keywords.extend(r.get("keywords", []))
        s = (r.get("summary") or "").strip()
        if s:
            summaries.append(s)
    return {
        "summary": "\n\n".join(summaries),
        "headings": merged_headings,
        "keywords": merged_keywords,
    }


async def call_llm_chunked_async(
    client: openai.AsyncOpenAI,
    title: str,
    p_nodes: list[dict],
    *,
    model: str,
    guided: bool = False,
    model_ctx: int = MODEL_CTX,
) -> tuple[dict, dict]:
    """When the input exceeds model_ctx, split it into paragraph-level chunks, call per chunk, then merge.
    The chunker itself reuses v1's budget estimate unchanged (see
    _chunk_p_nodes_by_token_budget), the same approximation
    mre_generator3.py carries over from v1 without modification."""
    chunks = _chunk_p_nodes_by_token_budget(p_nodes, title, model_ctx)
    if len(chunks) == 1:
        return await call_llm_async(
            client, title, chunks[0], model=model, guided=guided, model_ctx=model_ctx,
        )
    log.info("  [chunked] %d nodes → %d chunks [%s]", len(p_nodes), len(chunks), title[:50])
    chunk_results: list[dict] = []
    stats = _new_stats()
    for i, chunk in enumerate(chunks):
        label = f"{title} [chunk {i+1}/{len(chunks)}]"
        res, st = await call_llm_async(
            client, label, chunk, model=model, guided=guided, model_ctx=model_ctx,
        )
        chunk_results.append(res)
        _merge_stats(stats, st)
    return _merge_llm_chunks(chunk_results), stats
