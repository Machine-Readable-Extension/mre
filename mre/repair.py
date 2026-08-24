"""Misalignment detection and selective regeneration — paragraph granularity only.

Repairs two failure modes in the generation pass's parallel-array output
after the fact:

1. **Count mismatch** — the LLM emits fewer headings/keywords than there
   are paragraphs, leaving the tail paragraphs with an empty desc/keys
   (common on long documents).
2. **Keyword misalignment** — the proper nouns in ``keywords[i]`` don't
   actually appear in the i-th paragraph's text (entities from an
   adjacent paragraph shifted into the wrong slot).

Only the paragraphs affected by one of these are selected and sent back
for regeneration, explicitly anchored by paragraph `id` (an id-keyed
response rather than a positional one) — so the fix can't itself
reintroduce positional drift.
"""

from __future__ import annotations

import logging
import re
import textwrap
import time

import openai

from mre.llm_util import (
    MODEL_CTX,
    _InputTooLarge,
    _accumulate_usage,
    _chunk_p_nodes_by_token_budget,
    _extract_content,
    _merge_stats,
    _new_stats,
    _parse_openai_response,
    _resolve_max_tokens,
)

log = logging.getLogger(__name__)

_MISALIGN_MIN_GROUNDED_RATIO = 0.5   # minimum fraction of a paragraph's keywords that must ground in its text
_MISALIGN_FUZZ_THRESHOLD     = 88    # thefuzz partial_ratio threshold used as a fallback when substring match fails
_MISALIGN_MAX_RETRIES        = 2     # cap on regeneration rounds


def _norm_for_grounding(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def _split_keyword_phrases(keywords: str) -> list[str]:
    return [p.strip() for p in (keywords or "").split(",") if p.strip()]


def _phrase_grounded(phrase: str, para_norm: str, *, fuzz_threshold: int) -> bool:
    """Check whether a keyword phrase actually appears in the paragraph text.

    1. Normalized substring match — handles most proper nouns exactly.
    2. On failure, a thefuzz ``partial_ratio`` fallback absorbs minor
       spelling variation (a possessive, a leading "The", ...). Phrases
       shorter than 4 characters skip the fuzzy fallback, since short
       tokens (e.g. an acronym like "IWW") are prone to false-positive
       fuzzy matches.
    """
    p = _norm_for_grounding(phrase)
    if not p:
        return False
    if p in para_norm:
        return True
    if len(p) < 4:
        return False
    try:
        from thefuzz import fuzz
    except ImportError:
        return False
    return fuzz.partial_ratio(p, para_norm) >= fuzz_threshold


def _keywords_grounded_ratio(keywords: str, para_text: str, *, fuzz_threshold: int) -> tuple[int, int]:
    """Return ``(number of grounded phrases, total phrases)``."""
    phrases = _split_keyword_phrases(keywords)
    if not phrases:
        return 0, 0
    para_norm = _norm_for_grounding(para_text)
    grounded = sum(1 for ph in phrases if _phrase_grounded(ph, para_norm, fuzz_threshold=fuzz_threshold))
    return grounded, len(phrases)


def _paragraph_nodes(nodes: list[dict]) -> list[dict]:
    """Return only the paragraph nodes (headings excluded), in input order.

    The i-th paragraph here corresponds to ``headings[i]``/``keywords[i]``
    — the same mapping ``build_mre_xml`` uses for its ``para_idx``.
    """
    return [n for n in nodes if n.get("type", "paragraph") != "heading"]


def _find_misaligned_indices(
    paragraphs: list[dict],
    keywords: list[str],
    *,
    min_ratio: float,
    fuzz_threshold: int,
    include_misaligned: bool = False,
) -> list[tuple[int, str]]:
    """List the ``(paragraph index, reason)`` pairs that need regeneration.

    Judged from ``keywords`` only — headings are not content-checked.

    - ``"missing"``: the keywords array is shorter than the paragraph
      count, or the entry is an empty string (a count mismatch, i.e. tail
      truncation — since headings/keywords are truncated together in the
      same response, checking keywords alone catches this).
    - ``"misaligned"``: the keywords' grounding ratio is below
      ``min_ratio`` (entities not in this paragraph's text — positional
      drift or hallucination). With ``include_misaligned=False`` (the
      default), this reason is excluded from regeneration — an entity
      that isn't literally in the paragraph is often a useful
      cross-paragraph or document-topic signal for retrieval, and forcing
      literal grounding was found to hurt retrieval precision more than
      it helped.
    """
    bad: list[tuple[int, str]] = []
    for i, para in enumerate(paragraphs):
        k = keywords[i] if i < len(keywords) else ""
        if not (k or "").strip():
            bad.append((i, "missing"))
            continue
        if not include_misaligned:
            continue
        grounded, total = _keywords_grounded_ratio(k, para.get("text", ""), fuzz_threshold=fuzz_threshold)
        if total == 0 or grounded / total < min_ratio:
            bad.append((i, "misaligned"))
    return bad


REGEN_SYSTEM_PROMPT = textwrap.dedent("""
You are refining a Machine-Readable Extension (MRE) index. A previous pass produced
headings/keywords that were MISSING for some paragraphs. You are given ONLY
those paragraphs, each tagged with an exact id. Regenerate a heading and keywords for each,
strictly grounded in that paragraph's OWN text.

Return ONLY a single valid JSON object (no markdown fences, no commentary):
{
  "nodes": [
    {"id": "<the exact id given>", "heading": "<25-50 char heading>", "keywords": "<15-80 char keywords>"},
    ...
  ]
}

Rules:
1. Echo each paragraph's id EXACTLY as given. Produce exactly one object per input paragraph.
2. heading (25-50 chars): name BOTH (a) the main entity AND (b) its role/relation in THIS
   paragraph (e.g. "birthplace of", "directed by", "founded in", "won by"). Do NOT use generic
   words like "Overview", "Introduction", "Background", "Details".
3. keywords (15-80 chars): 4-5 comma-separated distinctive proper nouns that LITERALLY APPEAR
   in THIS paragraph's text. Every entity MUST be a verbatim (or near-verbatim) string from the
   paragraph body shown for that id. Do NOT invent entities, do NOT borrow entities from other
   paragraphs, and do NOT include years, dates, or numeric values.
""").strip()

REGEN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":       {"type": "string", "description": "the exact paragraph id given in input, e.g. 'p7'"},
                    "heading":  {"type": "string", "maxLength": 50},
                    "keywords": {"type": "string", "maxLength": 80},
                },
                "required": ["id", "heading", "keywords"],
            },
        },
    },
    "required": ["nodes"],
}


def _build_regen_response_format(guided: bool) -> dict | None:
    if not guided:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": "mre_regen", "schema": REGEN_JSON_SCHEMA, "strict": True},
    }


def _build_regen_user_prompt(title: str, para_nodes: list[dict]) -> str:
    lines = [
        f"Document title: {title}",
        "",
        "Regenerate the heading and keywords for ONLY these paragraphs. Echo each id EXACTLY.",
        "",
    ]
    for node in para_nodes:
        lines.append(f"[id={node['id']}] {node.get('text', '')}")
        lines.append("")
    return "\n".join(lines)


async def _call_regen_async(
    client: openai.AsyncOpenAI,
    title: str,
    para_nodes: list[dict],
    *,
    model: str,
    guided: bool = False,
    model_ctx: int = MODEL_CTX,
) -> tuple[dict, dict]:
    """One id-anchored regeneration call for a subset of misaligned/missing paragraphs."""
    user_prompt = _build_regen_user_prompt(title, para_nodes)
    messages = [
        {"role": "system", "content": REGEN_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    label = f"{title[:50]} [regen]"
    max_tok = _resolve_max_tokens(messages, model_ctx, label)
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tok,
        messages=messages,
        temperature=0.0,
        response_format=_build_regen_response_format(guided),
    )
    stats = _new_stats()
    _accumulate_usage(stats, response, time.perf_counter() - t0)
    return _parse_openai_response(_extract_content(response, label, max_tok)), stats


async def repair_misaligned_alignment_async(
    client: openai.AsyncOpenAI,
    title: str,
    nodes: list[dict],
    llm_data: dict,
    *,
    model: str,
    guided: bool = False,
    model_ctx: int = MODEL_CTX,
    max_retries: int = _MISALIGN_MAX_RETRIES,
    min_ratio: float = _MISALIGN_MIN_GROUNDED_RATIO,
    fuzz_threshold: int = _MISALIGN_FUZZ_THRESHOLD,
    include_misaligned: bool = False,
) -> tuple[dict, dict, dict]:
    """Find missing/misaligned paragraphs in generated headings/keywords and regenerate just those.

    - Pads/truncates the headings/keywords arrays to the paragraph count,
      normalizing any count mismatch to a plain index.
    - Batches the misaligned/missing paragraphs together for id-anchored
      regeneration, then patches each response back in place by id.
    - Re-checks after each round; stops once a round makes no further
      improvement, or ``max_retries`` is reached.
    - Tallies two separate counts per reason (``"missing"``/``"misaligned"``):
        * ``paragraphs`` — how many paragraphs were flagged for
          regeneration (a paragraph flagged again in a later round is
          counted again).
        * ``llm_calls`` — how many regeneration calls (chunks) actually
          included a paragraph with that reason. A chunk mixing both
          reasons increments both counts by 1.

    Args:
        client: OpenAI-compatible async client.
        title: Document title.
        nodes: The document's heading/paragraph nodes, in order.
        llm_data: The generation pass's output — ``headings``/``keywords``
            (and ``summary``, passed through unchanged).
        model: Model name to pass to ``client``.
        guided: Whether to constrain the regeneration call with a JSON schema.
        model_ctx: Model context length, used for chunking oversized batches.
        max_retries: Maximum number of regeneration rounds.
        min_ratio: Minimum grounded-keyword ratio before a paragraph counts as misaligned.
        fuzz_threshold: Fuzzy-match threshold used by the grounding check.
        include_misaligned: Whether to also regenerate keyword-misaligned
            paragraphs, not just paragraphs missing a heading/keywords
            entirely. Off by default (see the module docstring).

    Returns:
        A ``(repaired_llm_data, stats, regen_counts)`` tuple: a copy of
        ``llm_data`` with corrections applied, accumulated stats for the
        regeneration calls, and per-reason counts
        (``{"missing": {"paragraphs": N, "llm_calls": N}, "misaligned": {...}}``).
    """
    stats = _new_stats()
    regen_counts = {
        "missing":    {"paragraphs": 0, "llm_calls": 0},
        "misaligned": {"paragraphs": 0, "llm_calls": 0},
    }
    paragraphs = _paragraph_nodes(nodes)
    n_para = len(paragraphs)
    if n_para == 0:
        return llm_data, stats, regen_counts

    headings = list(llm_data.get("headings", []))
    keywords = list(llm_data.get("keywords", []))
    # Align both arrays to the paragraph count: pad shortfalls (tail
    # truncation) with empty strings, and truncate any excess (positional
    # drift/duplication).
    headings = (headings + [""] * n_para)[:n_para]
    keywords = (keywords + [""] * n_para)[:n_para]

    id_to_idx = {para["id"]: i for i, para in enumerate(paragraphs)}

    for attempt in range(1, max_retries + 1):
        bad_pairs = _find_misaligned_indices(
            paragraphs, keywords,
            min_ratio=min_ratio, fuzz_threshold=fuzz_threshold,
            include_misaligned=include_misaligned,
        )
        if not bad_pairs:
            break
        for _, reason in bad_pairs:
            regen_counts[reason]["paragraphs"] += 1
        bad = [i for i, _ in bad_pairs]
        reason_by_id = {paragraphs[i]["id"]: reason for i, reason in bad_pairs}
        log.info("  [misalign] %s: regenerating %d/%d paragraphs (round %d/%d)",
                 title[:40], len(bad), n_para, attempt, max_retries)

        bad_nodes = [dict(paragraphs[i]) for i in bad]
        try:
            chunks = _chunk_p_nodes_by_token_budget(bad_nodes, title, model_ctx)
        except _InputTooLarge:
            # A single paragraph alone exceeds the context — fall back to processing it individually.
            chunks = [[bn] for bn in bad_nodes]

        changed = False
        for chunk in chunks:
            if not chunk:
                continue
            # +1 llm_calls for every reason present in this chunk (= one LLM call).
            reasons_in_chunk = {reason_by_id[n["id"]] for n in chunk if n["id"] in reason_by_id}
            for reason in reasons_in_chunk:
                regen_counts[reason]["llm_calls"] += 1
            try:
                res, st = await _call_regen_async(
                    client, title, chunk, model=model, guided=guided, model_ctx=model_ctx,
                )
            except Exception as e:  # noqa: BLE001 — a failed regen call degrades to keeping the original
                log.warning("  [misalign] regen call failed [%s]: %s", title[:40], e)
                continue
            _merge_stats(stats, st)
            for item in (res.get("nodes") or []):
                pid = str(item.get("id", "")).strip()
                idx = id_to_idx.get(pid)
                if idx is None:
                    continue
                old_h, old_k = headings[idx], keywords[idx]
                new_h = (item.get("heading") or "").strip()
                new_k = (item.get("keywords") or "").strip()
                if new_h:
                    headings[idx] = new_h
                if new_k:
                    keywords[idx] = new_k
                if headings[idx] != old_h or keywords[idx] != old_k:
                    changed = True
        if not changed:
            # No actual change this round (the LLM returned the same result) — stop to avoid looping forever.
            break

    new_data = dict(llm_data)
    new_data["headings"] = headings
    new_data["keywords"] = keywords
    return new_data, stats, regen_counts
