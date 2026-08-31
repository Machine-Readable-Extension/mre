from __future__ import annotations

"""
Misalignment detection & selective regeneration — paragraph granularity only.

Post-hoc correction for two failure modes of the v2 parallel-array output:
  (1) count mismatch: the LLM emits fewer headings/keywords than there are
      paragraphs, leaving trailing paragraphs with an empty desc/keys (a
      tail gap on long documents).
  (2) keyword misalignment: the proper nouns in keywords[i] don't actually
      appear in the i-th paragraph's text (an adjacent paragraph's entities
      bled in due to a positional shift).
Only the paragraphs matching one of these are regenerated, with the
paragraph id explicitly anchored in the request (an id-based response
instead of a positional one), so the replacement can't drift again.

Ported from the same-named section of data_utils/mre_generator3.py into
this library's distribution boundary. The original parameterized
unit_filter/unit_label/regen_system_prompt/regen_schema so this function
could be shared with the section-granularity (v3 --granularity section)
caller; since this library is paragraph-only, that generalization is
stripped out and the paragraph rules (REGEN_SYSTEM_PROMPT/REGEN_JSON_SCHEMA/
_paragraph_nodes) are used directly.
"""

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
_MISALIGN_FUZZ_THRESHOLD     = 88    # thefuzz partial_ratio threshold for the fuzzy fallback when substring match fails
_MISALIGN_MAX_RETRIES        = 2     # upper bound on regeneration rounds


def _norm_for_grounding(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def _split_keyword_phrases(keywords: str) -> list[str]:
    return [p.strip() for p in (keywords or "").split(",") if p.strip()]


def _phrase_grounded(phrase: str, para_norm: str, *, fuzz_threshold: int) -> bool:
    """Check whether a keyword phrase actually appears in the paragraph text.

    1) Normalized substring match: catches most proper nouns exactly.
    2) On failure, fall back to thefuzz's partial_ratio fuzzy match, which
       absorbs minor variations like a possessive or leading article
       ("The ", "'s"). A length floor blocks fuzzy false positives on
       tokens that are too short.
    """
    p = _norm_for_grounding(phrase)
    if not p:
        return False
    if p in para_norm:
        return True
    if len(p) < 4:  # short abbreviations (e.g. IWW) risk fuzzy false positives, so treat as ungrounded on substring miss
        return False
    try:
        from thefuzz import fuzz
    except ImportError:
        return False
    return fuzz.partial_ratio(p, para_norm) >= fuzz_threshold


def _keywords_grounded_ratio(keywords: str, para_text: str, *, fuzz_threshold: int) -> tuple[int, int]:
    """Return (number of grounded phrases, total phrase count)."""
    phrases = _split_keyword_phrases(keywords)
    if not phrases:
        return 0, 0
    para_norm = _norm_for_grounding(para_text)
    grounded = sum(1 for ph in phrases if _phrase_grounded(ph, para_norm, fuzz_threshold=fuzz_threshold))
    return grounded, len(phrases)


def _paragraph_nodes(nodes: list[dict]) -> list[dict]:
    """Return only paragraph nodes (headings excluded), in input order.
    The i-th paragraph corresponds to headings[i]/keywords[i], same mapping as build_mre_xml's para_idx."""
    return [n for n in nodes if n.get("type", "paragraph") != "heading"]


def _find_misaligned_indices(
    paragraphs: list[dict],
    keywords: list[str],
    *,
    min_ratio: float,
    fuzz_threshold: int,
    include_misaligned: bool = False,
) -> list[tuple[int, str]]:
    """List of (paragraph index, reason) pairs that need regeneration. Checked against keywords only; heading content is not validated.

    - "missing"    : the keywords array is shorter than the paragraph count,
                      or the entry is an empty string (a count mismatch,
                      including a tail gap: headings/keywords get truncated
                      together in the same response, so checking keywords
                      for emptiness alone catches tail truncation).
    - "misaligned" : the keywords' grounding ratio is below min_ratio (an
                      entity absent from the paragraph text implies a
                      positional shift or hallucination). With
                      include_misaligned=False (the default), this reason is
                      not sent to regeneration: entities that aren't in the
                      paragraph's own text but come from a neighboring
                      paragraph or the document's topic are often a useful
                      cross-paragraph retrieval signal, and forcing literal
                      grounding was found to regress retrieval precision.
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
    """One id-anchored regeneration call for the subset of misaligned/missing paragraphs."""
    user_prompt = _build_regen_user_prompt(title, para_nodes)
    messages = [
        {"role": "system", "content": REGEN_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    label = f"{title[:50]} [regen]"
    max_tok = _resolve_max_tokens(messages, model_ctx, label)
    t0 = time.perf_counter()
    # messages is a plain list[dict] and response_format a plain dict -- same
    # deliberate looseness as generation.py's call_llm_async(), for backend portability.
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tok,
        messages=messages,
        temperature=0.0,
        response_format=_build_regen_response_format(guided),
    )  # type: ignore[call-overload]
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
    """Find missing/misaligned paragraphs in the generated headings/keywords and regenerate only those.

    - Pads/truncates the headings/keywords arrays to the paragraph count,
      normalizing a count mismatch into plain indices.
    - Batches only the misaligned/missing paragraphs into an id-anchored
      regeneration call, then patches the response back in by id.
    - Re-validates each round, stopping once there's no improvement or
      max_retries is reached.
    - Tracks two separate counts per reason ("missing"/"misaligned"):
        * paragraphs: how many paragraphs were flagged for regeneration
          (a paragraph flagged again across multiple rounds is counted each time).
        * llm_calls: how many actual LLM calls (chunks) included a paragraph
          for that reason. A chunk mixing missing and misaligned paragraphs
          increments both.

    Returns
    -------
    (a corrected copy of llm_data, cumulative regeneration-call stats,
     per-reason counts: {"missing": {"paragraphs": N, "llm_calls": N}, "misaligned": {...}})
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
    # Align the arrays to the unit count: pad shortfalls with empty strings
    # (a tail gap), truncate excess (a positional shift or duplicate).
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
            # Fall back to handling paragraphs one at a time if a single
            # paragraph alone exceeds the context.
            chunks = [[bn] for bn in bad_nodes]

        changed = False
        for chunk in chunks:
            if not chunk:
                continue
            # Every reason mixed into this chunk (= one LLM call) gets its call count incremented.
            reasons_in_chunk = {reason_by_id[n["id"]] for n in chunk if n["id"] in reason_by_id}
            for reason in reasons_in_chunk:
                regen_counts[reason]["llm_calls"] += 1
            try:
                res, st = await _call_regen_async(
                    client, title, chunk, model=model, guided=guided, model_ctx=model_ctx,
                )
            except Exception as e:  # noqa: BLE001 — degrade to keeping the original on a regen failure
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
            # No actual change this round (the LLM returned the same result), so stop to avoid looping forever.
            break

    new_data = dict(llm_data)
    new_data["headings"] = headings
    new_data["keywords"] = keywords
    return new_data, stats, regen_counts
