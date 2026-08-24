from __future__ import annotations

"""
Misalignment detection & selective regeneration — paragraph granularity only.

v2 parallel-array 출력의 두 가지 실패 모드를 사후 보정한다:
  (1) count mismatch — LLM 이 문단 수보다 적은 headings/keywords 를 내보내 뒤쪽 문단이
      빈 desc/keys 로 남는다 (긴 문서 꼬리 공백).
  (2) keyword misalignment — keywords[i] 의 고유명사가 실제로 i번째 문단 본문에 없음
      (인접 문단 엔티티가 위치 밀림으로 들어온 경우).
두 경우에 해당하는 문단만 골라, 문단 id 를 명시적으로 anchor 한 채 (위치 기반이 아닌
id 기반 응답) 재생성 요청한다 → 위치 밀림 재발 없이 해당 문단만 교체.

data_utils/mre_generator3.py의 동명 섹션을 이 라이브러리 배포 경계 안으로 옮겨왔다.
원본은 section-granularity(v3 --granularity section) 호출자와 이 함수를 공유하기 위해
unit_filter/unit_label/regen_system_prompt/regen_schema 를 매개변수화했지만, 이 라이브러리는
paragraph 전용이라 그 일반화는 걷어내고 paragraph 규칙(REGEN_SYSTEM_PROMPT/REGEN_JSON_SCHEMA/
_paragraph_nodes)을 직접 사용한다.
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

_MISALIGN_MIN_GROUNDED_RATIO = 0.5   # 문단 keywords 중 본문에 grounding 돼야 하는 최소 비율
_MISALIGN_FUZZ_THRESHOLD     = 88    # substring 실패 시 thefuzz partial_ratio fuzzy fallback 임계
_MISALIGN_MAX_RETRIES        = 2     # 재생성 라운드 수 상한


def _norm_for_grounding(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def _split_keyword_phrases(keywords: str) -> list[str]:
    return [p.strip() for p in (keywords or "").split(",") if p.strip()]


def _phrase_grounded(phrase: str, para_norm: str, *, fuzz_threshold: int) -> bool:
    """keyword phrase 가 문단 본문에 실제로 등장하는지 검사.

    1) 정규화 substring — 대부분의 고유명사 정확 매칭.
    2) 실패 시 thefuzz partial_ratio fuzzy fallback — 소유격/관사 접두어("The ", "'s")
       같은 사소한 표기 변형을 흡수. 너무 짧은 토큰의 fuzzy 오탐은 길이 하한으로 차단.
    """
    p = _norm_for_grounding(phrase)
    if not p:
        return False
    if p in para_norm:
        return True
    if len(p) < 4:  # 짧은 약어(IWW 등)는 fuzzy 오탐 위험 → substring 실패 시 미grounding 처리
        return False
    try:
        from thefuzz import fuzz
    except ImportError:
        return False
    return fuzz.partial_ratio(p, para_norm) >= fuzz_threshold


def _keywords_grounded_ratio(keywords: str, para_text: str, *, fuzz_threshold: int) -> tuple[int, int]:
    """(grounding 된 phrase 수, 전체 phrase 수)."""
    phrases = _split_keyword_phrases(keywords)
    if not phrases:
        return 0, 0
    para_norm = _norm_for_grounding(para_text)
    grounded = sum(1 for ph in phrases if _phrase_grounded(ph, para_norm, fuzz_threshold=fuzz_threshold))
    return grounded, len(phrases)


def _paragraph_nodes(nodes: list[dict]) -> list[dict]:
    """heading 을 제외한 문단 노드만 입력 순서대로 반환.
    i번째 문단이 headings[i]/keywords[i] 에 대응한다 (build_mre_xml의 para_idx 매핑과 동일)."""
    return [n for n in nodes if n.get("type", "paragraph") != "heading"]


def _find_misaligned_indices(
    paragraphs: list[dict],
    keywords: list[str],
    *,
    min_ratio: float,
    fuzz_threshold: int,
    include_misaligned: bool = False,
) -> list[tuple[int, str]]:
    """재생성이 필요한 (문단 index, 사유) 목록. (keywords 기준만 — heading 은 내용 검증 안 함.)

    - "missing"    : keywords 배열이 문단보다 짧거나 해당 항목이 빈 문자열
                      (count mismatch → 꼬리 공백 포함. headings/keywords 는 같은 응답에서
                       함께 잘리므로 keywords 공백 검사만으로 tail truncation 이 잡힌다.)
    - "misaligned" : keywords 의 grounding 비율이 min_ratio 미만
                      (본문에 없는 엔티티 = 위치 밀림/환각). include_misaligned=False (기본)
                      에서는 이 사유는 재생성 대상에 넣지 않는다 — 문단 본문에 없는
                      "이웃 문단/문서 topic" 엔티티가 cross-paragraph 검색 신호로 유용한
                      경우가 많아, 리터럴 grounding 강제가 오히려 retrieval precision 을
                      떨어뜨리는 회귀가 확인됐다.
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
    """오정렬/누락 문단 하위집합에 대해 id-anchored 재생성 1회 호출."""
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
    """생성된 headings/keywords 에서 누락/오정렬 문단을 찾아 그 문단만 재생성한다.

    - headings/keywords 배열을 문단 수에 맞춰 pad/truncate → count mismatch 를 index 로 정규화.
    - 오정렬/누락 문단만 batch 로 묶어 id-anchored 재생성, 응답 id 로 제자리에 patch.
    - 라운드마다 재검증하며 개선이 없거나 max_retries 도달 시 종료.
    - 사유("missing"/"misaligned")별로 두 가지 카운트를 별도로 누계한다:
        * paragraphs — 재생성 대상으로 걸린 문단 수 (같은 문단이 여러 라운드에 걸쳐
          계속 걸리면 그만큼 여러 번 카운트).
        * llm_calls  — 그 사유의 문단을 포함해 실제로 나간 "LLM 호출(청크)" 수. 한 청크에
          missing/misaligned 문단이 섞여 있으면 두 사유 모두에 +1.

    Returns
    -------
    (보정된 llm_data 사본, 재생성 호출 누계 stats,
     사유별 카운트 {"missing": {"paragraphs": N, "llm_calls": N}, "misaligned": {...}})
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
    # 배열을 단위 수에 정렬 — 부족분은 빈 문자열 pad(꼬리 누락), 초과분은 절단(위치 밀림/중복).
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
        log.info("  [misalign] %s: %d/%d 문단 재생성 (round %d/%d)",
                 title[:40], len(bad), n_para, attempt, max_retries)

        bad_nodes = [dict(paragraphs[i]) for i in bad]
        try:
            chunks = _chunk_p_nodes_by_token_budget(bad_nodes, title, model_ctx)
        except _InputTooLarge:
            # 단일 문단이 통째로 컨텍스트 초과 시 문단별 개별 처리로 fallback.
            chunks = [[bn] for bn in bad_nodes]

        changed = False
        for chunk in chunks:
            if not chunk:
                continue
            # 이 청크(=LLM call 1회)에 섞여 있는 사유들 전부에 call 카운트 +1.
            reasons_in_chunk = {reason_by_id[n["id"]] for n in chunk if n["id"] in reason_by_id}
            for reason in reasons_in_chunk:
                regen_counts[reason]["llm_calls"] += 1
            try:
                res, st = await _call_regen_async(
                    client, title, chunk, model=model, guided=guided, model_ctx=model_ctx,
                )
            except Exception as e:  # noqa: BLE001 — 재생성 실패는 원본 유지로 degrade
                log.warning("  [misalign] regen 호출 실패 [%s]: %s", title[:40], e)
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
            # 이번 라운드에 실제 변경이 없으면 (LLM 이 같은 결과 반환) 무한 반복 방지 위해 종료.
            break

    new_data = dict(llm_data)
    new_data["headings"] = headings
    new_data["keywords"] = keywords
    return new_data, stats, regen_counts
