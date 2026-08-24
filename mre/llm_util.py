"""
LLM 호출 저수준 유틸리티 — 응답 파싱, 사용량/latency 집계, 컨텍스트 예산 계산, chunking.

data_utils/mre_generator.py(v1)에 있던 것과 동일한 로직을 이 라이브러리 배포 경계 안으로
옮겨왔다 — mre 패키지의 나머지 모듈이 data_utils/에 의존하지 않도록 하기 위함. 비용
리포팅(_build_cost_record 등, CLI 전용)은 포팅 대상이 아니다 — generate_mre()가 쓰지 않는다.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Token budget constants
# ─────────────────────────────────────────────
MAX_TOKENS = 16384             # 요청 가능 출력 토큰 상한 (모델 컨텍스트가 허락하는 한)
MODEL_CTX = 32768               # 기본 모델 컨텍스트 길이 — 실제 모델에 맞춰 호출자가 override
TOKEN_SAFETY_MARGIN = 512       # chat 템플릿/역마진
MIN_OUTPUT_TOKENS = 2048        # 이보다 작으면 입력 과다로 판단해 스킵


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
    """LLM 호출 비용 누적기(토큰/호출수/latency)."""
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
    """Conservative token estimate (~3 chars/token, mixed EN/KO 안전치)."""
    return len(text) // 3


def _compute_max_output(messages: list[dict], model_ctx: int) -> int:
    """입력 추정치를 빼고 남는 출력 토큰 한도. MAX_TOKENS로 캡."""
    input_est = sum(_estimate_tokens(m["content"]) for m in messages) + 100  # 채팅 템플릿 오버헤드
    available = model_ctx - input_est - TOKEN_SAFETY_MARGIN
    return min(MAX_TOKENS, available)


class _InputTooLarge(Exception):
    """입력이 모델 컨텍스트를 거의 다 먹어서 출력 여유가 없을 때 발생."""


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
# Chunk-budget-only 참조 프롬프트 (v1) — 실제 생성 호출에는 절대 쓰지 않는다.
# ─────────────────────────────────────────────
# data_utils/mre_generator.py(v1)의 SYSTEM_PROMPT/build_user_prompt 를 그대로 옮겨왔다.
# 용도는 오직 _chunk_p_nodes_by_token_budget() 의 고정 오버헤드 추정 — 위 docstring 참조.

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
    """p_nodes(heading+paragraph 혼합 리스트)를 LLM 입력 예산에 맞춰 분할한다.

    chunk 경계에서는 현재 open된 heading 스택을 다음 chunk의 prefix로 replay해
    LLM이 섹션 컨텍스트를 잃지 않도록 한다 (heading은 id가 없어 중복돼도 안전).

    고정 오버헤드(fixed_messages) 추정에 실제 호출 프롬프트(SYSTEM_PROMPT 등)가 아니라
    _V1_BUDGET_SYSTEM_PROMPT/_v1_budget_user_prompt(v1 mre_generator.py 의 SYSTEM_PROMPT/
    build_user_prompt 그대로)를 쓴다 — data_utils/mre_generator3.py 가 이 chunker를 v1에서
    변경 없이 그대로 재사용하기 때문. v1 프롬프트가 v2보다 길어 추정치가 다소 보수적(청크가
    약간 작아짐)이지만, mre_generator3.py 와 청크 경계를 정확히 동일하게 재현하려면 이
    근사를 그대로 따라야 한다 — model_ctx 가 큰 실제 운용에서는 문서가 대부분 단일 청크로
    처리돼 영향이 없다.

    Returns
    -------
    list[list[dict]] : 각 chunk는 자체적으로 LLM에 보낼 수 있는 p_nodes 슬라이스.
                       단일 호출이 가능하면 [p_nodes] 단일 원소 리스트.
    """
    fixed_messages = [
        {"role": "system", "content": _V1_BUDGET_SYSTEM_PROMPT},
        {"role": "user",   "content": _v1_budget_user_prompt(title, [])},
    ]
    fixed_input_tok  = sum(_estimate_tokens(m["content"]) for m in fixed_messages) + 100
    output_reserve   = MIN_OUTPUT_TOKENS + TOKEN_SAFETY_MARGIN
    nodes_budget     = model_ctx - fixed_input_tok - output_reserve

    if nodes_budget <= 0:
        # 시스템 프롬프트만으로도 컨텍스트가 가득 찰 정도면 chunking 불가.
        return [p_nodes]

    def _node_tok(n: dict) -> int:
        # 위치 인덱스/개행 등 구조 오버헤드 근사치 — 실측 대신 고정 padding(보수적, 안전 방향).
        return _estimate_tokens(n.get("text", "")) + 30

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tok = 0
    open_headings: list[dict] = []  # 현재 열려있는 heading 중첩 스택 (level 단조 증가)

    for node in p_nodes:
        # heading 스택 갱신은 split 결정 *전에* 수행해야 다음 chunk replay에 반영됨.
        if node.get("type") == "heading":
            level = node.get("level", 1)
            while open_headings and open_headings[-1].get("level", 1) >= level:
                open_headings.pop()
            open_headings.append(node)

        ntok = _node_tok(node)

        if current_tok + ntok > nodes_budget and current:
            # 현재 chunk를 닫는다. paragraph가 하나도 없으면(전부 heading) 버린다.
            if any(n.get("type") == "paragraph" for n in current):
                chunks.append(current)
            # 새 chunk 시작 — heading replay (지금 추가하려는 노드가 heading이면 그 자신은 제외)
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
