from __future__ import annotations

"""
Progressive 루프 — metadata-only 2단계 공개 방식의 완성형 진입점(run_agent).

core/pipeline.py 의 MREAgent._run_progressive 를 이 라이브러리 배포 경계 안으로 옮겨왔다.
원본과 갈라지는 지점:
  - ablation_mode 분기, dataset 이름 기반 SA/LA 판단, AgentTiming(턴별 상세 계측)은
    뺐다 — 이 서브패키지는 progressive 모드 하나만 다루고, 답변 형식도 SA/LA 구분 없이
    중립 문구 하나로 통일했다(mre.agent.prompts 참조). 대신 mre.llm_util 의 stats
    누적기(prompt/completion 토큰 + 호출 수)를 그대로 재사용한다.
  - `_fetch_blocks_v3`(core/pipeline.py 자체 구현) 대신 mre.fetch_block()/mre.fetch_opc()/
    mre.fetch_pdf() 를 쓴다 — generate_mre() 가 만든 문서를 그대로 소비할 수 있고(html 은
    generator-fingerprint 불일치 감지도 자동으로 딸려온다 — opc(hwpx/docx)/pdf 는 아직 그
    체크가 없다), 문서별 "html" vs "path" 키로 어느 쪽을 쓸지, "path"면 fmt로 opc/pdf 중
    어느 쪽을 쓸지 판단한다(_is_path_doc 참조).
  - GuidedLLM(in-process vLLM)이 아니라 OpenAI-호환 client 하나로 동작한다
    (mre.agent.llm 참조) — mre.generate_mre() 와 동일한 관례.
"""

import html
import json
import unicodedata
from dataclasses import dataclass, field

import openai

from mre import DocFormat, extract_mre_xml_opc, extract_mre_xml_pdf, fetch_block, fetch_opc, fetch_pdf
from mre.agent import llm as _llm
from mre.agent.prompts import ANSWER_FORMAT, SYSTEM_PROMPT
from mre.agent.schema import (
    CHECK_SUFFICIENCY_SCHEMA,
    MAX_DOCS_PER_TURN,
    MAX_PIDS_PER_DOC,
    MAX_TURNS,
    build_progressive_action_schema,
)
from mre.agent.views import metadata_view
from mre.llm_util import _merge_stats, _new_stats
from mre.reader import extract_mre_xml


class MRENotFoundError(Exception):
    """문서 html에서 <mre> 헤더를 찾지 못했을 때."""

    def __init__(self, title: str):
        self.title = title
        super().__init__(f"MRE 헤더 없음: {title}")


class BlockFetchError(Exception):
    """문서를 찾을 수 없거나 요청한 pid 의 fetch 결과가 없을 때."""

    def __init__(self, title: str, pid: str | None = None, reason: str = ""):
        self.title = title
        self.pid = pid
        self.reason = reason
        detail = f"pid={pid!r}" if pid else "문서 없음"
        super().__init__(f"블록 fetch 실패 [{title}] {detail}: {reason}")


@dataclass
class AgentResult:
    answer: str
    retrieved_context: str
    """fetch_blocks/fetch_doc 로 가져온 모든 블록의 합산 텍스트."""
    num_turns: int
    success: bool
    error: str = ""
    action_log: list = field(default_factory=list)
    """각 턴의 LLM 출력 기록. [{"turn": int, "action": str, "raw": str}, ...]"""
    messages: list = field(default_factory=list)
    """에이전트 전체 대화 기록 (system/user/assistant 메시지 리스트)."""
    stats: dict = field(default_factory=_new_stats)
    """누적 토큰/호출 수 — mre.generate_mre() 의 stats 와 동일 형태(mre.llm_util._new_stats)."""


def _normalize_title(title: str) -> str:
    """Title 매칭 정규화 — HTML 엔터티 디코딩 + NFC 정규화 + 공백 정리.
    LLM 이 title 을 echo 할 때 HTML 엔코딩이 섞이거나 유니코드 정규화 형태가 달라지는
    것이 docs dict 의 키와 어긋나는 것을 막는다."""
    title = html.unescape(title or "")
    title = unicodedata.normalize("NFC", title)
    return title.strip()


def _is_path_doc(doc: dict) -> bool:
    """docs[title] 이 html(사이트 어댑터) 문서인지 path 기반(opc: hwpx/docx, 또는 pdf)
    문서인지 판별. schema: html -> {"html", "url"}, path 기반 -> {"path", "fmt"}
    (mre.DocFormat.HWPX/DOCX/PDF) -- 어느 path 기반 포맷인지는 fmt로 다시 갈린다."""
    return "path" in doc


def _extract_raw_mre(doc: dict) -> str | None:
    if not _is_path_doc(doc):
        return extract_mre_xml(doc["html"])
    if doc["fmt"] is DocFormat.PDF:
        return extract_mre_xml_pdf(doc["path"])
    return extract_mre_xml_opc(doc["path"])


def _fetch_one(doc: dict, pid: str) -> str:
    if not _is_path_doc(doc):
        return fetch_block(doc["url"], doc["html"], pid)
    if doc["fmt"] is DocFormat.PDF:
        return fetch_pdf(doc["path"], pid)
    return fetch_opc(doc["path"], pid, doc["fmt"])


def _fetch_blocks(params_list: list[dict], docs: dict[str, dict]) -> str:
    """[{"title": "...", "requests": ["p1", "p3"]}, ...] 형태를 받아 mre.fetch_block()
    (html 문서) 또는 mre.fetch_opc()(hwpx/docx 문서)로 각 단락 텍스트를 가져와 이어붙인다.
    문서가 없거나 pid 결과가 없으면 BlockFetchError."""
    results: list[str] = []
    for param in params_list:
        title = _normalize_title(param.get("title", ""))
        pids = param.get("requests", [])
        if title not in docs:
            raise BlockFetchError(title, reason="문서를 찾을 수 없습니다")
        for pid in pids:
            content = _fetch_one(docs[title], pid)
            if content:
                results.append(f"[{title} :: {pid}]\n{content}")
            else:
                raise BlockFetchError(title, pid=pid, reason="블록 결과 없음")
    return "\n\n".join(results)


@dataclass
class _LoopState:
    """run_agent() 턴 반복 동안 누적되는 가변 상태 -- 액션 핸들러들이 공유해서 읽고 쓴다.
    run_agent() 바깥에 노출되지 않는 내부 전용 컨테이너(AgentResult 와는 별개)."""
    messages: list[dict]
    expanded_titles: set[str] = field(default_factory=set)
    retrieved_blocks: list[str] = field(default_factory=list)
    action_log: list[dict] = field(default_factory=list)
    seen_fetches: set[tuple[str, str]] = field(default_factory=set)
    # answer 가로채기로 보류 중인 초안 -- None 이 아니면 다음 턴은 check_sufficiency 만 허용.
    pending_answer_content: str | None = None
    num_turns: int = 0
    stats: dict = field(default_factory=_new_stats)


def _handle_expand_document(
    state: _LoopState, action: dict, raw_action: str, docs: dict[str, dict], raw_mre_cache: dict[str, str],
) -> None:
    titles_req = action.get("titles", [])[:MAX_DOCS_PER_TURN]
    newly_expanded: list[str] = []
    for t in titles_req:
        nt = _normalize_title(t)
        if nt not in docs or nt in state.expanded_titles:
            continue
        state.expanded_titles.add(nt)
        newly_expanded.append(f"[{nt} MRE — expanded]\n{raw_mre_cache[nt]}")

    state.messages.append({"role": "assistant", "content": raw_action})
    if newly_expanded:
        state.messages.append({"role": "user", "content": (
            "[expanded document structure]\n" + "\n\n".join(newly_expanded) +
            "\n\nSelect paragraph ids with fetch_blocks, or expand another document."
        )})
    else:
        state.messages.append({"role": "user", "content": (
            "[System] No new document was expanded (title already expanded, or "
            "not a valid candidate). Expand a different document, or call "
            "fetch_blocks on an already-expanded document."
        )})


def _handle_fetch_doc(
    state: _LoopState, action: dict, raw_action: str, docs: dict[str, dict], query: str,
) -> None:
    """metadata 만 보고 문서 전체가 필요하다고 판단한 경우."""
    titles_req = action.get("titles", [])[:MAX_DOCS_PER_TURN]
    requested_pairs = {(_normalize_title(t), "full") for t in titles_req}
    new_pairs = requested_pairs - state.seen_fetches
    if not new_pairs:
        state.messages.append({"role": "assistant", "content": raw_action})
        state.messages.append({"role": "user", "content": (
            "[System] All requested document(s) were already fetched in full in "
            "earlier turns; re-fetching does not add new information. Either "
            "fetch_doc a DIFFERENT document, expand_document/fetch_blocks a "
            "specific paragraph, or output an `answer` action."
        )})
        return

    params_list = [{"title": t, "requests": ["full"]} for t, _ in new_pairs if t in docs]
    if not params_list:
        state.messages.append({"role": "assistant", "content": raw_action})
        state.messages.append({"role": "user", "content": (
            "[System] None of the requested titles are valid candidate documents. "
            "Choose a title from the available documents list."
        )})
        return

    block_text = _fetch_blocks(params_list, docs)
    state.seen_fetches |= new_pairs
    state.retrieved_blocks.append(block_text)

    state.messages.append({"role": "assistant", "content": raw_action})
    state.messages.append({"role": "user", "content": (
        f"[full document text]\n{block_text}\n\n"
        f"{ANSWER_FORMAT.format(query=query)}\n"
        "If not, request other documents or paragraphs."
    )})


def _handle_fetch_blocks(
    state: _LoopState, action: dict, raw_action: str, docs: dict[str, dict], query: str,
) -> None:
    params_list = action.get("parameters", [])[:MAX_DOCS_PER_TURN]
    for p in params_list:
        if isinstance(p.get("requests"), list):
            p["requests"] = p["requests"][:MAX_PIDS_PER_DOC]

    requested_pairs: set[tuple[str, str]] = set()
    for p in params_list:
        t_norm = _normalize_title(p.get("title", ""))
        for pid in p.get("requests", []):
            requested_pairs.add((t_norm, pid))

    if requested_pairs and requested_pairs.issubset(state.seen_fetches):
        state.messages.append({"role": "assistant", "content": raw_action})
        state.messages.append({"role": "user", "content": (
            "[System] All requested blocks were already fetched in earlier turns; "
            "re-fetching does not add new information. Either request DIFFERENT "
            "blocks (different doc/id, expanding a new document if needed) or "
            "output an `answer` action using the blocks you already have."
        )})
        return

    block_text = _fetch_blocks(params_list, docs)
    state.seen_fetches |= requested_pairs
    state.retrieved_blocks.append(block_text)

    state.messages.append({"role": "assistant", "content": raw_action})
    state.messages.append({"role": "user", "content": (
        f"[returned block(s)]\n{block_text}\n\n"
        f"{ANSWER_FORMAT.format(query=query)}\n"
        "If not, request other blocks (expand another document if needed)."
    )})


def _handle_check_sufficiency(state: _LoopState, action: dict, raw_action: str) -> AgentResult | None:
    """answer 가로채기 직후에만 등장. 충분하다고 확인되면 여기서 최종 AgentResult 를 반환한다."""
    is_sufficient = bool(action.get("is_sufficient"))
    missing = (action.get("missing") or "").strip()
    state.messages.append({"role": "assistant", "content": raw_action})
    if is_sufficient:
        return AgentResult(
            answer=state.pending_answer_content or "",
            retrieved_context="\n\n".join(state.retrieved_blocks),
            num_turns=state.num_turns, success=True,
            action_log=state.action_log, messages=state.messages, stats=state.stats,
        )
    state.messages.append({"role": "user", "content": (
        f"[System] Noted as insufficient (missing: {missing!r}). "
        "Retrieve the missing information (expand another document, fetch_doc, "
        "or fetch_blocks), then try answering again."
    )})
    state.pending_answer_content = None
    return None


def _handle_answer(state: _LoopState, action: dict, raw_action: str) -> None:
    if not state.retrieved_blocks:
        state.messages.append({"role": "assistant", "content": raw_action})
        state.messages.append({"role": "user", "content": (
            "[System] You must retrieve actual document content first — either "
            "expand a document and call fetch_blocks with relevant paragraph ids, "
            "or call fetch_doc directly if the whole document is needed."
        )})
        return

    # 프롬프트에 미리 알리지 않고, 모델이 낸 answer 초안을 가로채 그 내용 자체를
    # 근거로 check_sufficiency 를 되묻는다 — 다음 턴은 CHECK_SUFFICIENCY_SCHEMA 로 고정.
    draft_content = action.get("content", "")
    state.messages.append({"role": "assistant", "content": raw_action})
    state.messages.append({"role": "user", "content": (
        "[System] Before finalizing, verify: does the evidence you've "
        f"retrieved explicitly and fully support this answer — \"{draft_content}\"? "
        "Call check_sufficiency: if fully supported, set is_sufficient=true; "
        "if something is missing or unverified, set is_sufficient=false and "
        "describe exactly what's missing in `missing`."
    )})
    state.pending_answer_content = draft_content


async def _forced_fallback_answer(
    state: _LoopState, query: str, client: openai.AsyncOpenAI, model: str,
) -> AgentResult:
    """턴 한도 도달 -- 누적 블록만으로 강제 답변(자유 텍스트, JSON 액션 아님)."""
    fallback_instruction = (
        "[System] No more retrieval rounds available. Using ONLY the blocks "
        "fetched so far, give the answer to the question. Output the answer "
        "text only — no JSON, no other words.\n"
        f"Question: {query}"
    )
    state.messages.append({"role": "user", "content": fallback_instruction})
    try:
        final_answer, call_stats = await _llm.generate_text(client, model, state.messages, max_tokens=512)
        _merge_stats(state.stats, call_stats)
    except Exception as e:
        final_answer = ""
        fallback_err = f"forced-answer 생성 실패: {e}"
    else:
        fallback_err = "max_turns 초과 — 누적 블록으로 강제 답변"
    state.action_log.append({"turn": state.num_turns, "action": "forced_answer", "raw": final_answer})

    return AgentResult(
        answer=final_answer, retrieved_context="\n\n".join(state.retrieved_blocks),
        num_turns=state.num_turns, success=bool(final_answer), error=fallback_err,
        action_log=state.action_log, messages=state.messages, stats=state.stats,
    )


async def run_agent(
    query: str,
    docs: dict[str, dict],
    *,
    client: openai.AsyncOpenAI,
    model: str,
    max_turns: int = MAX_TURNS,
) -> AgentResult:
    """Progressive 2단계 공개 방식(metadata-only → 지목한 문서만 전체 공개)으로 query 에 답한다.

    Parameters
    ----------
    query : 사용자 질문.
    docs  : title 별로 두 스키마 중 하나를 섞어서 쓸 수 있다 — 문서마다 원본 포맷이 달라도
            같은 docs dict 안에 같이 넣으면 된다.
              - html : {"html": <mre 가 임베드된 문서 HTML>, "url": <fetch_block() 이
                사이트 어댑터를 고르는 데 쓰는 원본 URL>}. generate_mre(fmt=DocFormat.HTML,
                ...) 결과의 embedded_html 을 그대로 넣으면 된다.
              - hwpx/docx/pdf : {"path": <mre 가 임베드된 hwpx/docx/pdf 파일 경로>,
                "fmt": DocFormat.HWPX, DocFormat.DOCX 또는 DocFormat.PDF}.
                generate_mre(fmt=..., ...) 가 in-place 로 갱신한 embedded_path 를 그대로
                넣으면 된다. fetch_block() 의 generator-fingerprint 불일치 감지는 이
                경로엔 아직 없다.
    client, model : mre.generate_mre() 와 동일한 관례 — OpenAI-호환 비동기 클라이언트와
            모델 이름을 호출자가 직접 넘긴다. 라이브러리는 기본 모델/백엔드를 강제하지 않는다.
    max_turns : 턴 한도. 기본 mre.agent.schema.MAX_TURNS(12) — expand+fetch 한 세트당
            2턴 소모를 감안한 6-hop 예산.
    """
    normalized_titles = [_normalize_title(t) for t in docs]
    docs = {_normalize_title(k): v for k, v in docs.items()}

    # ── 1단계 헤더: 모든 후보 문서의 metadata-only 뷰 ──
    raw_mre_cache: dict[str, str] = {}
    metadata_parts: list[str] = []
    for title in normalized_titles:
        doc = docs[title]
        raw_mre = _extract_raw_mre(doc)
        if not raw_mre:
            raise MRENotFoundError(title)
        raw_mre_cache[title] = raw_mre
        metadata_parts.append(f"[{title} MRE]\n{metadata_view(raw_mre)}")
    all_metadata_text = "\n\n".join(metadata_parts)

    state = _LoopState(messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Available documents: {normalized_titles}\n\n"
                f"[Document metadata]\n{all_metadata_text}\n\n"
                f"{ANSWER_FORMAT.format(query=query)}"
            ),
        },
    ])

    for _ in range(max_turns):
        state.num_turns += 1

        if state.pending_answer_content is not None:
            turn_schema = CHECK_SUFFICIENCY_SCHEMA
        else:
            turn_schema = build_progressive_action_schema(
                normalized_titles,
                has_expanded=bool(state.expanded_titles),
                has_retrieved=bool(state.retrieved_blocks),
            )

        raw_action, call_stats = await _llm.generate_action(client, model, state.messages, turn_schema)
        _merge_stats(state.stats, call_stats)

        try:
            action, _ = json.JSONDecoder().raw_decode(raw_action.strip())
        except json.JSONDecodeError:
            state.action_log.append({"turn": state.num_turns, "action": "parse_error", "raw": raw_action})
            return AgentResult(
                answer="", retrieved_context="\n\n".join(state.retrieved_blocks),
                num_turns=state.num_turns, success=False,
                error=f"JSON 파싱 실패: {raw_action[:300]}",
                action_log=state.action_log, messages=state.messages, stats=state.stats,
            )

        act = action.get("action")
        state.action_log.append({"turn": state.num_turns, "action": act, "raw": raw_action})

        if act == "expand_document":
            _handle_expand_document(state, action, raw_action, docs, raw_mre_cache)
        elif act == "fetch_doc":
            _handle_fetch_doc(state, action, raw_action, docs, query)
        elif act == "fetch_blocks":
            _handle_fetch_blocks(state, action, raw_action, docs, query)
        elif act == "check_sufficiency":
            result = _handle_check_sufficiency(state, action, raw_action)
            if result is not None:
                return result
        elif act == "answer":
            _handle_answer(state, action, raw_action)
        else:
            return AgentResult(
                answer="", retrieved_context="\n\n".join(state.retrieved_blocks),
                num_turns=state.num_turns, success=False,
                error=f"알 수 없는 액션: {act!r}  |  raw={raw_action[:200]}",
                action_log=state.action_log, messages=state.messages, stats=state.stats,
            )

    # ── 방어선: fetch 이력이 전혀 없으면 강제 답변하지 않는다 ──
    if not state.retrieved_blocks:
        return AgentResult(
            answer="", retrieved_context="", num_turns=state.num_turns, success=False,
            error="fetch 미실행 — 답변 거부 (MRE 헤더만으로 답변 시도)",
            action_log=state.action_log, messages=state.messages, stats=state.stats,
        )

    return await _forced_fallback_answer(state, query, client, model)
