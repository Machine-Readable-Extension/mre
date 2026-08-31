from __future__ import annotations

"""
The progressive loop: the complete entry point (run_agent) for the
metadata-only two-stage disclosure approach.

Ported from core/pipeline.py's MREAgent._run_progressive into this
library's distribution boundary. Where this diverges from the original:
  - The ablation_mode branch, dataset-name-based SA/LA classification, and
    AgentTiming (detailed per-turn instrumentation) were dropped: this
    subpackage handles only the progressive mode, and the answer format is
    unified into one neutral phrasing with no SA/LA distinction (see
    mre.agent.prompts). In their place, this reuses mre.llm_util's stats
    accumulator (prompt/completion tokens plus call count) as is.
  - Uses mre.fetch_block()/mre.fetch_opc()/mre.fetch_pdf() instead of
    `_fetch_blocks_v3` (core/pipeline.py's own implementation), so it can
    consume documents produced by generate_mre() directly (html also gets
    generator-fingerprint mismatch detection for free; opc (hwpx/docx)/pdf
    don't have that check yet). It decides which to use per document from
    the "html" vs "path" key, and for "path" documents, which of opc/pdf
    to use based on fmt (see _is_path_doc).
  - Runs on a single OpenAI-compatible client rather than GuidedLLM
    (in-process vLLM) (see mre.agent.llm), the same convention as
    mre.generate_mre().
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
    """Raised when no <mre> header is found in the document html."""

    def __init__(self, title: str):
        self.title = title
        super().__init__(f"No MRE header found: {title}")


class BlockFetchError(Exception):
    """Raised when a document can't be found, or the requested pid has no fetch result."""

    def __init__(self, title: str, pid: str | None = None, reason: str = ""):
        self.title = title
        self.pid = pid
        self.reason = reason
        detail = f"pid={pid!r}" if pid else "document not found"
        super().__init__(f"Block fetch failed [{title}] {detail}: {reason}")


@dataclass
class AgentResult:
    answer: str
    retrieved_context: str
    """The concatenated text of every block fetched via fetch_blocks/fetch_doc."""
    num_turns: int
    success: bool
    error: str = ""
    action_log: list = field(default_factory=list)
    """The LLM output record for each turn. [{"turn": int, "action": str, "raw": str}, ...]"""
    messages: list = field(default_factory=list)
    """The agent's full conversation history (a list of system/user/assistant messages)."""
    stats: dict = field(default_factory=_new_stats)
    """Cumulative token/call counts — same shape as mre.generate_mre()'s stats (mre.llm_util._new_stats)."""


def _normalize_title(title: str) -> str:
    """Normalize a title for matching: decode HTML entities, apply NFC normalization, trim whitespace.
    Prevents an LLM echoing a title with mixed-in HTML encoding or a
    different Unicode normalization form from failing to match the docs
    dict's keys."""
    title = html.unescape(title or "")
    title = unicodedata.normalize("NFC", title)
    return title.strip()


def _is_path_doc(doc: dict) -> bool:
    """Tell whether docs[title] is an html (site-adapter) document or a path-based one (opc: hwpx/docx, or pdf).
    Schema: html -> {"html", "url"}, path-based -> {"path", "fmt"}
    (mre.DocFormat.HWPX/DOCX/PDF); fmt further decides which path-based format it is."""
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
    """Take [{"title": "...", "requests": ["p1", "p3"]}, ...] and fetch each
    paragraph's text via mre.fetch_block() (html documents) or
    mre.fetch_opc() (hwpx/docx documents), concatenating the results.
    Raises BlockFetchError if the document isn't found or a pid has no result."""
    results: list[str] = []
    for param in params_list:
        title = _normalize_title(param.get("title", ""))
        pids = param.get("requests", [])
        if title not in docs:
            raise BlockFetchError(title, reason="document not found")
        for pid in pids:
            content = _fetch_one(docs[title], pid)
            if content:
                results.append(f"[{title} :: {pid}]\n{content}")
            else:
                raise BlockFetchError(title, pid=pid, reason="no block result")
    return "\n\n".join(results)


@dataclass
class _LoopState:
    """Mutable state accumulated across run_agent()'s turn loop, shared for
    reading and writing by the action handlers. An internal-only container
    not exposed outside run_agent() (separate from AgentResult)."""
    messages: list[dict]
    expanded_titles: set[str] = field(default_factory=set)
    retrieved_blocks: list[str] = field(default_factory=list)
    action_log: list[dict] = field(default_factory=list)
    seen_fetches: set[tuple[str, str]] = field(default_factory=set)
    # A draft answer held pending by interception. When not None, the next turn only allows check_sufficiency.
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
    """Handles the case where the agent judged, from metadata alone, that it needs the whole document."""
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
    """Only appears right after an answer is intercepted. Returns the final AgentResult here once confirmed sufficient."""
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

    # Without telling the prompt in advance, intercept the model's answer
    # draft and ask it to check_sufficiency against that content itself;
    # the next turn is locked to CHECK_SUFFICIENCY_SCHEMA.
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
    """Turn limit reached: force an answer from the accumulated blocks alone (free text, not a JSON action)."""
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
        fallback_err = f"forced-answer generation failed: {e}"
    else:
        fallback_err = "max_turns exceeded, forced an answer from accumulated blocks"
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
    """Answer query using the progressive two-stage disclosure approach (metadata-only -> full disclosure only for documents picked out).

    Parameters
    ----------
    query : the user's question.
    docs  : per title, either of two schemas can be mixed — documents with
            different source formats can all live in the same docs dict.
              - html : {"html": <the document HTML with mre embedded>, "url":
                <the original URL fetch_block() uses to pick a site adapter>}.
                Pass generate_mre(fmt=DocFormat.HTML, ...)'s result's
                embedded_html directly.
              - hwpx/docx/pdf : {"path": <the hwpx/docx/pdf file path with mre
                embedded>, "fmt": DocFormat.HWPX, DocFormat.DOCX, or
                DocFormat.PDF}. Pass generate_mre(fmt=..., ...)'s in-place
                updated embedded_path directly. fetch_block()'s
                generator-fingerprint mismatch detection doesn't exist yet for
                this path.
    client, model : same convention as mre.generate_mre() — the caller passes
            an OpenAI-compatible async client and model name directly. The
            library doesn't enforce a default model/backend.
    max_turns : the turn limit. Defaults to mre.agent.schema.MAX_TURNS (12) —
            a 6-hop budget assuming 2 turns consumed per expand+fetch pair.
    """
    normalized_titles = [_normalize_title(t) for t in docs]
    docs = {_normalize_title(k): v for k, v in docs.items()}

    # -- Stage 1 header: the metadata-only view of every candidate document --
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
                error=f"JSON parse failed: {raw_action[:300]}",
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
                error=f"Unknown action: {act!r}  |  raw={raw_action[:200]}",
                action_log=state.action_log, messages=state.messages, stats=state.stats,
            )

    # -- Safety net: never force an answer if fetch was never called --
    if not state.retrieved_blocks:
        return AgentResult(
            answer="", retrieved_context="", num_turns=state.num_turns, success=False,
            error="No fetch was ever run; refusing to answer from the MRE header alone",
            action_log=state.action_log, messages=state.messages, stats=state.stats,
        )

    return await _forced_fallback_answer(state, query, client, model)
