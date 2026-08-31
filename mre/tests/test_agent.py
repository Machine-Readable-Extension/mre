import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bs4 import BeautifulSoup

from mre import DocFormat, embed_mre_opc, embed_mre_pdf, parse_opc, parse_pdf
from mre.agent import MRENotFoundError, run_agent
from mre.html_site_adapter import get_site_adapter
from mre.xml_builder import build_mre_xml

_URL = "https://en.wikipedia.org/wiki/David_Lightfoot"
_TITLE = "David Lightfoot"
_DOCX_TITLE = "Sample Report"
_PDF_TITLE = "Quarterly Safety Report"


def _make_docs(david_lightfoot_html: str) -> tuple[dict, list[str]]:
    adapter = get_site_adapter(_URL)
    soup = BeautifulSoup(david_lightfoot_html, "lxml")
    adapter.preprocess(soup)
    raw_nodes = adapter.extract(soup)
    adapter.assign_ids(raw_nodes, _TITLE)
    stripped = adapter.strip(raw_nodes)
    para_nodes = [n for n in stripped if n["type"] == "paragraph"]
    mre_xml = build_mre_xml(
        {
            "summary": "David Lightfoot is an Australian film producer.",
            "headings": [f"heading for {n['id']}" for n in para_nodes],
            "keywords": [f"kw-{n['id']}" for n in para_nodes],
        },
        stripped, title=_TITLE,
    )
    embedded_html = adapter.embed(david_lightfoot_html, mre_xml)
    return {_TITLE: {"html": embedded_html, "url": _URL}}, [n["id"] for n in para_nodes]


def _make_opc_docs(path: Path, fmt: DocFormat, title: str) -> tuple[dict, list[str]]:
    """The hwpx/docx counterpart to _make_docs(): updates path in place via
    embed_mre_opc(), then builds a docs dict using run_agent()'s opc
    document schema ({"path", "fmt"})."""
    stripped = parse_opc(path, fmt)
    para_nodes = [n for n in stripped if n["type"] == "paragraph"]
    mre_xml = build_mre_xml(
        {
            "summary": "A short sample report.",
            "headings": [f"heading for {n['id']}" for n in para_nodes],
            "keywords": [f"kw-{n['id']}" for n in para_nodes],
        },
        stripped, title=title,
    )
    embed_mre_opc(path, mre_xml, fmt)
    return {title: {"path": path, "fmt": fmt}}, [n["id"] for n in para_nodes]


def _make_pdf_docs(path: Path, title: str) -> tuple[dict, list[str]]:
    """The pdf version of _make_opc_docs() (the hwpx/docx counterpart):
    updates path in place via embed_mre_pdf(), then builds it using
    run_agent()'s path-based schema ({"path", "fmt"})."""
    stripped = parse_pdf(path)
    para_nodes = [n for n in stripped if n["type"] == "paragraph"]
    mre_xml = build_mre_xml(
        {
            "summary": "A short quarterly safety report.",
            "headings": [f"heading for {n['id']}" for n in para_nodes],
            "keywords": [f"kw-{n['id']}" for n in para_nodes],
        },
        stripped, title=title,
    )
    embed_mre_pdf(path, mre_xml)
    return {title: {"path": path, "fmt": DocFormat.PDF}}, [n["id"] for n in para_nodes]


def _scripted_client(turns: list[dict]) -> SimpleNamespace:
    """Fake AsyncOpenAI-like client that returns each of `turns` in order,
    one per call.completions.create() invocation."""
    call_no = {"n": 0}

    async def _create(*, model, messages, max_completion_tokens, temperature,
                       response_format=None, extra_body=None):
        i = call_no["n"]
        call_no["n"] += 1
        payload = turns[i]
        message = SimpleNamespace(content=json.dumps(payload))
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


@pytest.mark.asyncio
async def test_expand_then_fetch_blocks_then_answer(david_lightfoot_html):
    docs, pids = _make_docs(david_lightfoot_html)
    client = _scripted_client([
        {"action": "expand_document", "titles": [_TITLE]},
        {"action": "fetch_blocks", "parameters": [{"title": _TITLE, "requests": [pids[0]]}]},
        {"action": "answer", "content": "He is an Australian film producer."},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("What does David Lightfoot do?", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "He is an Australian film producer."
    assert result.num_turns == 4
    assert [a["action"] for a in result.action_log] == [
        "expand_document", "fetch_blocks", "answer", "check_sufficiency",
    ]
    assert result.stats["llm_calls"] == 4
    assert pids[0] in result.retrieved_context


@pytest.mark.asyncio
async def test_fetch_doc_shortcut_skips_expand(david_lightfoot_html):
    docs, _ = _make_docs(david_lightfoot_html)
    client = _scripted_client([
        {"action": "fetch_doc", "titles": [_TITLE]},
        {"action": "answer", "content": "Full doc answer."},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "Full doc answer."
    assert result.num_turns == 3


@pytest.mark.asyncio
async def test_check_sufficiency_false_triggers_retry(david_lightfoot_html):
    docs, pids = _make_docs(david_lightfoot_html)
    client = _scripted_client([
        {"action": "expand_document", "titles": [_TITLE]},
        {"action": "fetch_blocks", "parameters": [{"title": _TITLE, "requests": [pids[0]]}]},
        {"action": "answer", "content": "wrong draft"},
        {"action": "check_sufficiency", "is_sufficient": False, "missing": "the actual fact"},
        {"action": "fetch_blocks", "parameters": [{"title": _TITLE, "requests": [pids[1]]}]},
        {"action": "answer", "content": "corrected answer"},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "corrected answer"
    assert result.num_turns == 7


@pytest.mark.asyncio
async def test_answer_before_any_fetch_is_rejected(david_lightfoot_html):
    docs, _ = _make_docs(david_lightfoot_html)
    client = _scripted_client([
        {"action": "answer", "content": "premature"},
        {"action": "expand_document", "titles": [_TITLE]},
        {"action": "fetch_doc", "titles": [_TITLE]},
        {"action": "answer", "content": "real answer"},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "real answer"
    # the premature answer attempt was rejected without consuming a check_sufficiency turn
    assert [a["action"] for a in result.action_log][:2] == ["answer", "expand_document"]


@pytest.mark.asyncio
async def test_no_fetch_ever_refuses_to_answer(david_lightfoot_html):
    docs, _ = _make_docs(david_lightfoot_html)
    # expand_document with an invalid title every turn -> never actually fetches anything
    client = _scripted_client([{"action": "expand_document", "titles": ["NoSuchTitle"]}] * 3)

    result = await run_agent("q", docs, client=client, model="stub", max_turns=3)

    assert not result.success
    assert "fetch" in result.error
    assert result.answer == ""


@pytest.mark.asyncio
async def test_turn_limit_forces_fallback_answer(david_lightfoot_html):
    docs, pids = _make_docs(david_lightfoot_html)

    # fetch once, then stall re-expanding the same (already-expanded) doc forever ->
    # the action-schema calls all go through response_format=json_schema, while the
    # final forced-answer call (llm.generate_text) has no response_format at all --
    # branch on that to know which kind of reply to synthesize.
    async def _create(*, model, messages, max_completion_tokens=None, max_tokens=None,
                       temperature, response_format=None, extra_body=None):
        if response_format is not None:
            if _create.n == 0:
                payload = {"action": "expand_document", "titles": [_TITLE]}
            elif _create.n == 1:
                payload = {"action": "fetch_blocks",
                           "parameters": [{"title": _TITLE, "requests": [pids[0]]}]}
            else:
                payload = {"action": "expand_document", "titles": [_TITLE]}  # no-op, already expanded
            _create.n += 1
            message = SimpleNamespace(content=json.dumps(payload))
        else:
            message = SimpleNamespace(content="Fallback answer from accumulated blocks.")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[choice], usage=usage)

    _create.n = 0
    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = AsyncMock(side_effect=_create)

    result = await run_agent("q", docs, client=client, model="stub", max_turns=4)

    assert result.answer == "Fallback answer from accumulated blocks."
    assert result.success
    assert "max_turns" in result.error


@pytest.mark.asyncio
async def test_mre_not_found_raises(david_lightfoot_html):
    client = _scripted_client([])
    bad_docs = {"NoMRE": {"html": "<html><body>no mre here</body></html>", "url": _URL}}
    with pytest.raises(MRENotFoundError):
        await run_agent("q", bad_docs, client=client, model="stub")


# ─────────────────────────────────────────────
# hwpx/docx docs (opc schema: {"path", "fmt"} instead of {"html", "url"})
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_docx_doc_expand_then_fetch_blocks_then_answer(sample_docx):
    docs, pids = _make_opc_docs(sample_docx, DocFormat.DOCX, _DOCX_TITLE)
    client = _scripted_client([
        {"action": "expand_document", "titles": [_DOCX_TITLE]},
        {"action": "fetch_blocks", "parameters": [{"title": _DOCX_TITLE, "requests": [pids[0]]}]},
        {"action": "answer", "content": "Answer from the docx report."},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("What does the report say?", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "Answer from the docx report."
    assert pids[0] in result.retrieved_context


@pytest.mark.asyncio
async def test_hwpx_doc_fetch_doc_shortcut(sample_hwpx):
    docs, _ = _make_opc_docs(sample_hwpx, DocFormat.HWPX, "Press Release")
    client = _scripted_client([
        {"action": "fetch_doc", "titles": ["Press Release"]},
        {"action": "answer", "content": "Full hwpx doc answer."},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "Full hwpx doc answer."


@pytest.mark.asyncio
async def test_mixed_html_and_docx_docs_in_one_run(david_lightfoot_html, sample_docx):
    """The docs dict must be able to freely mix html/opc schemas per title."""
    html_docs, _ = _make_docs(david_lightfoot_html)
    docx_docs, docx_pids = _make_opc_docs(sample_docx, DocFormat.DOCX, _DOCX_TITLE)
    docs = {**html_docs, **docx_docs}

    client = _scripted_client([
        {"action": "expand_document", "titles": [_DOCX_TITLE]},
        {"action": "fetch_blocks", "parameters": [{"title": _DOCX_TITLE, "requests": [docx_pids[0]]}]},
        {"action": "answer", "content": "mixed-source answer"},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "mixed-source answer"


@pytest.mark.asyncio
async def test_opc_mre_not_found_raises(sample_docx):
    client = _scripted_client([])
    bad_docs = {_DOCX_TITLE: {"path": sample_docx, "fmt": DocFormat.DOCX}}  # never embed_mre_opc()'d
    with pytest.raises(MRENotFoundError):
        await run_agent("q", bad_docs, client=client, model="stub")


# ─────────────────────────────────────────────
# pdf docs (the path-based schema is the same as hwpx/docx: {"path", "fmt"}, fmt=DocFormat.PDF)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pdf_doc_expand_then_fetch_blocks_then_answer(sample_prose_pdf):
    docs, pids = _make_pdf_docs(sample_prose_pdf, _PDF_TITLE)
    client = _scripted_client([
        {"action": "expand_document", "titles": [_PDF_TITLE]},
        {"action": "fetch_blocks", "parameters": [{"title": _PDF_TITLE, "requests": [pids[0]]}]},
        {"action": "answer", "content": "Answer from the pdf report."},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("What does the report say?", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "Answer from the pdf report."
    assert pids[0] in result.retrieved_context


@pytest.mark.asyncio
async def test_pdf_doc_fetch_doc_shortcut(sample_prose_pdf):
    docs, _ = _make_pdf_docs(sample_prose_pdf, _PDF_TITLE)
    client = _scripted_client([
        {"action": "fetch_doc", "titles": [_PDF_TITLE]},
        {"action": "answer", "content": "Full pdf doc answer."},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "Full pdf doc answer."


@pytest.mark.asyncio
async def test_mixed_html_and_pdf_docs_in_one_run(david_lightfoot_html, sample_prose_pdf):
    """The docs dict must be able to freely mix html/path-based (opc, pdf) schemas per title."""
    html_docs, _ = _make_docs(david_lightfoot_html)
    pdf_docs, pdf_pids = _make_pdf_docs(sample_prose_pdf, _PDF_TITLE)
    docs = {**html_docs, **pdf_docs}

    client = _scripted_client([
        {"action": "expand_document", "titles": [_PDF_TITLE]},
        {"action": "fetch_blocks", "parameters": [{"title": _PDF_TITLE, "requests": [pdf_pids[0]]}]},
        {"action": "answer", "content": "mixed-source answer"},
        {"action": "check_sufficiency", "is_sufficient": True, "missing": ""},
    ])

    result = await run_agent("q", docs, client=client, model="stub")

    assert result.success
    assert result.answer == "mixed-source answer"


@pytest.mark.asyncio
async def test_pdf_mre_not_found_raises(sample_prose_pdf):
    client = _scripted_client([])
    bad_docs = {_PDF_TITLE: {"path": sample_prose_pdf, "fmt": DocFormat.PDF}}  # never embed_mre_pdf()'d
    with pytest.raises(MRENotFoundError):
        await run_agent("q", bad_docs, client=client, model="stub")
