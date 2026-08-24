import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bs4 import BeautifulSoup

from mre.agent import MRENotFoundError, run_agent
from mre.html_site_adapter import get_site_adapter
from mre.xml_builder import build_mre_xml

_URL = "https://en.wikipedia.org/wiki/David_Lightfoot"
_TITLE = "David Lightfoot"


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
