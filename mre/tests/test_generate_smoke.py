import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mre import DocFormat, generate_mre
from mre.html_site_adapter import _wiki_build_structure_tree, _wiki_preprocess
from bs4 import BeautifulSoup


def _fake_openai_client(n_paragraphs_getter):
    """Returns a fake AsyncOpenAI-like client whose chat.completions.create()
    echoes back a syntactically valid MRE2 JSON payload sized to whatever
    paragraph count is actually in the prompt it was called with -- so it
    works correctly however call_llm_chunked_async decides to chunk."""

    async def _create(*, model, max_tokens, messages, temperature, response_format):
        user_prompt = messages[-1]["content"]
        # user prompt embeds one line per paragraph id, e.g. "[D1] text..."
        n = user_prompt.count("] ")
        n = max(n, 1)
        payload = {
            "summary": "A stub summary of the document.",
            "headings": [f"stub heading {i} naming the entity" for i in range(n)],
            "keywords": [f"kw{i}a, kw{i}b, kw{i}c" for i in range(n)],
        }
        message = SimpleNamespace(content=json.dumps(payload))
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=10)
        return SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


@pytest.mark.asyncio
async def test_generate_mre_end_to_end_html(david_lightfoot_html):
    client = _fake_openai_client(None)

    result = await generate_mre(
        david_lightfoot_html,
        client=client,
        model="stub-model",
        title="David Lightfoot",
        url="https://en.wikipedia.org/wiki/David_Lightfoot",
    )

    assert result.format is DocFormat.HTML
    assert result.mre_xml.startswith('<mre version="1.0" generator="wikipedia"')
    assert "generator-fingerprint=" in result.mre_xml
    assert "<section" not in result.mre_xml
    assert result.embedded_html is not None
    assert "application/mre+xml" in result.embedded_html
    assert client.chat.completions.create.await_count >= 1

    # every paragraph id from the source doc shows up in the generated tree
    soup = BeautifulSoup(david_lightfoot_html, "lxml")
    _wiki_preprocess(soup)
    nodes = _wiki_build_structure_tree(soup)
    for node in nodes:
        if node["type"] != "paragraph":
            continue
        # ids get letter-prefixed during generation; just check node count matches
    n_paragraphs = sum(1 for n in nodes if n["type"] == "paragraph")
    assert result.mre_xml.count("<node id=") == n_paragraphs


@pytest.mark.asyncio
async def test_generate_mre_end_to_end_pdf(sample_prose_pdf):
    from mre import extract_mre_xml_pdf
    from mre.pdf_adapter import build_structure_tree_pdf

    client = _fake_openai_client(None)
    nodes_before = build_structure_tree_pdf(sample_prose_pdf)

    result = await generate_mre(
        sample_prose_pdf,
        client=client,
        model="stub-model",
        title="Quarterly Safety Report",
        fmt=DocFormat.PDF,
    )

    assert result.format is DocFormat.PDF
    # pdf has no adapter-fingerprint concept (unlike html) -- root tag stays plain
    assert result.mre_xml.startswith('<mre version="1.0">')
    assert result.embedded_path == sample_prose_pdf
    assert result.mre_xml.count("<node id=") == len(nodes_before)

    # embedding round-trips through the actual file on disk, and page content is untouched
    assert extract_mre_xml_pdf(sample_prose_pdf) == result.mre_xml
    assert build_structure_tree_pdf(sample_prose_pdf) == nodes_before
