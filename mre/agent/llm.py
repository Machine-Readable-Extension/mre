"""A thin wrapper handling both schema-constrained actions and free-text
generation through a single OpenAI-compatible client.

Follows the same convention as ``mre.generate_mre()``: the caller always
passes ``client`` and ``model`` explicitly, so the library never hardcodes
a default model or backend. To use a local vLLM server, point ``client``'s
``base_url`` at it — it's an OpenAI-compatible server like any other.
"""

from __future__ import annotations

import time

import openai

from mre.llm_util import _accumulate_usage, _new_stats

_TEMPERATURE = 0.0
MAX_ANSWER_TOKENS = 1024


async def generate_action(
    client: openai.AsyncOpenAI, model: str, messages: list[dict], schema: dict,
) -> tuple[str, dict]:
    """Generate one JSON action constrained by ``schema``.

    A vLLM OpenAI-compatible server (0.6+) supports guided decoding via
    ``response_format={"type": "json_schema", ...}``, but some gateways
    reject ``json_schema`` outright and only accept ``json_object`` +
    ``extra_body.guided_json``. This tries both, in order, for broad
    compatibility.

    Args:
        client: OpenAI-compatible async client.
        model: Model name to pass to ``client``.
        messages: Chat messages for this turn.
        schema: JSON schema the response must satisfy.

    Returns:
        A ``(raw_content, usage_stats)`` tuple.
    """
    kwargs = dict(
        model=model,
        messages=messages,
        max_completion_tokens=MAX_ANSWER_TOKENS,
        temperature=_TEMPERATURE,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "mre_agent_action", "schema": schema, "strict": True},
        },
    )
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(**kwargs)
    except TypeError:
        # Very old client: json_schema unsupported -> guided_json fallback
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = {"guided_json": schema}
        resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        err_s = str(e).lower()
        if "json_schema" in err_s or "response_format" in err_s:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["extra_body"] = {"guided_json": schema}
            resp = await client.chat.completions.create(**kwargs)
        elif "guided_decoding_backend" in err_s or "guided-decoding-backend" in err_s:
            kwargs["extra_body"] = {"guided_json": schema}
            resp = await client.chat.completions.create(**kwargs)
        elif "guided_json" in err_s or "extra" in err_s:
            # Gateway doesn't support extra_body at all: JSON-mode only
            kwargs.pop("extra_body", None)
            resp = await client.chat.completions.create(**kwargs)
        else:
            raise
    stats = _new_stats()
    _accumulate_usage(stats, resp, time.perf_counter() - t0)
    return resp.choices[0].message.content.strip(), stats


async def generate_text(
    client: openai.AsyncOpenAI, model: str, messages: list[dict], *, max_tokens: int = 512,
) -> tuple[str, dict]:
    """Generate free-form text (used to force an answer once the turn limit is hit).

    Args:
        client: OpenAI-compatible async client.
        model: Model name to pass to ``client``.
        messages: Chat messages for this turn.
        max_tokens: Maximum completion tokens.

    Returns:
        A ``(text, usage_stats)`` tuple.
    """
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=max_tokens, temperature=_TEMPERATURE,
    )
    stats = _new_stats()
    _accumulate_usage(stats, resp, time.perf_counter() - t0)
    return resp.choices[0].message.content.strip(), stats
