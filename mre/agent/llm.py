from __future__ import annotations

"""
A thin wrapper around a single OpenAI-compatible client that handles both
action generation (JSON-schema-enforced) and free-text generation. Same
convention as mre.generate_mre(): the caller always passes client and
model, and the library never hardcodes a default model or backend.

Ported from core/mre.py's OpenAIGuidedLLM into this library's distribution
boundary. The in-process vLLM version (GuidedLLM, which depends on
config.py global constants and local GPU model loading) is not ported: to
use local vLLM, stand it up as an OpenAI-compatible server and point
client's base_url at it.
"""

import time

import openai

from mre.llm_util import _accumulate_usage, _new_stats

_TEMPERATURE = 0.0
MAX_ANSWER_TOKENS = 1024


async def generate_action(
    client: openai.AsyncOpenAI, model: str, messages: list[dict], schema: dict,
) -> tuple[str, dict]:
    """Generate one JSON action constrained by schema. Returns (raw_content, usage_stats).

    vLLM's OpenAI-compatible server (0.6+) supports guided decoding via
    response_format={"type":"json_schema",...}, but some gateways reject
    json_schema outright and only accept json_object plus
    extra_body.guided_json. Both are tried in order for broad compatibility.
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
    # kwargs is built as a plain dict (schema/extra_body vary per fallback branch below),
    # not the SDK's typed overloads -- same deliberate looseness as generation.py's
    # call_llm_async() for backend portability across OpenAI-compatible gateways.
    try:
        resp = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
    except TypeError:
        # A very old client that doesn't support json_schema: fall back to guided_json
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = {"guided_json": schema}
        resp = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
    except Exception as e:
        err_s = str(e).lower()
        if "json_schema" in err_s or "response_format" in err_s:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["extra_body"] = {"guided_json": schema}
            resp = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        elif "guided_decoding_backend" in err_s or "guided-decoding-backend" in err_s:
            kwargs["extra_body"] = {"guided_json": schema}
            resp = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        elif "guided_json" in err_s or "extra" in err_s:
            # The gateway itself doesn't support extra_body: fall back to plain JSON mode
            kwargs.pop("extra_body", None)
            resp = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        else:
            raise
    stats = _new_stats()
    _accumulate_usage(stats, resp, time.perf_counter() - t0)
    return (resp.choices[0].message.content or "").strip(), stats


async def generate_text(
    client: openai.AsyncOpenAI, model: str, messages: list[dict], *, max_tokens: int = 512,
) -> tuple[str, dict]:
    """Generate free-form text (used to force an answer once the turn budget is exhausted). Returns (text, usage_stats)."""
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=max_tokens,  # type: ignore[arg-type]
        temperature=_TEMPERATURE,
    )
    stats = _new_stats()
    _accumulate_usage(stats, resp, time.perf_counter() - t0)
    return (resp.choices[0].message.content or "").strip(), stats
