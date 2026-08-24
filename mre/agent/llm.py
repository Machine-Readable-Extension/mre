"""
OpenAI-호환 client 하나로 action(JSON schema 강제) / 자유 텍스트 생성 둘 다 처리하는
얇은 래퍼. mre.generate_mre() 와 동일한 관례 — client 와 model 은 호출자가 항상 넘긴다,
라이브러리가 기본 모델/백엔드를 강제하지 않는다.

core/mre.py 의 OpenAIGuidedLLM 을 이 라이브러리 배포 경계 안으로 옮겨왔다. in-process
vLLM 버전(GuidedLLM — config.py 글로벌 상수 + 로컬 GPU 모델 로딩에 의존)은 포팅하지
않는다 — 로컬 vLLM 을 쓰고 싶으면 이미 OpenAI-호환 서버로 띄운 뒤 client 의 base_url 로
가리키면 된다.
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
    """schema 로 제약된 JSON 액션 하나를 생성한다. (raw_content, usage_stats) 반환.

    vLLM OpenAI-호환 서버(0.6+)는 response_format={"type":"json_schema",...} 로 guided
    decoding 을 지원하지만, 일부 게이트웨이는 json_schema 자체를 거부하고 json_object +
    extra_body.guided_json 만 받는다 — 순서대로 시도해서 폭넓게 호환한다.
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
        # 아주 옛 클라이언트: json_schema 미지원 → guided_json fallback
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
            # 게이트웨이 자체가 extra_body 미지원: JSON-mode만
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
    """자유 텍스트를 생성한다 (턴 한도 소진 시 강제 답변용). (text, usage_stats) 반환."""
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=max_tokens, temperature=_TEMPERATURE,
    )
    stats = _new_stats()
    _accumulate_usage(stats, resp, time.perf_counter() - t0)
    return resp.choices[0].message.content.strip(), stats
