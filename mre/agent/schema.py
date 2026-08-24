"""
Progressive 루프의 4-action(expand_document/fetch_doc/fetch_blocks/answer, 그리고
answer 가로채기 직후 한 턴만 쓰이는 check_sufficiency) guided-decoding JSON schema.

core/mre.py 의 build_progressive_action_schema/build_action_schema(check_sufficiency_only=True)
를 이 라이브러리 배포 경계 안으로 옮겨왔다. sa_mode(answer.content 에 maxLength=150 을 거는
short-answer 전용 분기)는 뺐다 — EM 채점 벤치마크 전용 개념이라 일반 라이브러리에는 안
맞는다(mre.agent.prompts 참조: 답변 형식도 SA/LA 구분 없이 중립 문구 하나로 통일했다).
"""

from __future__ import annotations

MAX_TURNS         = 12   # expand+fetch 가 한 세트라 hop 당 2턴 소모 — 6-hop 예산
MAX_DOCS_PER_TURN = 3
MAX_PIDS_PER_DOC  = 5
MAX_PID_LEN       = 20


def build_progressive_action_schema(
    doc_titles: list[str],
    *,
    has_expanded: bool = False,
    has_retrieved: bool = False,
) -> dict:
    """현재 상태(has_expanded/has_retrieved)에 맞는 schema 를 매 턴 새로 빌드해 넘긴다.

    `expand_document`/`fetch_doc`는 항상 허용(둘 다 metadata 만 보고 내리는 판단 —
    특정 문단이 필요하면 expand, 문서 전체가 필요하면 fetch_doc), `fetch_blocks`는 문서를
    1개 이상 expand 한 뒤에만, `answer`는 무언가(블록 또는 전체 문서)를 1회 이상 retrieve
    한 뒤에만 허용한다.

    has_expanded  : 지금까지 expand_document 로 펼친 문서가 1개 이상 있는가
                     — True 여야 fetch_blocks 브랜치가 열린다.
    has_retrieved : 지금까지 fetch_blocks 또는 fetch_doc 로 실제 콘텐츠를 가져온 적이
                     있는가 — True 여야 answer 브랜치가 열린다.

    fetch_blocks 의 title 은 doc_titles 로 enum 제약(환각 title 차단)하지만, pid 는 enum
    제약하지 않는다 — expand 되지 않은 문서·존재하지 않는 pid 를 요청하면 grammar 가 아니라
    런타임(mre.fetch_block -> 빈 문자열/에러)에서 걸러진다. "expand 후 fetch" 순서는
    프롬프트/워크플로 차원의 권고이지 grammar 강제가 아니다.

    doc_titles : 후보 문서 전체 제목 목록 — expand_document/fetch_doc.titles 와
                 fetch_blocks.title 전부의 enum.
    """
    seen_t: set[str] = set()
    uniq_titles = [t for t in doc_titles if not (t in seen_t or seen_t.add(t))]

    def _titles_action_schema(action_name: str) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": action_name},
                "titles": {
                    "type":     "array",
                    "items":    {"type": "string", "enum": uniq_titles},
                    "minItems": 1,
                    "maxItems": MAX_DOCS_PER_TURN,
                },
            },
            "required": ["action", "titles"],
            "additionalProperties": False,
        }

    branches: list[dict] = [
        _titles_action_schema("expand_document"),
        _titles_action_schema("fetch_doc"),
    ]

    if has_expanded:
        branches.append({
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "fetch_blocks"},
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "enum": uniq_titles},
                            "requests": {
                                "type":     "array",
                                "items":    {"type": "string", "maxLength": MAX_PID_LEN},
                                "minItems": 1,
                                "maxItems": MAX_PIDS_PER_DOC,
                            },
                        },
                        "required": ["title", "requests"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": MAX_DOCS_PER_TURN,
                },
            },
            "required": ["action", "parameters"],
            "additionalProperties": False,
        })

    if has_retrieved:
        branches.append({
            "type": "object",
            "properties": {
                "action":  {"type": "string", "const": "answer"},
                "content": {"type": "string"},
            },
            "required": ["action", "content"],
            "additionalProperties": False,
        })

    return {"oneOf": branches}


# answer 가로채기 직후 한 턴만 주입되는 고정 schema — 상태에 따라 달라지지 않으므로
# build_progressive_action_schema 와 달리 매 턴 새로 빌드할 필요가 없다. missing 은
# is_sufficient=true 일 때도 필드 자체는 필요(빈 문자열 허용) — 조건부 required 분기를
# 피하려 하나의 flat schema 로 유지한다.
CHECK_SUFFICIENCY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action":        {"type": "string", "const": "check_sufficiency"},
        "is_sufficient": {"type": "boolean"},
        "missing":       {"type": "string", "maxLength": 200},
    },
    "required": ["action", "is_sufficient", "missing"],
    "additionalProperties": False,
}
