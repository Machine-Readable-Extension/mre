from __future__ import annotations

"""
Progressive 루프의 4-action(expand_document/fetch_doc/fetch_blocks/answer, 그리고
answer 가로채기 직후 한 턴만 쓰이는 check_sufficiency) guided-decoding JSON schema.

core/mre.py 의 build_progressive_action_schema/build_action_schema(check_sufficiency_only=True)
를 이 라이브러리 배포 경계 안으로 옮겨왔다. sa_mode(answer.content 에 maxLength=150 을 거는
short-answer 전용 분기)는 뺐다 — EM 채점 벤치마크 전용 개념이라 일반 라이브러리에는 안
맞는다(mre.agent.prompts 참조: 답변 형식도 SA/LA 구분 없이 중립 문구 하나로 통일했다).
"""

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
    """Build and hand back the schema matching the current state (has_expanded/has_retrieved), freshly built each turn.

    `expand_document`/`fetch_doc` are always allowed (both are judgment calls
    made from metadata alone — expand when a specific paragraph is needed,
    fetch_doc when the whole document is needed); `fetch_blocks` is only
    allowed after at least one document has been expanded; `answer` is only
    allowed after retrieving something (a block or a whole document) at least
    once.

    has_expanded  : whether at least one document has been expanded so far via
                     expand_document — must be True for the fetch_blocks
                     branch to open up.
    has_retrieved : whether actual content has been fetched so far via
                     fetch_blocks or fetch_doc — must be True for the answer
                     branch to open up.

    fetch_blocks's title is enum-constrained by doc_titles (blocking
    hallucinated titles), but pid is not enum-constrained — requesting an
    unexpanded document or a nonexistent pid is caught at runtime
    (mre.fetch_block -> empty string/error), not by the grammar. The "expand
    then fetch" ordering is a prompt/workflow-level recommendation, not a
    grammar constraint.

    doc_titles : the full list of candidate document titles — the enum for
                 both expand_document/fetch_doc.titles and fetch_blocks.title.
    """
    seen_t: set[str] = set()
    uniq_titles: list[str] = []
    for t in doc_titles:
        if t not in seen_t:
            seen_t.add(t)
            uniq_titles.append(t)

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
