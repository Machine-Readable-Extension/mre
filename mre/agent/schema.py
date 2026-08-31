from __future__ import annotations

"""
The guided-decoding JSON schema for the progressive loop's four actions
(expand_document/fetch_doc/fetch_blocks/answer), plus check_sufficiency,
which is injected for exactly one turn right after answer is intercepted.

Ported from core/mre.py's
build_progressive_action_schema/build_action_schema(check_sufficiency_only=True)
into this library's distribution boundary. sa_mode (a short-answer-only
branch that caps answer.content at maxLength=150) was dropped: it's an
EM-scoring-benchmark-specific concept that doesn't fit a general library
(see mre.agent.prompts, where the answer format was also unified into one
neutral phrasing with no SA/LA distinction).
"""

MAX_TURNS         = 12   # expand+fetch form one set, so each hop costs 2 turns: a 6-hop budget
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


# A fixed schema injected for exactly one turn right after answer is
# intercepted. Unlike build_progressive_action_schema, it doesn't depend on
# state, so there's no need to rebuild it every turn. missing is still a
# required field even when is_sufficient=true (an empty string is
# allowed); kept as one flat schema to avoid a conditional-required branch.
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
