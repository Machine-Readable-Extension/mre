"""Guided-decoding JSON schemas for the progressive loop's four actions —
``expand_document``/``fetch_doc``/``fetch_blocks``/``answer`` — plus
``check_sufficiency``, injected for a single turn right after ``answer``
is intercepted.
"""

from __future__ import annotations

MAX_TURNS         = 12   # expand+fetch form one pair, so each hop costs 2 turns — a 6-hop budget
MAX_DOCS_PER_TURN = 3
MAX_PIDS_PER_DOC  = 5
MAX_PID_LEN       = 20


def build_progressive_action_schema(
    doc_titles: list[str],
    *,
    has_expanded: bool = False,
    has_retrieved: bool = False,
) -> dict:
    """Build the schema matching the current turn's state.

    ``expand_document``/``fetch_doc`` are always allowed (both are
    decisions made from metadata alone — expand if a specific paragraph is
    needed, fetch_doc if the whole document is). ``fetch_blocks`` is only
    allowed once at least one document has been expanded; ``answer`` is
    only allowed once something (a block or a whole document) has been
    retrieved at least once.

    ``fetch_blocks``'s ``title`` is enum-constrained to ``doc_titles``
    (blocking hallucinated titles), but its paragraph `id` is not
    enum-constrained — requesting an id from an unexpanded document, or
    one that doesn't exist, is caught at runtime (``mre.fetch_block`` ->
    empty string/error) rather than by the grammar. "expand before fetch"
    is a workflow recommendation carried by the prompt, not a
    grammar-enforced rule.

    Args:
        doc_titles: Full list of candidate document titles — the enum for
            ``expand_document``/``fetch_doc``'s ``titles`` and for
            ``fetch_blocks``'s ``title``.
        has_expanded: Whether at least one document has been expanded via
            ``expand_document`` so far. Must be ``True`` to open the
            ``fetch_blocks`` branch.
        has_retrieved: Whether actual content has been retrieved via
            ``fetch_blocks`` or ``fetch_doc`` so far. Must be ``True`` to
            open the ``answer`` branch.

    Returns:
        A ``oneOf`` JSON schema for the current turn.
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


# A fixed schema injected for a single turn right after `answer` is
# intercepted. Unlike build_progressive_action_schema, it doesn't depend
# on turn state, so it never needs rebuilding. `missing` is required even
# when is_sufficient=true (an empty string is fine) — kept as one flat
# schema to avoid a conditionally-required branch.
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
