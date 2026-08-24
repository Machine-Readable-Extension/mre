"""mre.agent — an opt-in agentic RAG loop built on top of MRE documents.

The core ``mre`` package deliberately stops at the document standard
(generation + fetch); deciding which ``id``s to request, turn by turn, is
a separate concern. This subpackage is for anyone who wants that loop too.

Only one retrieval strategy is implemented so far: progressive disclosure
(a metadata-only first stage, then per-document expansion). It can be used
two ways:

1. Directly via ``run_agent()`` — a complete, ready-to-use entry point.
2. By importing individual pieces (``build_progressive_action_schema``,
   ``metadata_view``, ``SYSTEM_PROMPT``, etc.) and wiring them into a
   different agent loop (LangChain, a hand-rolled one, ...) as tools.
"""

from mre.agent.loop import AgentResult, BlockFetchError, MRENotFoundError, run_agent
from mre.agent.prompts import ANSWER_FORMAT, SYSTEM_PROMPT
from mre.agent.schema import (
    CHECK_SUFFICIENCY_SCHEMA,
    MAX_DOCS_PER_TURN,
    MAX_PIDS_PER_DOC,
    MAX_TURNS,
    build_progressive_action_schema,
)
from mre.agent.views import metadata_view

__all__ = [
    "AgentResult",
    "BlockFetchError",
    "MRENotFoundError",
    "run_agent",
    "ANSWER_FORMAT",
    "SYSTEM_PROMPT",
    "CHECK_SUFFICIENCY_SCHEMA",
    "MAX_DOCS_PER_TURN",
    "MAX_PIDS_PER_DOC",
    "MAX_TURNS",
    "build_progressive_action_schema",
    "metadata_view",
]
