"""
mre.agent - an opt-in subpackage for an MRE-based agentic RAG loop.

The `mre` core package deliberately limits itself to the document standard
(generation and fetch): "the agent's turn-by-turn reasoning loop is not this
package's concern" (mre/README.md). This subpackage lives outside that
boundary for users who want an actual loop that reads MRE headers, calls
tools, and produces an answer. It is versioned and experimented on
independently of the core.

Only one strategy is implemented so far: progressive, two-stage
(metadata-only) disclosure. It can be used two ways:
  1. Call run_agent() directly as a ready-made entry point.
  2. Import the individual pieces (build_progressive_action_schema,
     metadata_view, SYSTEM_PROMPT, etc.) and wire them into a different loop
     (e.g. LangChain) as tools, the same pattern the mre core uses by
     exposing both generate_mre() and the lower-level
     HTMLSiteAdapter/fetch_block/build_mre_xml.
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
