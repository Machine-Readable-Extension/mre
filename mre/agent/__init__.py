"""
mre.agent — MRE 기반 agentic RAG 루프, 옵트인 서브패키지.

`mre` 코어 패키지는 문서 표준(생성 + fetch)만 다루는 게 의도적인 경계다 —
"에이전트의 턴별 추론 루프는 이 패키지의 관심사가 아니다"(mre/README.md). 이 서브패키지는
그 경계 바깥, 실제로 MRE 헤더를 읽고 도구를 호출하며 답을 도출하는 루프를 원하는 사용자를
위한 것 — 코어를 건드리지 않고 독립적으로 버저닝/실험한다.

지금은 progressive(metadata-only 2단계 공개) 방식 하나만 구현한다. 두 가지 방식으로
쓸 수 있다:
  1. run_agent() 하나로 바로 — 완성형 진입점.
  2. 개별 조각(build_progressive_action_schema/metadata_view/SYSTEM_PROMPT 등)을 가져다
     직접 다른 루프(LangChain 등)에 도구로 꽂아 쓰기 — mre 코어가 generate_mre() 와
     HTMLSiteAdapter/fetch_block/build_mre_xml 을 둘 다 공개하는 것과 동일한 패턴.
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
