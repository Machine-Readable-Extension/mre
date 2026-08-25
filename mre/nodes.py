from __future__ import annotations

"""
포맷 무관 node 정규화 — html/hwpx/docx/pdf 어댑터가 공통으로 쓰는 조각들.

data_utils/mre_generator.py(v1)의 동명 함수를 이 라이브러리 배포 경계 안으로 옮겨왔다.
"""

import re

_PID_RE = re.compile(r"^[A-Za-z]*(\d+)$")


def strip_to_text_nodes(nodes: list[dict]) -> list[dict]:
    """
    LLM 전송용: heading과 paragraph 노드를 순서 유지하며 추출합니다.
    - heading: type, level, text
    - paragraph: type, id, text
    """
    result = []
    for node in nodes:
        ntype = node.get("type", "paragraph")
        if ntype == "heading":
            result.append({"type": "heading", "level": node["level"], "text": node["text"]})
        else:
            result.append({"type": "paragraph", "id": node["id"], "text": node["text"]})
    return result


def fetch_paragraph_by_id(nodes: list[dict], node_id: str) -> str:
    """extract()가 만든 노드 리스트에서 node_id 단락의 전체 텍스트를 가져온다.
    id="full"이면 문서 전체 텍스트(문단 구분은 빈 줄). 못 찾으면 빈 문자열(예외 아님) —
    html_site_adapter.fetch_block()과 동일 계약.

    hwpx/docx(mre.opc_adapter)와 pdf(mre.pdf_adapter)가 공유하는 조각 — 두 포맷 모두
    "path 기반 문서 -> extract()를 다시 불러 인덱싱" 방식으로 fetch를 구현하므로
    (생성 시점과 fetch 시점이 완전히 같은 함수를 쓰는 single-truth 원칙, mre.appendix
    모듈 docstring 참조) id 파싱/조회 로직 자체는 포맷과 무관하다."""
    para_nodes = [n for n in nodes if n.get("type") == "paragraph"]
    if node_id == "full":
        return "\n\n".join(n["text"] for n in para_nodes)
    m = _PID_RE.match(node_id)
    if not m:
        return ""
    idx = int(m.group(1))
    if 1 <= idx <= len(para_nodes):
        return para_nodes[idx - 1]["text"]
    return ""
