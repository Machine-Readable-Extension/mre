"""
포맷 무관 node 정규화 — html/hwpx/docx 어댑터가 공통으로 쓰는 유일한 조각.

data_utils/mre_generator.py(v1)의 동명 함수를 이 라이브러리 배포 경계 안으로 옮겨왔다.
"""

from __future__ import annotations


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
