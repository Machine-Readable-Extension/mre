from __future__ import annotations

"""
Progressive 2단계 공개 중 1단계(metadata-only) 뷰를 만드는 변환.

core/ablations.py 의 ablation 실험용 10개 모드(desc_only/keys_only/no_metadata/...) 중
progressive 루프가 실제로 쓰는 건 <tree> 를 통째로 지우는 이 변환 하나뿐이라 이것만
옮겨왔다 — 나머지는 논문 ablation 실험 전용이라 이 서브패키지 범위 밖. 2단계(문서 지목 후
전체 공개)는 별도 변환이 필요 없다 — mre 의 <mre> 스키마는 애초에 <section> 중첩이 없는
flat tree라 원본 mre_xml 을 그대로 보여주면 된다.
"""

import re

_TREE_RE = re.compile(r"\s*<tree\b[^>]*>.*?</tree>", re.DOTALL)


def metadata_view(mre_xml: str) -> str:
    """<tree>(문단 맵)를 통째로 제거해 <metadata>(title+summary)만 남긴다.

    2단계 공개의 1단계 — 모든 후보 문서를 이 뷰로 먼저 저렴하게 보여주고, 에이전트가
    특정 문서를 지목(expand_document)하면 그 문서만 원본 mre_xml(전체 <tree> 포함)을
    그대로 보여준다."""
    return _TREE_RE.sub("", mre_xml).strip()
