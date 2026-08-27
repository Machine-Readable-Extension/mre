from __future__ import annotations

"""
Embed된 HTML에서 <mre> 헤더 원문을 통째로 다시 읽어오는 유틸리티 — fetch_block()(단락
하나만 읽어오는)의 대칭점. 사이트 어댑터 dispatch가 필요 없다 — 어떤 사이트가 만들었든
<script type="application/mre+xml"> 태그 위치·형식은 동일하므로, html_site_adapter.py의
사이트별 레지스트리와는 별도로 이 모듈에 둔다.

core/mre.py의 MREParser.extract_mre를 이 라이브러리 배포 경계 안으로 옮겨왔다.
"""

import re

from bs4 import BeautifulSoup

# 레거시 <resources> 섹션 제거 — mre 의 자체 생성 스키마(xml_builder.build_mre_xml)는
# <resources> 를 만들지 않지만, 다른 생성기로 만들어진 문서와도 호환되도록 유지한다
# (html_site_adapter._wiki_fetch 의 동일 원칙 참조).
_RESOURCES_RE = re.compile(r"\s*<resources>.*?</resources>", re.DOTALL)


def extract_mre_xml(html: str) -> str | None:
    """Return the raw <mre>...</mre> content from embedded html. None if there's no MRE.

    Used anywhere an agent needs to read the full header (title/summary/tree)
    before navigating a document — e.g. before mre.agent's progressive loop
    builds a metadata-only view of candidate documents, or before showing the
    full tree of a document picked out by expand_document."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"type": "application/mre+xml"})
    if script is None:
        return None
    raw = script.string
    if raw is None:
        return None
    mre_text = _RESOURCES_RE.sub("", raw.strip())
    return mre_text.strip()
