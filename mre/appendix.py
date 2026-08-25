from __future__ import annotations

"""
Wikipedia appendix section 필터 + 짧은 section 통합 — HTML 문서를 LLM에 보내기 전 정리한다.

data_utils/mre_generator3.py의 동명 섹션을 이 라이브러리 배포 경계 안으로 옮겨왔다.

**주의(single truth)**: 여기서 soup 를 변형(부록 제거/짧은 section 병합)한 순서를 기준으로
생성 시점에 문단 id(pN)가 부여된다. 최종 문서에는 원본(미변형) HTML 이 그대로 embed 되므로,
나중에 pid 로 fetch 할 때도 반드시 이 두 함수를 같은 순서로 다시 적용한 뒤 문단을 walk 해야
생성 시점의 pid 순번과 fetch 시점의 walk 순번이 일치한다 (core/pipeline.py의 _fetch_blocks_v3
가 실제로 이렇게 한다). 이 규칙이 어긋나면 pid 가 엉뚱한 블록을 가리키게 된다.
"""

import re

from bs4 import BeautifulSoup, PageElement, Tag

_APPENDIX_HEADING_IDS: set[str] = {
    "External_links",
    "References",
    "Reference",
    "See_also",
    "Notes",
    "Further_reading",
    "Bibliography",
}
_APPENDIX_HEADING_TEXT: set[str] = {
    "External links",
    "References",
    "Reference",
    "See also",
    "Notes",
    "Further reading",
    "Bibliography",
}
_HEADING_TAG_RE = re.compile(r"^h[1-6]$")


def _heading_level_from_div(div: Tag) -> int | None:
    """`<div class="mw-heading mw-heading{k}">` 에서 k 를 뽑는다. 안에 h1-h6 도 검사."""
    for cls in div.get("class") or []:
        if cls.startswith("mw-heading") and cls != "mw-heading":
            try:
                return int(cls[len("mw-heading"):])
            except ValueError:
                continue
    h = div.find(_HEADING_TAG_RE)
    if h:
        try:
            return int(h.name[1])
        except (ValueError, IndexError):
            pass
    return None


def _is_appendix_heading_div(div: Tag) -> tuple[bool, str, int]:
    """div 가 부록 heading 컨테이너 (`<div class="mw-heading...">` 안에 부록 h#) 인지 판정.
    반환: (부록 여부, 텍스트, 레벨).
    """
    raw_cls: str | list[str] = div.get("class") or []
    cls: list[str] = [raw_cls] if isinstance(raw_cls, str) else list(raw_cls)
    if not any(c.startswith("mw-heading") for c in cls):
        return False, "", 0
    h = div.find(_HEADING_TAG_RE)
    if not h:
        return False, "", 0
    hid = str(h.get("id") or "").strip()
    htext = h.get_text(strip=True)
    if hid in _APPENDIX_HEADING_IDS or htext in _APPENDIX_HEADING_TEXT:
        level = _heading_level_from_div(div) or 2
        return True, (htext or hid.replace("_", " ")), level
    return False, "", 0


def _strip_appendix_sections(soup: BeautifulSoup) -> list[tuple[int, str]]:
    """부록 (`External links` / `References` / `See also` 등) 을 in-place 로 제거한다.
    반환값은 발견된 부록 heading 리스트 (문서 순서대로) — [(level, text), ...]. 현재는 이
    라이브러리도 원본 data_utils/mre_generator3.py 도 이 반환값을 실제로 소비하지 않는다
    (양쪽 다 in-place 제거 효과만 쓰고 호출부에서 버림) — 과거 "부록을 빈 <section> marker로
    남긴다" 구상의 흔적이지만 지금은 구현이 없다. 시그니처만 남아 있으므로 향후 필요해지면
    호출부에서 받아쓰면 된다.

    두 가지 HTML 형태 지원:
      1. Parsoid: 최상위 섹션이 `<section aria-labelledby="…">` 로 감싸짐 → section decompose
      2. non-Parsoid (2wiki/older dump): `<section>` 태그 없고 `<div class="mw-heading">` 이
         sibling 으로만 존재. body container 를 walk 하며 mw-heading div (h# id 가 부록 집합에
         포함) 부터 다음 mw-heading div 직전까지의 노드들을 통째로 제거. asbox/navbox/stub
         div 도 이 sibling 스캔에서 자연스럽게 함께 사라진다.

    heading level 은 `mw-heading{k}` 클래스 또는 h# 태그명에서 뽑는다.
    """
    appendix: list[tuple[int, str]] = []

    # 1) Parsoid: <section aria-labelledby="External_links"> 통째 삭제
    for sec in list(soup.find_all("section")):
        labelled_by = str(sec.get("aria-labelledby") or "").strip()
        first_heading = sec.find(_HEADING_TAG_RE)
        htext = first_heading.get_text(strip=True) if first_heading else ""
        is_appendix = (
            labelled_by in _APPENDIX_HEADING_IDS
            or htext in _APPENDIX_HEADING_TEXT
        )
        if not is_appendix:
            continue
        try:
            level = int(first_heading.name[1]) if first_heading else 2
        except (ValueError, IndexError):
            level = 2
        text = htext or labelled_by.replace("_", " ") or "Appendix"
        appendix.append((level, text))
        sec.decompose()

    # 2) non-Parsoid: body container 안에서 mw-heading div sibling 스캔
    container = (
        soup.find(id="bodyContent")
        or soup.find("div", class_="mw-parser-output")
        or soup.find("body")
        or soup
    )
    if container is None:
        return appendix

    # mw-heading div 를 문서 순서대로 수집 (build_structure_tree 와 동일 walk 규약).
    # container 자식 재귀 없이 top-level heading 만 봐도 대부분 케이스는 커버되지만,
    # 안전하게 descendants 도 훑어 heading 위치를 잡고 sibling-체인 으로 삭제한다.
    for div in list(container.find_all("div", class_=lambda c: c and "mw-heading" in c)):
        is_app, text, level = _is_appendix_heading_div(div)
        if not is_app:
            continue
        appendix.append((level, text))
        # 해당 div 부터 다음 mw-heading div 직전까지 sibling 을 모두 제거
        cur: PageElement | None = div
        while cur is not None:
            nxt = cur.next_sibling
            # 다음 mw-heading div 만나면 stop (그 위치가 다음 섹션 시작)
            if isinstance(nxt, Tag) and nxt.name == "div":
                raw_nxt_cls: str | list[str] = nxt.get("class") or []
                nxt_cls: list[str] = [raw_nxt_cls] if isinstance(raw_nxt_cls, str) else list(raw_nxt_cls)
                if any(c.startswith("mw-heading") for c in nxt_cls):
                    cur.decompose()
                    break
            cur.decompose()
            cur = nxt

    return appendix


# 짧은 섹션 통합 임계값 (문자 수) — 이 미만이면 섹션의 모든 <p>/<ul>/<ol> 를 하나의 <p> 로 합쳐
# 한 개의 요약 노드만 생성한다. Wikipedia 스텁/짧은 subsection 이 문단마다 desc/keys 생성되어
# 노이즈가 되는 것을 방지. 500 은 문단 3-4개 정도의 규모 — 이 이하면 개별 문단으로 나눠도
# heading/keywords 가 서로 거의 같아 정보량이 없음.
_SHORT_SECTION_MERGE_THRESHOLD_CHARS = 500


def _consolidate_short_sections(
    soup: BeautifulSoup,
    threshold_chars: int = _SHORT_SECTION_MERGE_THRESHOLD_CHARS,
) -> int:
    """짧은 <section> 내부 direct <p>/<ul>/<ol> 을 하나의 <p> 로 합친다 (in-place).

    각 <section> 의 direct 자식 중 <p>/<ul>/<ol> (references 제외) 의 텍스트 총량이
    threshold_chars 미만이면, 그 요소들을 모두 지우고 텍스트를 합친 새 <p> 하나로 교체한다.
    nested <section> 안 콘텐츠는 각자의 <section> 스코프에서 별도 판정 (재귀 아님, 상위
    section 이 안쪽 sub-section 텍스트를 상속하지 않는다). heading (<div class="mw-heading">)
    은 건드리지 않으므로 section 제목은 그대로 남는다.

    반환값: 통합된 section 수.
    """
    n_merged = 0
    for sec in soup.find_all("section"):
        content_els: list[Tag] = []
        for child in sec.children:
            if not isinstance(child, Tag):
                continue
            if child.name not in ("p", "ul", "ol"):
                continue
            if child.name == "ol" and "references" in (child.get("class") or []):
                continue
            content_els.append(child)
        if len(content_els) < 2:
            continue   # 이미 0개거나 1개면 합칠 게 없음
        texts: list[str] = []
        for el in content_els:
            t = el.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
        if not texts:
            continue
        combined = " ".join(texts)
        if len(combined) >= threshold_chars:
            continue
        new_p = soup.new_tag("p")
        new_p.string = combined
        content_els[0].replace_with(new_p)
        for el in content_els[1:]:
            el.decompose()
        n_merged += 1
    return n_merged
