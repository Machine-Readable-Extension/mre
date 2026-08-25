from __future__ import annotations

"""
HTML MRE 생성 — 사이트(도메인)별 파싱 로직 dispatch.

웹페이지는 사이트마다 body 구조가 다르다 (Wikipedia의 mw-heading div 구조가
대표적). 도메인별로 파싱 어댑터를 등록해두고, 문서의 출처 URL의 netloc으로
맞는 어댑터를 골라 쓴다. Wikipedia 어댑터의 실체(_wiki_build_structure_tree 등,
아래 참조)는 원래 data_utils/mre_generator.py(v1)에 있던 Wikipedia 전용 파싱
로직을 이 라이브러리 배포 경계 안으로 이식한 것이다.

사이트 소유자가 자기 어댑터를 별도 패키지로 배포할 수 있도록, entry points
("mre.site_adapters" 그룹)로 설치된 어댑터를 임포트 시점에 자동 발견한다 —
아래 "플러그인 자동 발견" 절 참조.
"""

import hashlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Callable
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from mre.appendix import _strip_appendix_sections, _consolidate_short_sections
from mre.nodes import strip_to_text_nodes

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HTMLSiteAdapter:
    """사이트 하나에 대한 HTML 파싱/임베딩 로직 묶음.

    preprocess : (선택) BeautifulSoup 문서를 extract() 전에 in-place로 정리한다
                 (예: 부록 section 제거, 짧은 section 문단 통합). 반환값(제거된 부록 heading
                 목록)은 이 라이브러리(paragraph-granularity 전용, <section> 미생성)에서는
                 쓰지 않는다 — section-granularity 를 나중에 포팅할 때를 위해 시그니처만
                 남겨둠. None(기본)이면 전처리 없이 extract()를 soup에 그대로 적용한다.
                 **주의**: 여기서 문서를 변형하면, 그 변형된 순서를 기준으로 문단 id(pN)가
                 부여된다 — 이 id는 나중에 원본(미변형) 문서를 다시 걷는 fetch 쪽 파서와
                 반드시 같은 전처리 규칙을 공유해야 한다 (예: core/pipeline.py의
                 _fetch_blocks_v3가 생성 시점과 동일하게 _strip_appendix_sections +
                 _consolidate_short_sections를 다시 적용). 규칙이 어긋나면 pid가 엉뚱한
                 블록을 가리키게 된다.
    extract : BeautifulSoup 파싱된 문서 -> heading/paragraph 노드 리스트
              ({"type": "heading", "level", "text"} | {"type": "paragraph", "id", "text"})
    assign_ids : (선택) extract() 결과(문단 타입만)의 id를 title 기반으로 in-place 재작성.
                 None(기본)이면 extract()가 부여한 id(보통 "p1", "p2", ...)를 그대로 쓴다.
                 mre_generator3.py 와 동일하게, Wikipedia 어댑터는 이 훅으로 제목 첫 글자
                 접두어를 붙인다(cross-document id collision 완화).
    strip   : extract() 결과 -> LLM 전송용으로 정리된 노드 리스트
    embed   : (원본 html, 조립된 mre xml) -> mre가 삽입된 html.
              전처리로 변형한 soup가 아니라 항상 "원본" html 문자열에 삽입한다 — 부록
              section 등은 실제 문서 렌더링에서는 그대로 남아 있어야 하므로.
    fetch   : (선택) (mre가 embed된 html, node id) -> 그 문단의 전체(비절단) 텍스트.
              RAG 에이전트가 실제로 문단 내용을 가져올 때 쓰는 콜러블 — extract()가
              생성-시점에 만드는 "LLM에 보여줄 짧은 미리보기 텍스트"와 달리, 여기서는
              길이 제한 없이 원문 그대로 반환해야 한다. id 는 "full"이면 문서 전체
              텍스트(모든 문단 이어붙임)를 반환한다(하나만 보고 문서 전체가 필요한지
              판단하는 워크플로용). 생성 시점과 동일한 전처리(preprocess)를 다시 적용한
              뒤 같은 순서로 walk 해야 id 매핑이 어긋나지 않는다 — extract()와 별개
              구현이어도 반드시 같은 규칙을 공유해야 한다(파일 상단 preprocess 설명 참조).
              None(기본)이면 이 어댑터로 fetch 는 지원 안 함(생성 전용 어댑터).
    domains : 이 어댑터가 처리하는 도메인 목록. 서브도메인까지 매치된다 —
              domains=("wikipedia.org",)면 en.wikipedia.org, ko.wikipedia.org 등 전부 매치.
    """
    name: str
    extract: Callable[[BeautifulSoup], list[dict]]
    strip: Callable[[list[dict]], list[dict]]
    embed: Callable[[str, str], str]
    domains: tuple[str, ...] = field(default_factory=tuple)
    preprocess: Callable[[BeautifulSoup], list[tuple[int, str]]] | None = None
    assign_ids: Callable[[list[dict], str], None] | None = None
    fetch: Callable[[str, str], str] | None = None


def compute_adapter_fingerprint(adapter: HTMLSiteAdapter) -> str:
    """extract/preprocess/assign_ids/fetch 네 함수의 소스 코드를 해시해 어댑터
    fingerprint 를 계산한다.

    이 네 함수가 (같이든 따로든) "id 가 어느 문단을 가리키는가"를 결정한다 —
    extract/preprocess/assign_ids 는 생성 시점의 id 부여, fetch 는 그 id 로 다시 문단을
    찾아내는 쪽. 넷 중 하나라도 바뀌면 fingerprint 가 자동으로 달라진다 — 어댑터
    작성자가 버전을 손으로 올리는 걸 잊어도 놓치지 않는다(semver 수동 관리의 약점을
    피하려고 이 방식을 선택했다). generate_mre() 가 생성 시점에 이 값을
    <mre generator-fingerprint="..."> 로 문서에 새기고, fetch_block() 이 fetch 시점에
    다시 계산해 비교한다 — 다르면 그 사이 어댑터 로직이 바뀌었다는 뜻이다.

    코드가 아주 사소하게(변수명, 주석, 포맷팅)만 바뀌어도 값이 달라지는 건 알고 있는
    한계다 — 그런 false positive 는 "id 가 조용히 엉뚱한 문단을 가리키는" false
    negative 보다 훨씬 안전한 실패 방향이라 받아들인다.

    소스를 읽을 수 없는 함수(컴파일된 확장, C 구현 등)는 repr() 로 폴백한다 — 그런
    경우 fingerprint 가 코드 변경을 못 잡아낼 수 있지만 최소한 크래시는 안 한다.
    """
    parts: list[str] = []
    for fn in (adapter.extract, adapter.preprocess, adapter.assign_ids, adapter.fetch):
        if fn is None:
            parts.append("None")
            continue
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):
            parts.append(repr(fn))
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


class UnknownSiteError(LookupError):
    """URL의 도메인에 등록된 HTMLSiteAdapter가 없고 fallback도 지정되지 않았을 때."""


# name -> adapter (domains 는 adapter.domains 에 있음)
_REGISTRY: dict[str, HTMLSiteAdapter] = {}


def register_site(adapter: HTMLSiteAdapter) -> None:
    """adapter.domains 목록으로 어댑터를 등록한다. 같은 adapter.name 으로 다시
    등록하면 덮어쓴다 — 플러그인이 내장 어댑터를 의도적으로 대체할 수 있게 하기
    위함(자동 발견 순서는 _register_builtin_sites() 이후 플러그인이므로, 플러그인이
    같은 name 을 쓰면 내장 어댑터를 이긴다)."""
    if not adapter.domains:
        raise ValueError(f"adapter.domains 가 비어있음: {adapter.name!r}")
    if adapter.name in _REGISTRY:
        log.info("사이트 어댑터 %r 재등록(덮어씀)", adapter.name)
    _REGISTRY[adapter.name] = adapter


def _domain_matches(netloc: str, domain: str) -> bool:
    netloc = netloc.lower().split(":")[0]  # 포트 제거
    return netloc == domain or netloc.endswith("." + domain)


def detect_site(url: str) -> str | None:
    """url의 netloc으로 등록된 사이트 name을 찾는다. 매칭 실패 시 None."""
    netloc = urlparse(url).netloc
    if not netloc:
        return None
    for name, adapter in _REGISTRY.items():
        if any(_domain_matches(netloc, d.lower()) for d in adapter.domains):
            return name
    return None


def get_site_adapter(url: str, *, fallback: HTMLSiteAdapter | None = None) -> HTMLSiteAdapter:
    """url에 맞는 HTMLSiteAdapter를 반환.

    매칭되는 등록 사이트가 없으면 fallback이 주어진 경우 그것을 쓰고,
    없으면 UnknownSiteError를 낸다 (아직 범용 HTML 어댑터가 없으므로
    묵묵히 잘못된 구조로 파싱하는 것보다 명시적으로 실패시키는 쪽을 택함).
    """
    name = detect_site(url)
    if name is not None:
        return _REGISTRY[name]
    if fallback is not None:
        return fallback
    raise UnknownSiteError(f"등록된 사이트 어댑터 없음 (도메인 미매칭): {url!r}")


def registered_sites() -> dict[str, tuple[str, ...]]:
    """현재 등록된 {사이트 name: domains} 목록 — 내장 + 플러그인 자동 발견 결과 전부."""
    return {name: adapter.domains for name, adapter in _REGISTRY.items()}


def parse_html(
    url: str, html: str, title: str, *, fallback: HTMLSiteAdapter | None = None
) -> list[dict]:
    """url로 사이트를 감지해 맞는 어댑터로 html을 파싱하고,
    LLM 전송용으로 정리된 노드 리스트(strip 결과)를 반환한다.

    adapter.preprocess가 있으면 extract() 전에, adapter.assign_ids가 있으면 extract() 직후
    (title 기반 id 재작성)에 적용한다 — 그래야 이 함수가 돌려주는 문단 id 가 실제 생성
    경로(generate_mre)와 동일하게 나온다."""
    adapter = get_site_adapter(url, fallback=fallback)
    soup = BeautifulSoup(html, "lxml")
    if adapter.preprocess is not None:
        adapter.preprocess(soup)
    nodes = adapter.extract(soup)
    if adapter.assign_ids is not None:
        adapter.assign_ids(nodes, title)
    return adapter.strip(nodes)


class FetchNotSupportedError(NotImplementedError):
    """매칭된 HTMLSiteAdapter 가 fetch 를 구현하지 않았을 때(생성 전용 어댑터)."""


class GeneratorFingerprintMismatch(RuntimeError):
    """strict=True 에서, 문서에 새겨진 generator-fingerprint 가 지금 설치된 어댑터의
    fingerprint 와 다를 때 — 이 문서가 생성된 이후 어댑터의 파싱 로직이 바뀌어
    id-to-paragraph 매핑이 어긋났을 수 있다는 뜻."""


_MRE_ROOT_TAG_RE = re.compile(r"<mre\b([^>]*)>")
_XML_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def _extract_mre_root_attrs(html: str) -> dict[str, str]:
    """html에 embed된 <script type="application/mre+xml"> 안 <mre ...> 루트 태그의
    속성을 뽑는다. 문서 전체를 XML 로 파싱하지 않고 <mre ...> 여는 태그만 정규식으로
    본다 — MREParser.extract_mre 가 이미 하는 것과 동일한 가벼운 접근."""
    m = _MRE_ROOT_TAG_RE.search(html)
    if not m:
        return {}
    return dict(_XML_ATTR_RE.findall(m.group(1)))


def _check_generator_fingerprint(adapter: HTMLSiteAdapter, html: str, *, strict: bool) -> None:
    attrs = _extract_mre_root_attrs(html)
    embedded_name = attrs.get("generator")
    embedded_fp = attrs.get("generator-fingerprint")
    if embedded_name is None or embedded_fp is None:
        return  # 이 기능 도입 전에 생성된 문서 등 — 비교 대상 없음, 통과
    if embedded_name != adapter.name:
        return  # 이 어댑터가 만든 문서가 아님 — fingerprint 비교 대상 아님
    current_fp = compute_adapter_fingerprint(adapter)
    if current_fp == embedded_fp:
        return
    msg = (
        f"어댑터 {adapter.name!r} 의 파싱 로직이 이 문서가 생성된 시점 이후 바뀐 것 "
        f"같습니다 (생성 시 fingerprint={embedded_fp!r}, 지금 설치된 어댑터="
        f"{current_fp!r}). id-to-paragraph 매핑이 어긋났을 수 있습니다."
    )
    if strict:
        raise GeneratorFingerprintMismatch(msg)
    log.warning(msg)


def fetch_block(
    url: str, html: str, node_id: str, *, fallback: HTMLSiteAdapter | None = None, strict: bool = False
) -> str:
    """url로 사이트를 감지해 맞는 어댑터의 fetch() 로 node_id 문단의 전체 텍스트를 가져온다.

    generate_mre() 가 만든 MRE 를 실제로 소비하는 RAG 에이전트가 이 함수 하나로 어떤
    사이트의 문서든 동일하게 문단을 가져올 수 있다 — 에이전트 코드가 "이 문서가
    Wikipedia 인지" 알 필요가 없다. 매칭된 어댑터가 fetch 를 구현하지 않았으면
    FetchNotSupportedError.

    fetch 전에 문서에 새겨진 generator-fingerprint(있다면)를 지금 설치된 어댑터의
    fingerprint 와 비교한다(compute_adapter_fingerprint 참조) — 다르면 어댑터가
    이 문서 생성 이후 바뀐 것이므로 id 매핑이 어긋났을 수 있다는 경고를 로그로 남긴다.
    strict=True 면 경고 대신 GeneratorFingerprintMismatch 를 던진다. 문서에
    fingerprint 자체가 없으면(이 기능 도입 전 생성 등) 비교를 건너뛴다."""
    adapter = get_site_adapter(url, fallback=fallback)
    if adapter.fetch is None:
        raise FetchNotSupportedError(
            f"사이트 어댑터 {adapter.name!r} 는 fetch 를 지원하지 않음(생성 전용)"
        )
    _check_generator_fingerprint(adapter, html, strict=strict)
    return adapter.fetch(html, node_id)


# ─────────────────────────────────────────────
# Wikipedia 어댑터 구현
# ─────────────────────────────────────────────
# data_utils/mre_generator.py(v1)의 동명 함수를 이 라이브러리 배포 경계 안으로 옮겨왔다.

_WIKI_NODE_TEXT_LIMIT = 400
# 부록성 섹션은 build_structure_tree 자체에서도 한 번 더 걸러진다 — appendix.py의
# _strip_appendix_sections(preprocess 단계, extract 전에 soup에서 통째로 제거)와 이중 방어.
# preprocess가 커버하지 못하는 호출 경로(예: build_structure_tree를 preprocess 없이 직접
# 호출하는 경우)에서도 최소한의 필터가 남도록 원본 동작을 그대로 보존한다.
_WIKI_APPENDIX_HEADINGS = {"References", "See also", "External links"}


def _wiki_extract_node_text(element: Tag, limit: int = _WIKI_NODE_TEXT_LIMIT) -> str:
    """Extract visible text from a tag, stripped and truncated."""
    text = element.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _wiki_build_structure_tree(soup: BeautifulSoup) -> list[dict]:
    """
    Wikipedia HTML에서 mw-heading 섹션 제목과 <p> 단락을 문서 순서대로 추출합니다.

    반환: nodes — 각 항목은 두 가지 타입 중 하나
      - {"type": "heading", "level": int, "text": str}
      - {"type": "paragraph", "id": str, "text": str}
    """
    container = (
        soup.find(id="bodyContent")
        or soup.find("div", class_="mw-parser-output")
        or soup.find("body")
        or soup
    )

    nodes: list[dict] = []
    p_counter = [0]

    def process_element(el: Tag) -> None:
        if not isinstance(el, Tag):
            return

        # div.mw-heading.mw-heading{k} → heading 노드
        if el.name == "div":
            raw_classes: str | list[str] = el.get("class") or []
            classes: list[str] = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
            for cls in classes:
                m = re.match(r"mw-heading(\d)", cls)
                if m:
                    level = int(m.group(1))
                    h_tag = el.find(f"h{level}")
                    if h_tag:
                        htext = h_tag.get_text(strip=True)
                        if htext in _WIKI_APPENDIX_HEADINGS:
                            return  # 부록성 섹션은 구조 트리에 포함하지 않음
                        nodes.append({
                            "type": "heading",
                            "level": level,
                            "text": htext,
                        })
                    return  # mw-heading 내부는 재귀하지 않음

        # <p>, <ul>, <ol> → paragraph 노드
        # MediaWiki는 *(불릿) / #(넘버) 위키 문법을 <ul><li> / <ol><li>로 렌더링하므로
        # 본문 단락처럼 취급하고, <li> 텍스트를 합쳐 본문으로 사용한다.
        # 단, <ol class="references">(부록성 출처 목록)는 제외한다.
        #
        # pid 카운팅 정책: 텍스트가 *비어있지 않을 때만* counter 증가 + node 추가.
        # 이렇게 해야 MRE의 <node id="pN"> 가 p1, p2, p3, ... 연속한다 (LLM이 보이지 않는
        # gap 안에 paragraph가 있을 거라고 환각하는 것을 방지). MREParser.fetch_blocks
        # 의 indexing(core/mre.py)도 동일 규칙(비어있지 않은 paragraph만 카운팅).
        if el.name in ("p", "ul", "ol"):
            if el.name == "ol" and "references" in (el.get("class") or []):
                return
            text = _wiki_extract_node_text(el)
            if text:
                p_counter[0] += 1
                pid = el.get("id") or f"p{p_counter[0]}"
                nodes.append({
                    "type": "paragraph",
                    "id": pid,
                    "text": text,
                })
            return  # 내부는 재귀하지 않음

        # 그 외 요소는 자식을 재귀 탐색
        for child in el.children:
            if isinstance(child, Tag):
                process_element(child)

    for child in container.children:
        if isinstance(child, Tag):
            process_element(child)

    return nodes


def _wiki_inject_mre_into_html(html: str, mre_xml: str) -> str:
    """
    Inject the MRE block as a <script type="application/mre+xml"> tag
    inside <head>. Falls back to prepending if no <head> is found.
    """
    mre_tag = f'\n<script type="application/mre+xml">\n{mre_xml}\n</script>\n'

    head_end = re.search(r"</head\s*>", html, re.IGNORECASE)
    if head_end:
        pos = head_end.start()
        return html[:pos] + mre_tag + html[pos:]

    html_start = re.search(r"<html[^>]*>", html, re.IGNORECASE)
    if html_start:
        pos = html_start.end()
        return html[:pos] + "\n<head>" + mre_tag + "</head>" + html[pos:]

    return mre_tag + html


def _wiki_title_letter_prefix(title: str) -> str:
    """문서 제목의 첫 알파벳(대문자)을 노드 id 접두어로 사용.

    cross-document id collision 완화용 — 후보 문서 여러 개의 헤더가 한 프롬프트에 같이
    노출될 때 순수 숫자만 다른 "p3" 같은 id는 문서 간에 겹쳐서, 에이전트가 잘못된 문서의
    pid를 요청하는 환각(hallucination)을 유발한다. 문서마다 고유한 접두어를 붙이면 최소한
    "그 pid가 어느 문서 것인지"는 시각적으로 구분된다 (data_utils/mre_generator3.py의
    동명 함수와 동일 스킴 — 정확한 재현을 위해 규칙을 그대로 옮김). 알파벳이 전혀 없는
    제목은 'X'로 폴백.
    """
    m = re.search(r"[A-Za-z]", title)
    return m.group(0).upper() if m else "X"


def _wiki_apply_title_letter_ids(nodes: list[dict], title: str) -> None:
    """nodes(문단 타입만)의 id를 "p{N}" → "{letter}{N}"으로 in-place 재작성.

    _wiki_build_structure_tree()가 부여한 원래 등장 순서(N)는 그대로 유지하고 접두어만
    바꾼다. "p{N}" 폴백 패턴에만 매치하므로, <p id="...">처럼 HTML이 실제로 갖고 있던
    id(드묾)는 건드리지 않는다. LLM 호출(strip 이후) 전에 호출해야 LLM도 새 id를 보고
    그대로 echo해서, 이후 XML 조립까지 새 id 체계로 일관되게 흐른다 — HTMLSiteAdapter의
    assign_ids 훅으로 extract() 직후, strip() 이전에 적용된다.
    """
    letter = _wiki_title_letter_prefix(title)
    for node in nodes:
        if node.get("type") != "paragraph":
            continue
        m = re.match(r"^p(\d+)$", node.get("id", ""))
        if m:
            node["id"] = f"{letter}{m.group(1)}"


# ─────────────────────────────────────────────
# 내장 사이트 어댑터 등록
# ─────────────────────────────────────────────

def _wiki_preprocess(soup: BeautifulSoup) -> list[tuple[int, str]]:
    # v3 전용 전처리 — References/See also/External links 등 부록 section을 통째로 지우고
    # (그대로 두면 IMDb 링크/스텁 안내문 같은 부록 콘텐츠에도 desc/keys가 생겨버림),
    # 짧은 subsection의 문단들을 하나로 통합한다 (노이즈성 개별 desc/keys 방지).
    # data_utils/mre_generator3.py의 process_db_document_async2와 동일 순서 — extract()와
    # fetch() 양쪽 모두 이 함수를 그대로 다시 적용하는 "single truth"이므로, 여기서 순서나
    # 함수를 바꾸면 생성-fetch 간 id 매핑이 어긋난다.
    appendix_sections = _strip_appendix_sections(soup)
    _consolidate_short_sections(soup)
    return appendix_sections


def _wiki_fetch(html: str, node_id: str) -> str:
    """MRE 가 embed된 Wikipedia 문서 html에서 node_id 문단의 전체(비절단) 텍스트를 가져온다.

    _wiki_build_structure_tree()가 부여하는 순번과 일치시키려면 생성 시점과 동일하게
    _wiki_preprocess(부록 제거 + 짧은 section 통합)를 다시 적용한 뒤 같은 규칙
    (<p>/<ul>/<ol>, ol.references 제외, 매칭 요소 내부로 재귀 안 함, 빈 문단 스킵)으로
    walk 해야 한다 — extract()의 _wiki_extract_node_text()는 LLM 프롬프트용으로 400자에서
    자르지만, 여기서는 에이전트가 실제로 읽을 문단이므로 자르지 않는다.

    node_id 가 "full"이면 문서의 모든 문단 텍스트를 이어붙여 반환한다(문단 하나만 보고는
    부족하고 문서 전체가 필요하다고 판단하는 워크플로용). 그 외에는 접두 알파벳을 무시하고
    끝의 숫자만으로 몇 번째 문단인지 찾는다 — "p1"/"B1"/"P15" 등 어떤 접두어 스킴이든
    (assign_ids 가 뭘 적용했든) 동일하게 동작한다. 찾지 못하면 빈 문자열을 반환한다
    (예외를 던지지 않음 — core/pipeline.py 의 _fetch_blocks_v3 와 동일 계약).

    이 문서에 <script type="application/mre+xml"> 안에 (레거시 스키마의) <resources>
    <target id="..."> 가 있으면 문단 텍스트 뒤에 이어붙인다 — mre 의 자체 생성 스키마
    (build_mre_xml)는 <resources>를 만들지 않지만, 다른 생성기로 만들어진 문서와도
    호환되도록 유지한다.
    """
    soup = BeautifulSoup(html, "lxml")
    _wiki_preprocess(soup)

    container = (
        soup.find(id="bodyContent")
        or soup.find("div", class_="mw-parser-output")
        or soup.find("body")
        or soup
    )

    blocks: list[Tag] = []

    def _walk(node) -> None:
        if not isinstance(node, Tag):
            return
        if node.name in ("p", "ul", "ol"):
            if node.name == "ol" and "references" in (node.get("class") or []):
                return
            text = node.get_text(separator=" ", strip=True)
            if text:
                blocks.append(node)
            return  # 매칭된 요소 내부로는 descend 안 함
        for child in node.children:
            _walk(child)

    _walk(container)

    mre_script = soup.find("script", {"type": "application/mre+xml"})

    def _resources_text(target_id: str | None) -> str:
        if not (mre_script and mre_script.string):
            return ""
        if target_id is None:
            all_targets = re.findall(
                r"<target\s+id=[\"'][^\"']*[\"'][^>]*>(.*?)</target>",
                mre_script.string, re.DOTALL,
            )
            return " ".join(
                " ".join(re.sub(r"<[^>]+>", " ", t).split()) for t in all_targets
            ).strip()
        target_pat = re.compile(
            r"<target\s+id=[\"']" + re.escape(target_id) + r"[\"'][^>]*>(.*?)</target>",
            re.DOTALL,
        )
        tm = target_pat.search(mre_script.string)
        if not tm:
            return ""
        return " ".join(re.sub(r"<[^>]+>", " ", tm.group(1)).split())

    if node_id == "full":
        if not blocks:
            return ""
        block_text = "\n\n".join(b.get_text(separator=" ", strip=True) for b in blocks)
        items_text = _resources_text(None)
        return f"{block_text}\n\n[resources]\n{items_text}" if items_text else block_text

    el = None
    m = re.match(r"^[A-Za-z]*(\d+)$", node_id)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(blocks):
            el = blocks[idx - 1]
    if el is None:
        return ""

    block_text = el.get_text(separator=" ", strip=True)
    items_text = _resources_text(node_id)
    return f"{block_text}\n\n[resources]\n{items_text}" if items_text else block_text


# ─────────────────────────────────────────────
# 내장 사이트 어댑터 등록
# ─────────────────────────────────────────────

def _register_builtin_sites() -> None:
    register_site(
        HTMLSiteAdapter(
            name="wikipedia",
            domains=("wikipedia.org",),
            extract=_wiki_build_structure_tree,
            strip=strip_to_text_nodes,
            embed=_wiki_inject_mre_into_html,
            preprocess=_wiki_preprocess,
            assign_ids=_wiki_apply_title_letter_ids,
            fetch=_wiki_fetch,
        ),
    )


# ─────────────────────────────────────────────
# 플러그인 자동 발견 (entry points)
# ─────────────────────────────────────────────
# 사이트 소유자는 자기 어댑터를 별도 pip 패키지로 배포하고, 그 패키지의
# pyproject.toml 에 다음처럼 선언한다:
#
#   [project.entry-points."mre.site_adapters"]
#   my-site = "my_mre_adapter:ADAPTER"
#
# 값은 HTMLSiteAdapter 인스턴스이거나, 인스턴스를 반환하는 인자 없는 factory
# 여야 한다. mre 를 import 하는 시점에 설치된 모든 그런 패키지를 스캔해 자동
# 등록한다 — 코어 라이브러리 코드를 고치지 않고도 새 사이트가 추가된다.
#
# discover_plugin_adapters() 는 이 모듈이 아니라 mre/__init__.py 맨 끝에서 호출된다
# (여기서 바로 호출하면 안 됨) — 플러그인이 관례적으로 `from mre import HTMLSiteAdapter`
# 로 임포트하는데, 이 모듈은 mre.__init__ 이 자신을 임포트하는 도중에 로드되므로
# 그 시점엔 mre 패키지가 아직 다 초기화되지 않아 순환 임포트가 난다. import 이후 새
# 플러그인을 설치했다면(런타임 재설치) 다시 호출해도 안전 — register_site() 는 멱등적으로
# 덮어쓴다.
_ENTRY_POINT_GROUP = "mre.site_adapters"


def discover_plugin_adapters() -> None:
    """설치된 패키지 중 'mre.site_adapters' entry point 를 스캔해 자동 등록한다.
    mre 를 import 할 때 mre/__init__.py 가 자동으로 한 번 호출한다 — 직접 호출할 필요는
    보통 없고, 프로세스 실행 중에 새 어댑터 패키지를 설치한 뒤 재발견하고 싶을 때만 쓴다."""
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            adapter = obj() if not isinstance(obj, HTMLSiteAdapter) else obj
        except Exception as e:  # noqa: BLE001 — 플러그인 하나의 실패가 나머지 발견을 막으면 안 됨
            log.warning("mre.site_adapters entry point 로드 실패 [%s]: %s", ep.name, e)
            continue
        if not isinstance(adapter, HTMLSiteAdapter):
            log.warning(
                "mre.site_adapters entry point %r 가 HTMLSiteAdapter 를 반환하지 않음: %r",
                ep.name, adapter,
            )
            continue
        register_site(adapter)
        log.info("플러그인 사이트 어댑터 등록: %s (entry point %r)", adapter.name, ep.name)


_register_builtin_sites()
