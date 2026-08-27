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
    """The bundle of HTML parsing/embedding logic for a single site.

    preprocess : (optional) cleans the BeautifulSoup document in-place before
                 extract() runs (e.g. removing appendix sections, merging short
                 section paragraphs). The return value (the list of removed
                 appendix headings) is unused by this library (paragraph-
                 granularity only, no <section> nodes) — the signature is kept
                 for a future section-granularity port. None (default) means
                 extract() runs on the soup with no preprocessing.
                 **Caution**: transforming the document here means paragraph
                 ids (pN) are assigned based on the transformed order — this
                 id must later share the exact same preprocessing rules with
                 whatever fetch-side parser re-walks the original (untransformed)
                 document (e.g. core/pipeline.py's _fetch_blocks_v3 re-applies
                 _strip_appendix_sections + _consolidate_short_sections
                 identically to generation time). If the rules diverge, a pid
                 ends up pointing at the wrong block.
    extract : BeautifulSoup-parsed document -> list of heading/paragraph nodes
              ({"type": "heading", "level", "text"} | {"type": "paragraph", "id", "text"})
    assign_ids : (optional) rewrites the ids of extract()'s result (paragraph
                 nodes only) in-place, based on title. None (default) keeps the
                 ids extract() assigned (usually "p1", "p2", ...). Matching
                 mre_generator3.py, the Wikipedia adapter uses this hook to
                 prepend the title's first letter (mitigating cross-document
                 id collisions).
    strip   : extract()'s result -> node list cleaned up for sending to the LLM
    embed   : (original html, assembled mre xml) -> html with mre inserted.
              Always inserts into the "original" html string, never the soup
              mutated by preprocess — appendix sections etc. must remain intact
              in the actual rendered document.
    fetch   : (optional) (html with mre embedded, node id) -> that paragraph's
              full (untruncated) text. The callable a RAG agent uses to actually
              fetch paragraph content — unlike the short preview text extract()
              produces at generation time for showing the LLM, this must return
              the raw text with no length limit. If id is "full", returns the
              whole document's text (all paragraphs concatenated) — for
              workflows that decide whether the whole document is needed after
              seeing just one paragraph. Must re-apply the same preprocessing
              used at generation time and walk in the same order, or the id
              mapping breaks — even as a separate implementation from extract(),
              it must share the exact same rules (see the preprocess note
              above). None (default) means this adapter doesn't support fetch
              (generation-only adapter).
    domains : the list of domains this adapter handles. Subdomains match too —
              domains=("wikipedia.org",) matches en.wikipedia.org,
              ko.wikipedia.org, etc.
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
    """Compute an adapter fingerprint by hashing the source code of the four
    functions extract/preprocess/assign_ids/fetch.

    These four functions (together or separately) determine "which paragraph
    does this id point to" — extract/preprocess/assign_ids assign ids at
    generation time, and fetch locates the paragraph again by that id. If any
    one of the four changes, the fingerprint automatically changes too — so an
    adapter author who forgets to bump a version by hand doesn't slip through
    (this scheme was chosen specifically to avoid the weakness of manual semver
    bumping). generate_mre() stamps this value into the document at generation
    time as <mre generator-fingerprint="...">, and fetch_block() recomputes and
    compares it at fetch time — a mismatch means the adapter's logic changed in
    between.

    A known limitation: even a purely cosmetic change (variable names,
    comments, formatting) changes the value. That false positive is accepted
    as a much safer failure mode than the false negative of "an id silently
    points to the wrong paragraph."

    Functions whose source can't be read (compiled extensions, C
    implementations, etc.) fall back to repr() — in that case the fingerprint
    may miss real code changes, but at least it won't crash.
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
    """Raised when no HTMLSiteAdapter is registered for the URL's domain and no fallback was given."""


# name -> adapter (domains 는 adapter.domains 에 있음)
_REGISTRY: dict[str, HTMLSiteAdapter] = {}


def register_site(adapter: HTMLSiteAdapter) -> None:
    """Register the adapter under its adapter.domains list. Re-registering with
    the same adapter.name overwrites the previous entry — this lets a plugin
    deliberately replace a builtin adapter (auto-discovery runs plugins after
    _register_builtin_sites(), so a plugin using the same name wins over the
    builtin one)."""
    if not adapter.domains:
        raise ValueError(f"adapter.domains 가 비어있음: {adapter.name!r}")
    if adapter.name in _REGISTRY:
        log.info("사이트 어댑터 %r 재등록(덮어씀)", adapter.name)
    _REGISTRY[adapter.name] = adapter


def _domain_matches(netloc: str, domain: str) -> bool:
    netloc = netloc.lower().split(":")[0]  # 포트 제거
    return netloc == domain or netloc.endswith("." + domain)


def detect_site(url: str) -> str | None:
    """Find the registered site name matching url's netloc. Returns None if no match."""
    netloc = urlparse(url).netloc
    if not netloc:
        return None
    for name, adapter in _REGISTRY.items():
        if any(_domain_matches(netloc, d.lower()) for d in adapter.domains):
            return name
    return None


def get_site_adapter(url: str, *, fallback: HTMLSiteAdapter | None = None) -> HTMLSiteAdapter:
    """Return the HTMLSiteAdapter matching url.

    If no registered site matches, uses fallback when one is given; otherwise
    raises UnknownSiteError (since there's no generic HTML adapter yet, this
    chooses to fail explicitly rather than silently parsing with the wrong
    structure).
    """
    name = detect_site(url)
    if name is not None:
        return _REGISTRY[name]
    if fallback is not None:
        return fallback
    raise UnknownSiteError(f"등록된 사이트 어댑터 없음 (도메인 미매칭): {url!r}")


def registered_sites() -> dict[str, tuple[str, ...]]:
    """The current {site name: domains} listing — builtins plus everything auto-discovered from plugins."""
    return {name: adapter.domains for name, adapter in _REGISTRY.items()}


def parse_html(
    url: str, html: str, title: str, *, fallback: HTMLSiteAdapter | None = None
) -> list[dict]:
    """Detect the site from url, parse html with the matching adapter, and
    return the node list cleaned up for the LLM (the strip() result).

    If adapter.preprocess exists it's applied before extract(); if
    adapter.assign_ids exists it's applied right after extract() (title-based
    id rewriting) — this keeps the paragraph ids this function returns
    identical to what the real generation path (generate_mre) produces."""
    adapter = get_site_adapter(url, fallback=fallback)
    soup = BeautifulSoup(html, "lxml")
    if adapter.preprocess is not None:
        adapter.preprocess(soup)
    nodes = adapter.extract(soup)
    if adapter.assign_ids is not None:
        adapter.assign_ids(nodes, title)
    return adapter.strip(nodes)


class FetchNotSupportedError(NotImplementedError):
    """Raised when the matched HTMLSiteAdapter doesn't implement fetch (a generation-only adapter)."""


class GeneratorFingerprintMismatch(RuntimeError):
    """Raised under strict=True when the generator-fingerprint stamped in the
    document differs from the currently installed adapter's fingerprint —
    meaning the adapter's parsing logic changed since this document was
    generated, and the id-to-paragraph mapping may have drifted."""


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
    """Detect the site from url and fetch node_id's full paragraph text via the matching adapter's fetch().

    Lets a RAG agent that consumes an MRE produced by generate_mre() fetch a
    paragraph the same way for any site through this one function — the agent
    code never needs to know "is this document Wikipedia?". Raises
    FetchNotSupportedError if the matched adapter doesn't implement fetch.

    Before fetching, compares the generator-fingerprint stamped in the
    document (if any) against the currently installed adapter's fingerprint
    (see compute_adapter_fingerprint) — a mismatch logs a warning that the
    adapter changed since this document was generated and the id mapping may
    have drifted. Under strict=True, raises GeneratorFingerprintMismatch
    instead of warning. If the document has no fingerprint at all (e.g.
    generated before this feature existed), the comparison is skipped."""
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


_MRE_SCRIPT_TAG_RE = re.compile(
    r'\s*<script\s+type="application/mre\+xml"\s*>.*?</script>\s*',
    re.IGNORECASE | re.DOTALL,
)


def _wiki_inject_mre_into_html(html: str, mre_xml: str) -> str:
    """
    Inject the MRE block as a <script type="application/mre+xml"> tag
    inside <head>. Falls back to prepending if no <head> is found.

    Strips any existing MRE script tag(s) first, so calling this again on
    already-embedded html replaces the block instead of leaving the old one
    in place -- extract_mre_xml() finds the first match in document order,
    so without this the OLDEST embed would keep winning on re-embed, the
    opposite of insert_mre_into_zip()/embed_mre_pdf(), which both replace.
    """
    html = _MRE_SCRIPT_TAG_RE.sub("", html)
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
    """Scan installed packages for the 'mre.site_adapters' entry point and auto-register them.
    mre/__init__.py calls this automatically once when mre is imported — you usually don't need
    to call it directly, only when you've installed a new adapter package mid-process and want
    to re-discover it."""
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
