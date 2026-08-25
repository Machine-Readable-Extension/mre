from __future__ import annotations

"""
OPC(Open Packaging Conventions) zip 컨테이너 기반 문서 어댑터 — hwpx, docx.

hwpx와 docx는 내부가 둘 다 zip이고, MRE는 그 zip root에 mre.xml 파일 하나를
추가하는 방식으로 임베딩된다 (archive 기반 포맷 공통 규칙 — html의 <head><script>
삽입에 대응). 이 embed/exists 연산은 두 포맷에서 완전히 동일한 zip 조작이라
insert_mre_into_zip / _mre_xml_exists_in_zip 하나씩만 두고 공용으로 쓴다
(data_utils/mre_generator.py(v1)에서 이 라이브러리 배포 경계 안으로 이식).

파싱(extract)만 포맷별로 다르다:
  - hwpx: Contents/section*.xml 안의 <hp:p>/<hp:t> — 헤딩 개념이 없어 문단만 나온다
    (data_utils/mre_generator.py(v1)의 build_structure_tree_hwpx를 이식).
  - docx: word/document.xml 안의 <w:p> — pStyle이 HeadingN/Title이면 heading,
    아니면 문단으로 분류 (신규 구현, 이 레포에 기존 docx 처리 코드가 없었음).
    표(<w:tbl>) 내부 문단은 이번 구현 범위에서 제외 — hwpx의 "표는 바깥 문단에
    흡수" 같은 규칙이 docx엔 없어 임의로 흡수시키면 표 순서가 뒤틀리므로,
    본문 흐름(body 직속 <w:p>)만 다룬다.
"""

import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mre.format_detect import DocFormat
from mre.html_site_adapter import FetchNotSupportedError
from mre.nodes import strip_to_text_nodes


@dataclass(frozen=True)
class OPCAdapter:
    """OPC zip 문서 하나에 대한 파싱/임베딩/fetch 로직 묶음.

    extract : 문서 경로 -> heading/paragraph 노드 리스트
              ({"type": "heading", "level", "text"} | {"type": "paragraph", "id", "text"})
    strip   : extract() 결과 -> LLM 전송용으로 정리된 노드 리스트
    embed   : (문서 경로, 조립된 mre xml) -> None (zip root에 mre.xml을 in-place 삽입/교체)
    exists  : 문서 경로 -> 이미 mre.xml이 삽입되어 있는지 여부
    fetch   : (문서 경로, node id) -> 그 단락의 전체 텍스트. id="full"이면 문서 전체 텍스트를
              이어붙여 반환. 못 찾으면 빈 문자열(예외 아님) — html_site_adapter.fetch_block()과
              동일 계약. None이면 이 어댑터는 fetch를 지원하지 않음(fetch_opc()가
              FetchNotSupportedError).
    """
    name: str
    extract: Callable[[Path], list[dict]]
    strip: Callable[[list[dict]], list[dict]]
    embed: Callable[[Path, str], None]
    exists: Callable[[Path], bool]
    fetch: Callable[[Path, str], str] | None = None


# ─────────────────────────────────────────────
# docx 파싱 (신규)
# ─────────────────────────────────────────────

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_VAL = f"{_W_NS}val"
_HEADING_STYLE_RE = re.compile(r"heading\s*(\d)", re.IGNORECASE)


def _heading_level_from_style(style_val: str | None) -> int | None:
    """word paragraph의 pStyle 값에서 heading level을 뽑는다.

    python-docx/LibreOffice/MS Word가 만드는 기본 스타일 ID는 로캘과 무관하게
    "Heading1".."Heading9" (또는 "Title")로 고정되는 것이 일반적 관례이므로
    이를 신뢰한다 — 커스텀 템플릿이 스타일 ID 자체를 바꾼 경우는 감지되지 않는다.
    """
    if not style_val:
        return None
    m = _HEADING_STYLE_RE.search(style_val)
    if m:
        return int(m.group(1))
    if style_val.strip().lower() == "title":
        return 1
    return None


def build_structure_tree_docx(docx_path: Path) -> list[dict]:
    """DOCX(word/document.xml)에서 heading/paragraph 노드 리스트를 문서 순서대로 추출.

    body 직속 <w:p>만 다룬다 (표 안 <w:p>는 제외 — 모듈 docstring 참조).

    Returns
    -------
    nodes : [{"type": "heading", "level": int, "text": str}
             | {"type": "paragraph", "id": "pN", "text": str}, ...]
    """
    nodes: list[dict] = []
    p_counter = 0

    with zipfile.ZipFile(docx_path, "r") as zf:
        with zf.open("word/document.xml") as xml_file:
            root = ET.parse(xml_file).getroot()

    body = root.find(f"{_W_NS}body")
    if body is None:
        return nodes

    for el in body:
        if el.tag != f"{_W_NS}p":
            continue  # w:tbl 등 문단이 아닌 요소는 건너뜀

        style_val = None
        p_pr = el.find(f"{_W_NS}pPr")
        if p_pr is not None:
            p_style = p_pr.find(f"{_W_NS}pStyle")
            if p_style is not None:
                style_val = p_style.get(_W_VAL)

        text = "".join(t.text or "" for t in el.iter(f"{_W_NS}t")).strip()
        if not text:
            continue

        level = _heading_level_from_style(style_val)
        if level is not None:
            nodes.append({"type": "heading", "level": level, "text": text})
        else:
            p_counter += 1
            nodes.append({"type": "paragraph", "id": f"p{p_counter}", "text": text})

    return nodes


# ─────────────────────────────────────────────
# hwpx 파싱/임베딩 (data_utils/mre_generator.py v1에서 이식)
# ─────────────────────────────────────────────
# 각 단락은 OWPML 의 <hp:p> 요소이고, 그 안의 <hp:t> 들이 실제 텍스트.
# ElementTree 는 태그를 ``{namespace-uri}localname`` 으로 노출하므로 endswith 로 식별.

_HWPX_SECTION_RE = re.compile(r"^Contents/section\d+\.xml$")
_HWPX_SECTION_IDX_RE = re.compile(r"\d+")
_MRE_ENTRY_NAME = "mre.xml"
_HWPX_MIN_PARA_CHARS = 50   # 이 길이 이하의 단락은 뒤따라오는 단락에 통합 (제목/날짜 라벨 등 파편 흡수)


def _coalesce_short_paragraphs(nodes: list[dict]) -> list[dict]:
    """50자 이하 단락은 다음 단락에 통합. 끝까지 남으면 직전 단락에 합쳐 흡수.

    HWPX 보도자료의 ``보도자료`` / ``보도시점`` / 날짜 등 짧은 라벨 단락이 의미상 다음
    본문 단락의 일부이므로 LLM 에 보낼 때도 묶어서 보내는 게 자연스럽다.
    """
    if not nodes:
        return nodes

    merged: list[dict] = []
    buf = ""
    for n in nodes:
        combined = (buf + "\n" + n["text"]) if buf else n["text"]
        if len(combined) <= _HWPX_MIN_PARA_CHARS:
            buf = combined
            continue
        merged.append({
            "type": "paragraph",
            "id": "",  # 아래에서 renumber
            "text": combined,
        })
        buf = ""
    if buf:
        if merged:
            merged[-1]["text"] = merged[-1]["text"] + "\n" + buf
        else:
            merged.append({"type": "paragraph", "id": "", "text": buf})

    for i, n in enumerate(merged, 1):
        n["id"] = f"p{i}"
    return merged


def build_structure_tree_hwpx(hwpx_path: Path) -> list[dict]:
    """HWPX OPC ZIP 에서 paragraph 노드 리스트를 추출.

    최외곽 <hp:p> 하나를 하나의 단락으로 본다. 표/텍스트박스 등 내부에 nested <hp:p> 가
    있어도 그 텍스트는 모두 외곽 <hp:p> 의 단락 텍스트로 흡수되며, 내부 <hp:p> 는 별도
    paragraph 로 추가되지 않는다 (중복 카운팅 방지).

    Returns
    -------
    nodes : [{"type": "paragraph", "id": "pN", "text": "..."}, ...]
    """
    nodes: list[dict] = []
    p_counter = 0

    with zipfile.ZipFile(hwpx_path, "r") as zf:
        numbered: list[tuple[int, str]] = []
        for name in zf.namelist():
            if not _HWPX_SECTION_RE.match(name):
                continue
            idx_m = _HWPX_SECTION_IDX_RE.search(name)
            if idx_m:
                numbered.append((int(idx_m.group()), name))
        numbered.sort(key=lambda pair: pair[0])
        section_files = [name for _, name in numbered]
        for sec in section_files:
            with zf.open(sec) as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # ElementTree 는 부모 포인터를 노출하지 않으므로 직접 parent map 작성.
                parent_map = {child: parent for parent in tree.iter() for child in parent}

                def _has_p_ancestor(elem) -> bool:
                    cur = parent_map.get(elem)
                    while cur is not None:
                        if cur.tag.endswith("}p"):
                            return True
                        cur = parent_map.get(cur)
                    return False

                for elem in root.iter():
                    if not elem.tag.endswith("}p"):
                        continue
                    if _has_p_ancestor(elem):
                        # nested <hp:p> — 최외곽 단락에 흡수됨
                        continue
                    text_parts: list[str] = []
                    for sub in elem.iter():
                        if sub.tag.endswith("}t") and sub.text:
                            text_parts.append(sub.text)
                    text = "".join(text_parts).strip()
                    if not text:
                        continue
                    p_counter += 1
                    nodes.append({
                        "type": "paragraph",
                        "id": f"p{p_counter}",
                        "text": text,
                    })
    return _coalesce_short_paragraphs(nodes)


def _mre_xml_exists_in_zip(opc_path: Path) -> bool:
    try:
        with zipfile.ZipFile(opc_path, "r") as zf:
            return _MRE_ENTRY_NAME in zf.namelist()
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def extract_mre_xml_opc(opc_path: str | Path) -> str | None:
    """OPC zip(hwpx/docx) root의 mre.xml 엔트리 원문을 반환한다. 없으면 None.

    mre.reader.extract_mre_xml(html) 의 OPC 대응 — html 은 <script> 태그를 파싱해야
    하지만 OPC 는 embed 가 애초에 mre.xml 을 별도 zip 엔트리로 넣으므로(insert_mre_into_zip)
    포맷(hwpx/docx) 구분 없이 그대로 읽으면 된다."""
    try:
        with zipfile.ZipFile(opc_path, "r") as zf:
            return zf.read(_MRE_ENTRY_NAME).decode("utf-8")
    except (zipfile.BadZipFile, FileNotFoundError, KeyError):
        return None


def insert_mre_into_zip(opc_path: Path, mre_xml: str) -> None:
    """OPC ZIP root 에 mre.xml 을 삽입한다 (이미 있으면 덮어쓴다).

    zipfile 은 in-place 삭제/수정을 지원하지 않으므로 새 zip 으로 복사한 뒤 ``os.replace``
    로 원자적 교체. 같은 디렉토리에 임시 파일을 만들어 cross-device rename 회피.
    """
    if not opc_path.exists():
        raise FileNotFoundError(opc_path)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".mre_tmp_", suffix=".zip", dir=str(opc_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(opc_path, "r") as zin, \
             zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == _MRE_ENTRY_NAME:
                    continue  # 새 mre.xml 로 대체
                zout.writestr(item, zin.read(item.filename))
            zout.writestr(_MRE_ENTRY_NAME, mre_xml.encode("utf-8"))
        os.replace(tmp_path, opc_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ─────────────────────────────────────────────
# fetch (hwpx/docx 공통 — id 조회 로직은 동일, extract 함수만 다름)
# ─────────────────────────────────────────────
# html_site_adapter._wiki_fetch와 달리 별도 재파싱 로직이 필요 없다: extract()가 만드는
# paragraph 노드의 text는 (Wikipedia의 _wiki_extract_node_text와 달리) LLM 프롬프트용으로
# 잘리지 않은 전체 텍스트이므로, extract()를 그대로 다시 불러 인덱싱하면 곧 fetch가 된다 —
# 진짜 "single truth"(생성 시점과 fetch 시점이 완전히 같은 함수를 씀).

_PID_RE = re.compile(r"^[A-Za-z]*(\d+)$")


def _fetch_from_paragraphs(nodes: list[dict], node_id: str) -> str:
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


def _hwpx_fetch(path: Path, node_id: str) -> str:
    return _fetch_from_paragraphs(build_structure_tree_hwpx(path), node_id)


def _docx_fetch(path: Path, node_id: str) -> str:
    return _fetch_from_paragraphs(build_structure_tree_docx(path), node_id)


# ─────────────────────────────────────────────
# 어댑터 등록 (hwpx/docx 공통 — embed/exists는 두 포맷에서 동일한 zip 조작)
# ─────────────────────────────────────────────

_REGISTRY: dict[DocFormat, OPCAdapter] = {}


def get_opc_adapter(fmt: DocFormat) -> OPCAdapter:
    try:
        return _REGISTRY[fmt]
    except KeyError:
        raise ValueError(f"OPC 어댑터 미등록 포맷: {fmt!r} (등록됨: {list(_REGISTRY)})") from None


def parse_opc(path: str | Path, fmt: DocFormat) -> list[dict]:
    """path를 fmt 어댑터로 파싱하고 LLM 전송용으로 정리된 노드 리스트를 반환."""
    adapter = get_opc_adapter(fmt)
    path = Path(path)
    return adapter.strip(adapter.extract(path))


def embed_mre_opc(path: str | Path, mre_xml: str, fmt: DocFormat) -> None:
    """path(hwpx/docx)의 zip root에 mre.xml을 in-place 삽입/교체."""
    get_opc_adapter(fmt).embed(Path(path), mre_xml)


def fetch_opc(path: str | Path, node_id: str, fmt: DocFormat) -> str:
    """path(hwpx/docx)에서 node_id 단락의 전체 텍스트를 가져온다. id="full"이면 문서
    전체 텍스트. 어댑터가 fetch를 지원하지 않으면 FetchNotSupportedError."""
    adapter = get_opc_adapter(fmt)
    if adapter.fetch is None:
        raise FetchNotSupportedError(
            f"어댑터 {adapter.name!r} 는 fetch 를 구현하지 않았습니다."
        )
    return adapter.fetch(Path(path), node_id)


def _register_builtin_adapters() -> None:
    _REGISTRY[DocFormat.HWPX] = OPCAdapter(
        name="hwpx",
        extract=build_structure_tree_hwpx,
        strip=strip_to_text_nodes,
        embed=insert_mre_into_zip,
        exists=_mre_xml_exists_in_zip,
        fetch=_hwpx_fetch,
    )
    _REGISTRY[DocFormat.DOCX] = OPCAdapter(
        name="docx",
        extract=build_structure_tree_docx,
        strip=strip_to_text_nodes,
        embed=insert_mre_into_zip,   # mre.xml as a plain zip entry — 포맷 무관 동일 연산
        exists=_mre_xml_exists_in_zip,
        fetch=_docx_fetch,
    )


_register_builtin_adapters()
