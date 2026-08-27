from __future__ import annotations

"""
Legacy HWP 5.0(OLE2/Compound File Binary) 문서에서 문단 텍스트를 추출.

HWP 5.0 은 OPC(zip) 가 아니라 OLE2 compound file 컨테이너다 — mre.opc_adapter 의
hwpx/docx 와는 컨테이너 포맷 자체가 달라 별도 모듈로 둔다. 문서 본문은
BodyText/Section0, Section1, ... 스트림에 있고, FileHeader 스트림의 속성 플래그
bit0 이 켜져 있으면 각 섹션 스트림은 raw deflate(zlib, wbits=-15)로 압축돼 있다.
압축 해제된 바이트열은 (tag_id, level, size) 레코드의 연속이고, HWPTAG_PARA_TEXT(0x43)
레코드가 문단 하나의 텍스트(UTF-16LE, 인라인 컨트롤 문자 섞임)를 담는다.

참고: https://pgc0419.tistory.com/entry/Python-한글-파일hwp-텍스트txt로-변환
(레코드 헤더 파싱 골격은 이 글과 동일 — tag_id/level/size 비트필드, 0xFFF 확장
크기. 그 글은 인라인 컨트롤 문자의 부가 파라미터 바이트를 스킵하지 않고 UTF-16
전체를 디코딩한 뒤 정규식으로 사후 필터링하는데, 그러면 파라미터 바이트가 우연히
읽을 수 있는 문자로 디코딩됐을 때 걸러지지 않거나, 반대로 whitelist가 한글/영문
알파벳만 허용해 다른 언어 본문을 깨뜨린다. 이 모듈은 대신 컨트롤 문자 뒤에 오는
부가 파라미터를 실제로 건너뛴다.)

⚠️ olefile 은 읽기 전용이라 mre.xml 을 원본 파일에 in-place 로 삽입(embed)할 방법이
없다 — 순수 파이썬으로 OLE2 CFB 에 새 스트림을 쓰는 실용적인 라이브러리가 마땅치
않다. 그래서 이 모듈은 extract/strip 만 제공하고, opc_adapter.OPCAdapter 같은
embed/exists/fetch 는 아직 없다 — generate_mre() 의 fmt=HWP 경로도 여전히
NotImplementedError.

⚠️ 인라인 컨트롤 문자(코드 1~31)의 부가 파라미터 크기는 HWP 5.0 배포용 문서 스펙을
따른다: 대부분의 확장 컨트롤은 문자 자신 + 7 WCHAR(14바이트)의 파라미터로 총
8 WCHAR 를 차지하고, 줄 나눔(10)과 문단 나눔(13)만 예외로 파라미터가 없다. 이 표는
임베디드 개체/하이퍼링크/수식처럼 실제 파일에서 흔한 케이스에 대한 실물 검증이
제한적이다 — 만에 하나 어긋나도 피해가 "이후 텍스트 전체가 깨짐"으로 번지지 않도록
섹션별 파싱을 try/except 로 감싸고, 최종 텍스트에서 남은 C0 컨트롤 문자를 한 번 더
걸러낸다(언어 whitelist 로 걸러내는 방식은 비-한국어 문서를 깨뜨리므로 쓰지 않는다).
"""

import re
import struct
import unicodedata
import zlib
from pathlib import Path

import olefile

from mre.nodes import strip_to_text_nodes

HWPTAG_BEGIN = 0x10
HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51  # 0x43

_SECTION_RE = re.compile(r"^BodyText/Section(\d+)$", re.IGNORECASE)
_RECORD_HEADER_SIZE = 4
_EXT_SIZE_SENTINEL = 0xFFF
# 컨트롤 코드 -> 텍스트로 남길 문자(그 외 컨트롤은 파라미터만 스킵하고 텍스트엔 안 남김).
# 9(TAB)/10(줄 나눔)/13(문단 나눔)만 실제 공백/구분자 의미가 있어 예외로 둔다.
_INLINE_TEXT_CHAR = {9: "\t", 10: "\n", 13: "\n"}
# 컨트롤 코드 -> 부가 파라미터 WCHAR 수. 표에 없는 1~31 코드는 기본값(7)을 쓴다
# (대부분의 확장 컨트롤이 실제로 7이므로). 9/10/13 은 텍스트로 남기더라도 부가
# 파라미터는 그대로 스킵해야 한다(9는 탭 정의 정보 7 WCHAR, 10/13은 없음).
_INLINE_EXTRA_WCHARS = {9: 7, 10: 0, 13: 0}
_DEFAULT_INLINE_EXTRA_WCHARS = 7


def _is_compressed(ole: olefile.OleFileIO) -> bool:
    """FileHeader 스트림의 속성 플래그(오프셋 36, 4바이트 LE) bit0 = 압축 여부."""
    with ole.openstream("FileHeader") as f:
        header = f.read(40)
    if len(header) < 40:
        return False
    flags = struct.unpack("<I", header[36:40])[0]
    return bool(flags & 0x1)


def _section_stream_names(ole: olefile.OleFileIO) -> list[str]:
    """BodyText/SectionN 스트림 경로를 섹션 번호 순으로 정렬해 반환."""
    numbered: list[tuple[int, str]] = []
    for entry in ole.listdir(streams=True, storages=False):
        path = "/".join(entry)
        m = _SECTION_RE.match(path)
        if m:
            numbered.append((int(m.group(1)), path))
    numbered.sort(key=lambda pair: pair[0])
    return [path for _, path in numbered]


def _decode_para_text(payload: bytes) -> str:
    """PARA_TEXT 레코드 payload(UTF-16LE, 인라인 컨트롤 섞임) -> 문단 텍스트."""
    n_units = len(payload) // 2
    out: list[str] = []
    i = 0
    while i < n_units:
        code = struct.unpack_from("<H", payload, i * 2)[0]
        if code >= 32:
            out.append(chr(code))
            i += 1
            continue
        extra = _INLINE_EXTRA_WCHARS.get(code, _DEFAULT_INLINE_EXTRA_WCHARS)
        if code in _INLINE_TEXT_CHAR:
            out.append(_INLINE_TEXT_CHAR[code])
        # 그 외 제어문자는 텍스트에 안 남기고 인라인 파라미터만 스킵.
        i += 1 + extra
    text = "".join(out)
    # 안전망: 위 표가 실제 파일과 어긋나 파라미터 바이트가 문자로 잘못 디코딩됐을
    # 경우를 대비해 남은 C0 제어문자만 한 번 더 제거(개행/탭 제외).
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return text.strip()


def _parse_records(data: bytes) -> list[str]:
    """압축 해제된 섹션 바이트열 -> PARA_TEXT 레코드들의 텍스트 리스트(문서 순서)."""
    texts: list[str] = []
    pos = 0
    n = len(data)
    while pos + _RECORD_HEADER_SIZE <= n:
        header = struct.unpack_from("<I", data, pos)[0]
        pos += _RECORD_HEADER_SIZE
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == _EXT_SIZE_SENTINEL:
            if pos + 4 > n:
                break
            size = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        if pos + size > n:
            break
        payload = data[pos:pos + size]
        pos += size
        if tag_id == HWPTAG_PARA_TEXT:
            text = _decode_para_text(payload)
            if text:
                texts.append(text)
    return texts


def build_structure_tree_hwp(hwp_path: str | Path) -> list[dict]:
    """Extract a paragraph node list, in document order, from a legacy HWP (OLE2) document.

    Like hwpx, there's no heading concept, so only paragraphs come out (see
    mre.opc_adapter.build_structure_tree_hwpx). If a section breaks mid-parse,
    only that section is skipped and the rest continue — the judgment being
    that a partial extraction is better than failing the whole document when
    the control-character parameter size table diverges from a real file's
    edge case (embedded objects, etc.).

    Returns
    -------
    nodes : [{"type": "paragraph", "id": "pN", "text": "..."}, ...]
    """
    hwp_path = Path(hwp_path)
    nodes: list[dict] = []
    p_counter = 0

    with olefile.OleFileIO(str(hwp_path)) as ole:
        compressed = _is_compressed(ole)
        for name in _section_stream_names(ole):
            with ole.openstream(name) as f:
                raw = f.read()
            if compressed:
                try:
                    raw = zlib.decompressobj(-15).decompress(raw)
                except zlib.error:
                    continue
            try:
                para_texts = _parse_records(raw)
            except (struct.error, IndexError):
                continue
            for text in para_texts:
                if not text:
                    continue
                p_counter += 1
                nodes.append({"type": "paragraph", "id": f"p{p_counter}", "text": text})

    return nodes


def parse_hwp(path: str | Path) -> list[dict]:
    """Parse path and return the node list cleaned up for the LLM (the counterpart to opc_adapter.parse_opc)."""
    return strip_to_text_nodes(build_structure_tree_hwp(path))
