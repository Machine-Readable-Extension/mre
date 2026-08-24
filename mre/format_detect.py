from __future__ import annotations

"""
문서 포맷 자동 감지 — 매직 바이트 기반.

지원 포맷: html, pdf, hwp, hwpx, docx.
확장자는 보지 않는다 — bytes/스트림 입력엔 확장자가 없거나 신뢰할 수 없기 때문.
zip 계열(hwpx/docx)은 공통 PK 매직만으로는 구분이 안 되므로, zip을 열어
내부 엔트리로 한 번 더 구분한다.
"""

import io
import zipfile
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Union

SourceLike = Union[str, Path, bytes, bytearray, BinaryIO]


class DocFormat(str, Enum):
    HTML = "html"
    PDF = "pdf"
    HWP = "hwp"
    HWPX = "hwpx"
    DOCX = "docx"


class FormatDetectionError(ValueError):
    """source가 지원 포맷(html/pdf/hwp/hwpx/docx) 중 어느 것으로도 식별되지 않을 때."""


_PDF_MAGIC = b"%PDF-"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy HWP (Compound File Binary)
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")  # local / empty / spanned

_PEEK_SIZE = 8  # 가장 긴 매직(OLE2, 8바이트)을 덮는 최소 헤더 길이
_HTML_SCAN_SIZE = 4096  # 바이너리 매직에 안 걸린 나머지 중 HTML 여부를 찾는 스캔 범위
_HTML_MARKERS = (b"<!doctype html", b"<html")


def _read_header(source: SourceLike, n: int) -> bytes:
    """source 앞 n바이트를 반환. 파일 객체는 읽은 뒤 원래 위치로 되돌려 재사용 가능하게 둔다."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return f.read(n)
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:n])
    if hasattr(source, "read"):
        pos = source.tell()
        try:
            return source.read(n)
        finally:
            source.seek(pos)
    raise TypeError(f"지원하지 않는 source 타입: {type(source)!r}")


def _open_as_zip(source: SourceLike) -> zipfile.ZipFile:
    if isinstance(source, (str, Path)):
        return zipfile.ZipFile(source)
    if isinstance(source, (bytes, bytearray)):
        return zipfile.ZipFile(io.BytesIO(source))
    if hasattr(source, "read"):
        pos = source.tell()
        try:
            return zipfile.ZipFile(source)
        finally:
            source.seek(pos)
    raise TypeError(f"지원하지 않는 source 타입: {type(source)!r}")


def _detect_zip_subtype(source: SourceLike) -> DocFormat:
    """PK 매직을 통과한 zip 컨테이너 내부를 열어 hwpx/docx를 구분한다.

    - docx (OOXML): word/document.xml 포함.
    - hwpx: mimetype 엔트리 내용이 application/hwp+zip 이거나 Contents/content.hpf,
      Contents/header.xml 중 하나를 포함.
    """
    try:
        with _open_as_zip(source) as zf:
            names = set(zf.namelist())
            if "word/document.xml" in names:
                return DocFormat.DOCX
            if "Contents/content.hpf" in names or "Contents/header.xml" in names:
                return DocFormat.HWPX
            if "mimetype" in names:
                try:
                    mimetype = zf.read("mimetype").decode("ascii", errors="ignore").strip()
                except (KeyError, UnicodeDecodeError):
                    mimetype = ""
                if mimetype == "application/hwp+zip":
                    return DocFormat.HWPX
    except zipfile.BadZipFile as e:
        raise FormatDetectionError(f"zip 매직은 있으나 유효한 zip이 아님: {e}") from e

    raise FormatDetectionError(
        "zip 컨테이너이지만 hwpx/docx 시그니처(word/document.xml, "
        "Contents/content.hpf, Contents/header.xml, mimetype=application/hwp+zip) "
        "중 어느 것도 찾지 못함."
    )


def detect_format(source: SourceLike) -> DocFormat:
    """source(파일 경로 / bytes / seekable 파일 객체)의 문서 포맷을 매직 바이트로 감지한다.

    Parameters
    ----------
    source : str | Path | bytes | bytearray | BinaryIO
        감지 대상. 파일 객체는 seek/tell을 지원해야 하며, 호출 후 원래 위치로 복원된다.

    Raises
    ------
    FormatDetectionError
        5개 지원 포맷 중 어느 시그니처와도 매칭되지 않을 때.
    """
    header = _read_header(source, _PEEK_SIZE)

    if header.startswith(_PDF_MAGIC):
        return DocFormat.PDF
    if header.startswith(_OLE2_MAGIC):
        return DocFormat.HWP
    if any(header.startswith(magic) for magic in _ZIP_MAGICS):
        return _detect_zip_subtype(source)

    html_head = _read_header(source, _HTML_SCAN_SIZE).lower()
    if any(marker in html_head for marker in _HTML_MARKERS):
        return DocFormat.HTML

    raise FormatDetectionError(
        "지원 포맷(html/pdf/hwp/hwpx/docx) 중 어느 시그니처도 매칭되지 않음."
    )
