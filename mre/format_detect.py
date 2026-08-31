from __future__ import annotations

"""
Document format auto-detection, based on magic bytes.

Supported formats: html, pdf, hwp, hwpx, docx.
File extensions are ignored: bytes/stream inputs often have none, or an
unreliable one. The zip-based formats (hwpx/docx) share the same PK magic,
so a zip that matches gets opened and disambiguated by its internal entries.
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
    """Raised when source cannot be identified as any supported format (html/pdf/hwp/hwpx/docx)."""


_PDF_MAGIC = b"%PDF-"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy HWP (Compound File Binary)
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")  # local / empty / spanned

_PEEK_SIZE = 8  # covers the longest magic (OLE2, 8 bytes)
_HTML_SCAN_SIZE = 4096  # scan range for HTML markers once binary magics miss
_HTML_MARKERS = (b"<!doctype html", b"<html")


def _read_header(source: SourceLike, n: int) -> bytes:
    """Return the first n bytes of source. File objects are seeked back to their original position afterward, so they remain reusable."""
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
    raise TypeError(f"Unsupported source type: {type(source)!r}")


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
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _detect_zip_subtype(source: SourceLike) -> DocFormat:
    """Open a zip container that already passed the PK magic check and tell hwpx apart from docx.

    - docx (OOXML): contains word/document.xml.
    - hwpx: the mimetype entry reads application/hwp+zip, or the zip
      contains Contents/content.hpf or Contents/header.xml.
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
        raise FormatDetectionError(f"Zip magic matched but this isn't a valid zip: {e}") from e

    raise FormatDetectionError(
        "This is a zip container, but none of the hwpx/docx signatures "
        "(word/document.xml, Contents/content.hpf, Contents/header.xml, "
        "mimetype=application/hwp+zip) were found."
    )


def detect_format(source: SourceLike) -> DocFormat:
    """Detect source's document format from its magic bytes (file path / bytes / seekable file object).

    Parameters
    ----------
    source : str | Path | bytes | bytearray | BinaryIO
        The target to detect. File objects must support seek/tell, and are
        restored to their original position after the call.

    Raises
    ------
    FormatDetectionError
        When none of the 5 supported formats' signatures match.
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
        "None of the supported format signatures (html/pdf/hwp/hwpx/docx) matched."
    )
