from __future__ import annotations

"""
Extract paragraph text from a legacy HWP 5.0 (OLE2/Compound File Binary) document.

HWP 5.0 is an OLE2 compound file container, not OPC (zip), so it gets its
own module rather than living alongside mre.opc_adapter's hwpx/docx
handling: the container format itself is different. The document body
lives in the BodyText/Section0, Section1, ... streams; if bit 0 of the
FileHeader stream's attribute flags is set, each section stream is
compressed with raw deflate (zlib, wbits=-15). The decompressed bytes are a
sequence of (tag_id, level, size) records, and an HWPTAG_PARA_TEXT (0x43)
record holds one paragraph's text (UTF-16LE, with inline control characters
mixed in).

Reference: https://pgc0419.tistory.com/entry/Python-%ED%95%9C%EA%B8%80-%ED%8C%8C%EC%9D%BChwp-%ED%85%8D%EC%8A%A4%ED%8A%B8txt%EB%A1%9C-%EB%B3%80%ED%99%98
(a Korean blog post on parsing HWP in Python). The record-header parsing
skeleton here matches that post: the tag_id/level/size bitfields, the
0xFFF extended-size sentinel. That post decodes the whole UTF-16 payload
without skipping an inline control character's extra parameter bytes, then
filters afterward with a regex; that approach either fails to catch
parameter bytes that happen to decode into readable characters, or, if it
whitelists only Hangul and Latin letters, corrupts text in other languages.
This module instead actually skips the extra parameters that follow a
control character.

Caution: olefile is read-only, so there's no way to embed mre.xml into the
original file in place; there's no practical pure-Python library for
writing a new stream into an OLE2 CFB container. This module therefore only
provides extract/strip; there's no embed/exists/fetch like
opc_adapter.OPCAdapter yet, and generate_mre()'s fmt=HWP path still raises
NotImplementedError.

Caution: the extra-parameter size for inline control characters (codes
1-31) follows the HWP 5.0 distribution document spec: most extended
controls take up 8 WCHARs total (the character itself plus a 7-WCHAR/14-byte
parameter), with line break (10) and paragraph break (13) as the only
parameter-less exceptions. This table has only limited real-world
verification against common cases like embedded objects, hyperlinks, or
equations. To keep a mismatch from corrupting all subsequent text, each
section is parsed inside its own try/except, and the final text gets one
more pass to strip any leftover C0 control characters (a language whitelist
isn't used for this, since it would corrupt non-Korean documents).
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
# Control code -> character to keep in the text (any other control code has
# its parameter skipped and nothing added to the text). Only 9 (TAB),
# 10 (line break), and 13 (paragraph break) carry real whitespace/separator
# meaning, so they're the exceptions.
_INLINE_TEXT_CHAR = {9: "\t", 10: "\n", 13: "\n"}
# Control code -> number of extra parameter WCHARs. Codes 1-31 not listed
# here default to 7, since that's the actual value for most extended
# controls. Even when 9/10/13 are kept in the text, their extra parameters
# still need to be skipped (9 has 7 WCHARs of tab-definition info; 10/13
# have none).
_INLINE_EXTRA_WCHARS = {9: 7, 10: 0, 13: 0}
_DEFAULT_INLINE_EXTRA_WCHARS = 7


def _is_compressed(ole: olefile.OleFileIO) -> bool:
    """Bit 0 of the FileHeader stream's attribute flags (offset 36, 4 bytes LE) tells whether the content is compressed."""
    with ole.openstream("FileHeader") as f:
        header = f.read(40)
    if len(header) < 40:
        return False
    flags = struct.unpack("<I", header[36:40])[0]
    return bool(flags & 0x1)


def _section_stream_names(ole: olefile.OleFileIO) -> list[str]:
    """Return BodyText/SectionN stream paths, sorted by section number."""
    numbered: list[tuple[int, str]] = []
    for entry in ole.listdir(streams=True, storages=False):
        path = "/".join(entry)
        m = _SECTION_RE.match(path)
        if m:
            numbered.append((int(m.group(1)), path))
    numbered.sort(key=lambda pair: pair[0])
    return [path for _, path in numbered]


def _decode_para_text(payload: bytes) -> str:
    """Decode a PARA_TEXT record payload (UTF-16LE with inline controls mixed in) into paragraph text."""
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
        # Any other control character is dropped from the text; only its inline parameter is skipped.
        i += 1 + extra
    text = "".join(out)
    # Safety net: in case the table above diverges from a real file and a
    # parameter byte got misdecoded as a character, strip any remaining C0
    # control characters once more (newline/tab excluded).
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return text.strip()


def _parse_records(data: bytes) -> list[str]:
    """Decode a decompressed section's bytes into the text of its PARA_TEXT records, in document order."""
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
