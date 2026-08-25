"""Minimal OLE2/CFB (Compound File Binary) writer, for test fixtures only.

olefile (and every other pure-Python CFB library available on PyPI) is
read-only, so there's no library to build a real ``.hwp`` fixture with. This
hand-rolls just enough of [MS-CFB] to produce a file olefile can read back:
one FAT sector, one directory sector, and a handful of stream sectors --
deliberately no MiniFAT/ministream subsystem, which this builder doesn't
implement. olefile silently overrides a header ``mini_stream_cutoff_size``
of 0 back to the mandatory default (4096), so a stream can't opt out of
MiniFAT that way; instead every stream here is zero-padded up to that cutoff
so its declared size is never small enough to be MiniFAT-eligible.

Not a general-purpose CFB writer: fixed at 512-byte sectors (v3), a single
FAT sector (so at most ~128 total sectors, far more than any test fixture
here needs), and a directory tree that's a simple right-linked chain (a
degenerate, but perfectly legal, red-black tree) rather than a balanced one.
"""

from __future__ import annotations

import struct

_SECTOR_SIZE = 512
_MINI_STREAM_CUTOFF = 4096  # [MS-CFB] mandatory value; see module docstring
_FREESECT = 0xFFFFFFFF
_ENDOFCHAIN = 0xFFFFFFFE
_FATSECT = 0xFFFFFFFD
_NOSTREAM = 0xFFFFFFFF
_STGTY_STORAGE = 0x1
_STGTY_STREAM = 0x2
_STGTY_ROOT = 0x5
_STRUCT_DIRENTRY = "<64sHBBIII16sIQQIII"
_STRUCT_HEADER = "<8s16sHHHHHHLLLLLLLLLL"
_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _pad_sector(data: bytes) -> bytes:
    pad = (-len(data)) % _SECTOR_SIZE
    return data + b"\x00" * pad


def _avoid_minifat(data: bytes) -> bytes:
    """Zero-pad up to the MiniFAT cutoff so this stream's declared size is
    never MiniFAT-eligible (see module docstring). Trailing zero bytes are
    harmless to callers: hwp_adapter._is_compressed only reads the first 40
    bytes, and hwp_adapter._parse_records treats an all-zero record header
    (tag_id=0, size=0) as a 4-byte no-op it just steps over."""
    if len(data) >= _MINI_STREAM_CUTOFF:
        return data
    return data + b"\x00" * (_MINI_STREAM_CUTOFF - len(data))


def _name_field(name: str) -> tuple[bytes, int]:
    raw = name.encode("utf-16-le") + b"\x00\x00"
    if len(raw) > 64:
        raise ValueError(f"name too long for a CFB directory entry: {name!r}")
    return raw + b"\x00" * (64 - len(raw)), len(raw)


def _dir_entry(
    name: str, entry_type: int, sid_left: int, sid_right: int, sid_child: int,
    isect_start: int, size: int,
) -> bytes:
    name_raw, namelength = _name_field(name)
    return struct.pack(
        _STRUCT_DIRENTRY,
        name_raw, namelength, entry_type, 1,  # color: irrelevant to a reader, always "black"
        sid_left, sid_right, sid_child,
        b"\x00" * 16,  # clsid
        0,  # dwUserFlags
        0, 0,  # createTime, modifyTime
        isect_start, size, 0,  # isectStart, sizeLow, sizeHigh
    )


def build_ole2_hwp(sections: list[bytes], file_header: bytes) -> bytes:
    """Build a minimal valid OLE2 CFB file with FileHeader + BodyText/SectionN streams.

    Parameters
    ----------
    sections : one entry per BodyText/SectionN stream, in order, already
        compressed if the caller wants ``FileHeader``'s flag bit to say so
        (this function just writes the bytes it's given -- it doesn't
        compress anything itself).
    file_header : raw bytes for the FileHeader stream (must be >=40 bytes
        for mre.hwp_adapter._is_compressed to read the flags word).
    """
    if len(sections) > 2:
        # directory-sector layout below assumes <=4 entries (fits in one
        # 512-byte directory sector); extend _build_dir_sector if more are
        # ever needed.
        raise ValueError("this test builder supports at most 2 sections")

    file_header = _avoid_minifat(file_header)
    sections = [_avoid_minifat(s) for s in sections]

    sectors: list[bytes] = [b""]  # index 0 reserved for the FAT sector itself

    def alloc_chain(data: bytes) -> int:
        if not data:
            data = b""
        chunks = [data[i:i + _SECTOR_SIZE] for i in range(0, len(data), _SECTOR_SIZE)] or [b""]
        start = len(sectors)
        for chunk in chunks:
            sectors.append(_pad_sector(chunk))
        return start

    filehdr_start = alloc_chain(file_header)
    section_starts = [alloc_chain(s) for s in sections]

    # sid layout: 0=Root Entry, 1=BodyText storage, 2=FileHeader stream,
    # 3..=SectionN streams (right-linked sibling chain under BodyText).
    section_sids = list(range(3, 3 + len(sections)))
    dir_entries = [
        _dir_entry("Root Entry", _STGTY_ROOT, _NOSTREAM, _NOSTREAM, 1, _ENDOFCHAIN, 0),
        _dir_entry("BodyText", _STGTY_STORAGE, _NOSTREAM, 2, section_sids[0], 0, 0),
        _dir_entry("FileHeader", _STGTY_STREAM, _NOSTREAM, _NOSTREAM, _NOSTREAM,
                   filehdr_start, len(file_header)),
    ]
    for i, (sid, start, data) in enumerate(zip(section_sids, section_starts, sections)):
        right = section_sids[i + 1] if i + 1 < len(section_sids) else _NOSTREAM
        dir_entries.append(_dir_entry(
            f"Section{i}", _STGTY_STREAM, _NOSTREAM, right, _NOSTREAM, start, len(data),
        ))
    dir_bytes = b"".join(dir_entries)
    dir_start = alloc_chain(dir_bytes)

    # ---- FAT: chain each stream's sectors, mark sector 0 as the FAT sector itself ----
    total_sectors = len(sectors)
    fat = [_FREESECT] * max(total_sectors, _SECTOR_SIZE // 4)
    fat[0] = _FATSECT

    def chain(start: int, data: bytes) -> None:
        n = max(1, -(-len(data) // _SECTOR_SIZE))  # ceil-div, at least 1 sector
        for i in range(n - 1):
            fat[start + i] = start + i + 1
        fat[start + n - 1] = _ENDOFCHAIN

    chain(dir_start, dir_bytes)
    chain(filehdr_start, file_header)
    for start, data in zip(section_starts, sections):
        chain(start, data)

    sectors[0] = _pad_sector(struct.pack(f"<{len(fat)}I", *fat))

    # ---- header ----
    difat = [0] + [_FREESECT] * 108  # our one FAT sector is sector 0
    header = struct.pack(
        _STRUCT_HEADER,
        _MAGIC, b"\x00" * 16,
        0x003E, 3, 0xFFFE, 9, 6, 0,  # minor, dll(v3), byte_order, sector_shift(512), mini_sector_shift, reserved1
        0,  # reserved2
        0,  # num_dir_sectors (must be 0 for v3)
        1,  # num_fat_sectors
        dir_start,
        0,  # transaction_signature_number
        _MINI_STREAM_CUTOFF,  # mandatory value -- see module docstring / _avoid_minifat
        _ENDOFCHAIN, 0,  # first_mini_fat_sector, num_mini_fat_sectors
        _ENDOFCHAIN, 0,  # first_difat_sector, num_difat_sectors
    ) + struct.pack("<109I", *difat)
    assert len(header) == 512, len(header)

    return header + b"".join(sectors)
