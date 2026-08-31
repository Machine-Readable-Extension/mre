from __future__ import annotations

"""
Document adapters for OPC (Open Packaging Conventions) zip containers: hwpx, docx.

Both hwpx and docx are zip files internally, and MRE gets embedded by
adding a single mre.xml file at the zip root (the archive-format
equivalent of inserting a <head><script> tag in html). This embed/exists
operation is identical zip manipulation for both formats, so
insert_mre_into_zip / _mre_xml_exists_in_zip are each defined once and
shared (ported from data_utils/mre_generator.py (v1) into this library's
distribution boundary).

Only parsing (extract) differs per format:
  - hwpx: <hp:p>/<hp:t> inside Contents/section*.xml. There's no heading
    concept, so only paragraphs come out (ported from
    data_utils/mre_generator.py (v1)'s build_structure_tree_hwpx).
  - docx: <w:p> inside word/document.xml. Classified as a heading if
    pStyle is HeadingN/Title, otherwise a paragraph (a new implementation;
    this repo had no existing docx handling code). Paragraphs inside
    tables (<w:tbl>) are out of scope here: docx has no equivalent of
    hwpx's "absorb into the surrounding paragraph" rule for tables, and
    absorbing them arbitrarily would scramble table order, so only
    body-level <w:p> (the main text flow) is handled.
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
from mre.nodes import fetch_paragraph_by_id, strip_to_text_nodes


@dataclass(frozen=True)
class OPCAdapter:
    """The bundle of parsing/embedding/fetch logic for a single OPC zip document.

    extract : document path -> list of heading/paragraph nodes
              ({"type": "heading", "level", "text"} | {"type": "paragraph", "id", "text"})
    strip   : extract()'s result -> node list cleaned up for sending to the LLM
    embed   : (document path, assembled mre xml) -> None (inserts/replaces
              mre.xml at the zip root, in place)
    exists  : document path -> whether mre.xml is already inserted
    fetch   : (document path, node id) -> that paragraph's full text. If
              id="full", returns the whole document's text concatenated. Returns
              an empty string (not an exception) if not found — same contract
              as html_site_adapter.fetch_block(). None means this adapter
              doesn't support fetch (fetch_opc() raises FetchNotSupportedError).
    """
    name: str
    extract: Callable[[Path], list[dict]]
    strip: Callable[[list[dict]], list[dict]]
    embed: Callable[[Path, str], None]
    exists: Callable[[Path], bool]
    fetch: Callable[[Path, str], str] | None = None


# ─────────────────────────────────────────────
# docx parsing (new)
# ─────────────────────────────────────────────

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_VAL = f"{_W_NS}val"
_HEADING_STYLE_RE = re.compile(r"heading\s*(\d)", re.IGNORECASE)


def _heading_level_from_style(style_val: str | None) -> int | None:
    """Extract a heading level from a word paragraph's pStyle value.

    python-docx/LibreOffice/MS Word conventionally generate default style
    ids as "Heading1".."Heading9" (or "Title") regardless of locale, so this
    trusts that convention. A custom template that renames the style ids
    themselves won't be detected.
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
    """Extract a heading/paragraph node list from DOCX (word/document.xml), in document order.

    Only handles body-level <w:p> (paragraphs inside tables are excluded, see module docstring).

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
            continue  # skip non-paragraph elements such as w:tbl

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
# hwpx parsing/embedding (ported from data_utils/mre_generator.py v1)
# ─────────────────────────────────────────────
# Each paragraph is an OWPML <hp:p> element, and the <hp:t> elements inside
# it hold the actual text. ElementTree exposes tags as
# ``{namespace-uri}localname``, so they're matched with endswith.

_HWPX_SECTION_RE = re.compile(r"^Contents/section\d+\.xml$")
_HWPX_SECTION_IDX_RE = re.compile(r"\d+")
_MRE_ENTRY_NAME = "mre.xml"
_HWPX_MIN_PARA_CHARS = 50   # paragraphs at or below this length get merged into the following one (absorbs fragments like title/date labels)


def _coalesce_short_paragraphs(nodes: list[dict]) -> list[dict]:
    """Merge paragraphs of 50 characters or fewer into the following paragraph; a trailing leftover merges into the previous one instead.

    HWPX press releases often have short label paragraphs (e.g. "press
    release", "release date", or a bare date) that are semantically part of
    the following body paragraph, so it's more natural to send them to the
    LLM merged together.
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
            "id": "",  # renumbered below
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
    """Extract a paragraph node list from an HWPX OPC ZIP.

    Each outermost <hp:p> counts as one paragraph. A nested <hp:p> inside a
    table/text box, etc., has its text absorbed into the outer <hp:p>'s
    paragraph text and is not added as a separate paragraph (avoids double-counting).

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
                # ElementTree doesn't expose parent pointers, so build a parent map manually.
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
                        # nested <hp:p>: absorbed into the outermost paragraph
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
    """Return the raw content of the mre.xml entry at an OPC zip's (hwpx/docx) root. None if absent.

    The OPC counterpart to mre.reader.extract_mre_xml(html) — html requires
    parsing a <script> tag, but OPC's embed already puts mre.xml in as a
    separate zip entry (insert_mre_into_zip), so it can just be read directly
    regardless of format (hwpx/docx)."""
    try:
        with zipfile.ZipFile(opc_path, "r") as zf:
            return zf.read(_MRE_ENTRY_NAME).decode("utf-8")
    except (zipfile.BadZipFile, FileNotFoundError, KeyError):
        return None


def insert_mre_into_zip(opc_path: Path, mre_xml: str) -> None:
    """Insert mre.xml at the OPC ZIP root, overwriting it if already present.

    zipfile doesn't support in-place delete/modify, so this copies to a new
    zip and atomically swaps it in with ``os.replace``. The temp file is
    created in the same directory to avoid a cross-device rename.
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
                    continue  # replaced by the new mre.xml
                zout.writestr(item, zin.read(item.filename))
            zout.writestr(_MRE_ENTRY_NAME, mre_xml.encode("utf-8"))
        os.replace(tmp_path, opc_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ─────────────────────────────────────────────
# fetch (shared by hwpx/docx: the id-lookup logic is identical, only extract differs)
# ─────────────────────────────────────────────
# Unlike html_site_adapter._wiki_fetch, no separate re-parsing logic is
# needed here: extract()'s paragraph node text is already the full,
# untruncated text meant for the LLM prompt (unlike Wikipedia's
# _wiki_extract_node_text), so simply re-running extract() and indexing into
# it amounts to a fetch. This is a genuine single source of truth:
# generation time and fetch time use the exact same function. The id-lookup
# logic itself is shared with mre.pdf_adapter, so it lives in
# mre.nodes.fetch_paragraph_by_id.


def _hwpx_fetch(path: Path, node_id: str) -> str:
    return fetch_paragraph_by_id(build_structure_tree_hwpx(path), node_id)


def _docx_fetch(path: Path, node_id: str) -> str:
    return fetch_paragraph_by_id(build_structure_tree_docx(path), node_id)


# ─────────────────────────────────────────────
# Adapter registration (shared by hwpx/docx: embed/exists are the same zip operation for both formats)
# ─────────────────────────────────────────────

_REGISTRY: dict[DocFormat, OPCAdapter] = {}


def get_opc_adapter(fmt: DocFormat) -> OPCAdapter:
    try:
        return _REGISTRY[fmt]
    except KeyError:
        raise ValueError(f"No OPC adapter registered for format: {fmt!r} (registered: {list(_REGISTRY)})") from None


def parse_opc(path: str | Path, fmt: DocFormat) -> list[dict]:
    """Parse path with fmt's adapter and return the node list cleaned up for the LLM."""
    adapter = get_opc_adapter(fmt)
    path = Path(path)
    return adapter.strip(adapter.extract(path))


def embed_mre_opc(path: str | Path, mre_xml: str, fmt: DocFormat) -> None:
    """Insert/replace mre.xml at the zip root of path (hwpx/docx), in place."""
    get_opc_adapter(fmt).embed(Path(path), mre_xml)


def fetch_opc(path: str | Path, node_id: str, fmt: DocFormat) -> str:
    """Fetch node_id's full paragraph text from path (hwpx/docx). If id="full",
    returns the whole document's text. Raises FetchNotSupportedError if the
    adapter doesn't support fetch."""
    adapter = get_opc_adapter(fmt)
    if adapter.fetch is None:
        raise FetchNotSupportedError(
            f"Adapter {adapter.name!r} does not implement fetch."
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
        embed=insert_mre_into_zip,   # mre.xml as a plain zip entry: identical operation regardless of format
        exists=_mre_xml_exists_in_zip,
        fetch=_docx_fetch,
    )


_register_builtin_adapters()
