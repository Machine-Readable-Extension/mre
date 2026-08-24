"""OPC (Open Packaging Conventions) zip-container document adapters — hwpx, docx.

Both hwpx and docx are zip archives internally, and MRE embeds into them
by adding one ``mre.xml`` file at the zip root (the archive-format
counterpart to HTML's `<head><script>` insertion). Since that embed/exists
operation is identical zip manipulation for both formats, a single pair of
functions (``insert_mre_into_zip`` / ``_mre_xml_exists_in_zip``) is shared
between them.

Only parsing (``extract``) differs per format:

- hwpx: ``<hp:p>``/``<hp:t>`` inside ``Contents/section*.xml`` — there's
  no heading concept, so only paragraphs come out.
- docx: ``<w:p>`` inside ``word/document.xml`` — classified as a heading
  if its ``pStyle`` is ``HeadingN``/``Title``, otherwise a paragraph.
  Paragraphs inside tables (``<w:tbl>``) are out of scope: hwpx's "a table
  is absorbed into the surrounding paragraph" rule has no docx
  equivalent, and absorbing table content arbitrarily would scramble
  table order — so only the body's own paragraph flow (`<w:p>` that is a
  direct child of the body) is handled.
"""

from __future__ import annotations

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
    """Parsing/embedding/fetch logic bundled for one OPC zip document format.

    Attributes:
        extract: Document path -> list of heading/paragraph nodes
            (``{"type": "heading", "level", "text"}`` |
            ``{"type": "paragraph", "id", "text"}``).
        strip: ``extract()`` output -> node list cleaned up for the LLM.
        embed: ``(document path, assembled mre xml) -> None`` — inserts
            or replaces ``mre.xml`` at the zip root, in place.
        exists: Document path -> whether ``mre.xml`` is already embedded.
        fetch: ``(document path, node id) -> that paragraph's full text``.
            ``id="full"`` returns the whole document's text, concatenated.
            An id that can't be found returns an empty string (not an
            exception) — the same contract as
            ``html_site_adapter.fetch_block()``. ``None`` means this
            adapter doesn't support fetch (``fetch_opc()`` then raises
            ``FetchNotSupportedError``).
    """
    name: str
    extract: Callable[[Path], list[dict]]
    strip: Callable[[list[dict]], list[dict]]
    embed: Callable[[Path, str], None]
    exists: Callable[[Path], bool]
    fetch: Callable[[Path, str], str] | None = None


# ─────────────────────────────────────────────
# docx parsing
# ─────────────────────────────────────────────

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_VAL = f"{_W_NS}val"
_HEADING_STYLE_RE = re.compile(r"heading\s*(\d)", re.IGNORECASE)


def _heading_level_from_style(style_val: str | None) -> int | None:
    """Extract a heading level from a Word paragraph's ``pStyle`` value.

    Relies on the convention that python-docx/LibreOffice/MS Word all
    generate locale-independent style IDs — "Heading1".."Heading9" (or
    "Title"). A custom template that renames these style IDs won't be
    detected.
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
    """Extract heading/paragraph nodes from DOCX (``word/document.xml``), in document order.

    Only ``<w:p>`` that is a direct child of the body is handled (table
    paragraphs are excluded — see the module docstring).

    Returns:
        Nodes as ``[{"type": "heading", "level": int, "text": str}
        | {"type": "paragraph", "id": "pN", "text": str}, ...]``.
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
# hwpx parsing/embedding
# ─────────────────────────────────────────────
# Each paragraph is an OWPML <hp:p> element, with its text held in nested
# <hp:t> elements. ElementTree exposes tags as ``{namespace-uri}localname``,
# so they're matched with endswith below.

_HWPX_SECTION_RE = re.compile(r"^Contents/section\d+\.xml$")
_HWPX_SECTION_IDX_RE = re.compile(r"\d+")
_MRE_ENTRY_NAME = "mre.xml"
_HWPX_MIN_PARA_CHARS = 50   # paragraphs at or under this length get merged into the paragraph that follows


def _coalesce_short_paragraphs(nodes: list[dict]) -> list[dict]:
    """Merge paragraphs of 50 characters or fewer into the next paragraph;
    if one is left over at the end, merge it into the previous paragraph instead.

    HWPX press releases often have short label paragraphs (e.g. "Press
    Release", "Release date", a standalone date) that are semantically
    part of the body paragraph that follows — grouping them together
    before sending to the LLM keeps that intent intact.
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
    """Extract paragraph nodes from an HWPX OPC zip.

    Each outermost ``<hp:p>`` counts as one paragraph. A nested ``<hp:p>``
    inside a table or text box has its text absorbed into the outer
    paragraph's text and is not added as a separate paragraph (avoiding
    double-counting).

    Returns:
        Nodes as ``[{"type": "paragraph", "id": "pN", "text": "..."}, ...]``.
    """
    nodes: list[dict] = []
    p_counter = 0

    def _section_idx(name: str) -> int:
        m = _HWPX_SECTION_IDX_RE.search(name)
        assert m is not None  # guaranteed: name already matched _HWPX_SECTION_RE, which requires \d+
        return int(m.group())

    with zipfile.ZipFile(hwpx_path, "r") as zf:
        section_files = [name for name in zf.namelist() if _HWPX_SECTION_RE.match(name)]
        section_files.sort(key=_section_idx)
        for sec in section_files:
            with zf.open(sec) as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # ElementTree doesn't expose parent pointers, so build a parent map ourselves.
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
                        # nested <hp:p> — absorbed into the outermost paragraph
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


def insert_mre_into_zip(opc_path: Path, mre_xml: str) -> None:
    """Insert ``mre.xml`` at the OPC zip root (overwriting it if already present).

    ``zipfile`` doesn't support in-place delete/modify, so this copies
    into a new zip and swaps it in atomically via ``os.replace``. The
    temporary file is created in the same directory to avoid a
    cross-device rename.
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
                    continue  # replaced with the new mre.xml
                zout.writestr(item, zin.read(item.filename))
            zout.writestr(_MRE_ENTRY_NAME, mre_xml.encode("utf-8"))
        os.replace(tmp_path, opc_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ─────────────────────────────────────────────
# fetch (shared between hwpx/docx — only extract() differs; the id lookup is identical)
# ─────────────────────────────────────────────
# Unlike html_site_adapter._wiki_fetch, no separate re-parsing logic is
# needed here: the paragraph text extract() produces is already the full,
# untruncated text (unlike Wikipedia's LLM-prompt text, which is
# truncated), so re-running extract() and indexing into it directly serves
# as fetch — generation and fetch genuinely share the same function.

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
# Adapter registration (hwpx/docx share embed/exists — identical zip manipulation either way)
# ─────────────────────────────────────────────

_REGISTRY: dict[DocFormat, OPCAdapter] = {}


def get_opc_adapter(fmt: DocFormat) -> OPCAdapter:
    try:
        return _REGISTRY[fmt]
    except KeyError:
        raise ValueError(f"No OPC adapter registered for format: {fmt!r} (registered: {list(_REGISTRY)})") from None


def parse_opc(path: str | Path, fmt: DocFormat) -> list[dict]:
    """Parse ``path`` with the ``fmt`` adapter and return nodes cleaned up for the LLM."""
    adapter = get_opc_adapter(fmt)
    path = Path(path)
    return adapter.strip(adapter.extract(path))


def embed_mre_opc(path: str | Path, mre_xml: str, fmt: DocFormat) -> None:
    """Insert or replace ``mre.xml`` at the zip root of ``path`` (hwpx/docx), in place."""
    get_opc_adapter(fmt).embed(Path(path), mre_xml)


def fetch_opc(path: str | Path, node_id: str, fmt: DocFormat) -> str:
    """Fetch ``node_id``'s full paragraph text from ``path`` (hwpx/docx).

    ``id="full"`` returns the whole document's text.

    Raises:
        FetchNotSupportedError: If the adapter for ``fmt`` doesn't implement fetch.
    """
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
        embed=insert_mre_into_zip,   # mre.xml as a plain zip entry — identical regardless of format
        exists=_mre_xml_exists_in_zip,
        fetch=_docx_fetch,
    )


_register_builtin_adapters()
