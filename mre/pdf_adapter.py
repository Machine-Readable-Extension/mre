from __future__ import annotations

"""
PDF paragraph text extraction, plus embed/fetch via PDF file attachments.

Unlike HTML/HWPX/DOCX, PDF has no reliable structural markup for
paragraphs -- a PDF's content stream just positions glyphs on a page, and
extractors like pypdf reconstruct "lines" from that positioning. There is
no universal signal for "these lines form one paragraph" the way there is
a `<p>` tag or a `HWPTAG_PARA_TEXT` record. This module uses pypdf's
`extraction_mode="layout"`, which reconstructs vertical whitespace as
blank lines proportional to the actual gap between lines on the page --
a paragraph's extra leading (even a few points beyond normal line height)
shows up as one or more blank lines in the extracted text. A run of
non-blank lines, separated from the next run by a blank line, is treated
as one paragraph. Plain-mode extraction (pypdf's default) collapses all
vertical gaps to a single "\n" and loses this signal entirely, which is
why this module explicitly requests layout mode instead. A PDF that never
has extra spacing between paragraphs (some do) falls back to one
paragraph per page rather than one paragraph per visual line, which would
be far noisier (every wrapped line would become its own "paragraph").

Generation itself (generate_mre(fmt=DocFormat.PDF, ...)) still raises
NotImplementedError -- there's no LLM-driven authoring step here, only
parsing an existing PDF and attaching an already-built mre.xml to it (see
the embed/fetch note below).

Only extracts the text layer -- scanned/image-only PDFs (no embedded text)
return no paragraphs; that needs OCR, a different problem this module
doesn't attempt.

Some PDFs draw bullet/marker glyphs (list dots, section markers) through a
custom symbol font with no ToUnicode mapping, so pypdf has nothing to
extract but the font's raw private-use-area codepoint (observed on a real
Korean corporate report: a bullet drawn as U+F000, e.g. "-1 (...)").
These carry no recoverable text meaning, so -- mirroring hwp_adapter.py's
leftover-control-character safety net -- they're stripped from the final
text alongside any stray C0 control characters.

**Embed/fetch**: PDF has a standard container mechanism for this, file
attachments (`/EmbeddedFiles` in the document catalog's Names tree,
PDF32000-1:2008 Section 7.11.3) -- the same role HWPX/DOCX's "extra zip
entry" plays. A reader that doesn't recognize it just ignores it, and it
doesn't touch the page content streams at all (verified: build_structure_
tree_pdf() returns byte-identical output before and after embedding).
pypdf.PdfWriter supports this natively (add_attachment / PdfReader.
attachments), so embed_mre_pdf() writes an "mre.xml" attachment, mirroring
insert_mre_into_zip()'s "mre.xml" zip entry in opc_adapter.py. pypdf has no
attachment-removal API, so a re-embed manually prunes any existing
"mre.xml" entry out of the Names/EmbeddedFiles array first (see
_remove_existing_mre_attachment) -- otherwise attachments accumulate under
the same name instead of being replaced. fetch_pdf() reuses
build_structure_tree_pdf() directly, the same single-source-of-truth
principle fetch_opc() uses (see mre.nodes for the shared id-lookup logic).
"""

import os
import re
import tempfile
import unicodedata
from pathlib import Path

import pypdf

from mre.nodes import fetch_paragraph_by_id, strip_to_text_nodes

# Two or more consecutive newlines (allowing whitespace-only lines in between)
# mark a paragraph break; anything else is just a wrapped line within one paragraph.
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_RUNS_OF_SPACES_RE = re.compile(r" {2,}")

# "Cc" = control characters, "Co" = private-use-area (unmapped custom-font glyphs,
# see module docstring). Neither carries recoverable text meaning.
_DROP_CATEGORIES = ("Cc", "Co")


def _clean_line(line: str) -> str:
    """Drop unrecoverable glyph codepoints from one line, then collapse the
    whitespace gaps that removing them tends to leave behind."""
    filtered = "".join(ch for ch in line if unicodedata.category(ch) not in _DROP_CATEGORIES)
    return _RUNS_OF_SPACES_RE.sub(" ", filtered).strip()


def _split_paragraphs(page_text: str) -> list[str]:
    """Split one page's extracted text into paragraph strings (see module docstring
    for the blank-line heuristic and its fallback)."""
    blocks = _PARAGRAPH_BREAK_RE.split(page_text)
    paragraphs: list[str] = []
    for block in blocks:
        # Within a block, wrapped lines get joined with a space -- a mid-paragraph
        # line break in the PDF is a layout artifact, not a real paragraph boundary.
        cleaned_lines = [_clean_line(line) for line in block.splitlines()]
        joined = " ".join(line for line in cleaned_lines if line)
        if joined:
            paragraphs.append(joined)
    return paragraphs


def build_structure_tree_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract paragraph nodes from a PDF's text layer, in document order.

    No heading concept, same as HWPX/HWP -- PDF has no reliable equivalent of
    DOCX's pStyle or HTML's <h1-6> without font-size heuristics, which this
    module doesn't attempt (see module docstring).

    Returns
    -------
    nodes : [{"type": "paragraph", "id": "pN", "text": "..."}, ...]
    """
    reader = pypdf.PdfReader(str(pdf_path))
    nodes: list[dict] = []
    p_counter = 0
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        for para_text in _split_paragraphs(text):
            p_counter += 1
            nodes.append({"type": "paragraph", "id": f"p{p_counter}", "text": para_text})
    return nodes


def parse_pdf(path: str | Path) -> list[dict]:
    """Parse path and return the LLM-ready stripped node list (parse_opc/parse_hwp counterpart)."""
    return strip_to_text_nodes(build_structure_tree_pdf(path))


# ─────────────────────────────────────────────
# embed / exists / fetch (mirrors insert_mre_into_zip / _mre_xml_exists_in_zip /
# fetch_opc in opc_adapter.py -- see module docstring for why a PDF file
# attachment is the right analog to a zip entry here)
# ─────────────────────────────────────────────

_MRE_ATTACHMENT_NAME = "mre.xml"


def _remove_existing_mre_attachment(writer: pypdf.PdfWriter) -> None:
    """Prune any existing "mre.xml" entries from the writer's Names/EmbeddedFiles
    array before add_attachment() adds a fresh one -- pypdf has no attachment
    removal API, and add_attachment() alone would accumulate duplicates under
    the same name on every re-embed instead of replacing. Non-mre attachments
    (if the source PDF already had unrelated ones) are left untouched.

    Reaches into pypdf's private object graph (_root_object) rather than a
    public API, because none exists for this. The Names/EmbeddedFiles tree
    shape is fixed by the PDF spec itself (PDF32000-1:2008 Section 7.7.4),
    not by pypdf, so it's unlikely to shift across pypdf versions -- but if
    it ever does, the KeyError/TypeError fallback below just skips pruning
    (re-embed silently falls back to accumulating, not to a crash).
    """
    try:
        # pypdf types _root_object as the generic PdfObject base, not the concrete
        # DictionaryObject/ArrayObject it actually is here -- deliberate reach into
        # untyped internals (see docstring), not a real type mismatch.
        names_arr = writer._root_object["/Names"]["/EmbeddedFiles"]["/Names"]  # type: ignore[index]
    except (KeyError, TypeError):
        return  # no prior attachments at all -- nothing to remove
    pruned = []
    for i in range(0, len(names_arr), 2):
        name, ref = names_arr[i], names_arr[i + 1]
        if name != _MRE_ATTACHMENT_NAME:
            pruned.extend([name, ref])
    del names_arr[:]
    names_arr.extend(pruned)


def embed_mre_pdf(path: str | Path, mre_xml: str) -> None:
    """Embed mre_xml into path as an "mre.xml" file attachment (in place,
    replacing any prior one). Page content is untouched."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    writer = pypdf.PdfWriter(clone_from=str(path))
    _remove_existing_mre_attachment(writer)
    writer.add_attachment(_MRE_ATTACHMENT_NAME, mre_xml.encode("utf-8"))

    # Same atomic-replace pattern as insert_mre_into_zip(): create the temp
    # file in the same directory to avoid a cross-device rename, and leave
    # the original file untouched on failure.
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".mre_tmp_", suffix=".pdf", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "wb") as f:
            writer.write(f)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def mre_xml_exists_pdf(path: str | Path) -> bool:
    """Whether mre.xml is already attached to path."""
    try:
        return _MRE_ATTACHMENT_NAME in pypdf.PdfReader(str(path)).attachments
    except (pypdf.errors.PdfReadError, FileNotFoundError):
        return False


def extract_mre_xml_pdf(path: str | Path) -> str | None:
    """Return the raw content of the mre.xml attached to path. None if absent.

    The PDF counterpart to mre.opc_adapter.extract_mre_xml_opc(). Even though
    multiple versions could theoretically linger from re-embedding
    (_remove_existing_mre_attachment already cleans up at embed time, but this
    function stays defensive about it), the most recent one [-1] is treated as
    authoritative."""
    try:
        entries = pypdf.PdfReader(str(path)).attachments.get(_MRE_ATTACHMENT_NAME)
    except (pypdf.errors.PdfReadError, FileNotFoundError):
        return None
    if not entries:
        return None
    try:
        return entries[-1].decode("utf-8")
    except UnicodeDecodeError:
        return None


def fetch_pdf(path: str | Path, node_id: str) -> str:
    """Fetch node_id's full paragraph text from path. If id="full", returns the
    whole document's text. Returns an empty string (not an exception) if not
    found -- same contract as fetch_opc()."""
    return fetch_paragraph_by_id(build_structure_tree_pdf(path), node_id)
