from __future__ import annotations

"""
generate_mre(): the top-level entry point that ties together format
detection (or an explicit override), site/format adapter dispatch, and
LLM generation.

The library doesn't hardcode a model choice: the caller passes an
openai.AsyncOpenAI client and a model name directly. An OpenAI-compatible
backend like vLLM is the same client type with a different base_url, so this
function never needs to know which backend it's talking to. It only ever
deals with the (client, model) pair.

Supported formats: html, hwpx, docx, pdf. hwp has no adapter yet, so it
raises NotImplementedError. pdf supports parsing/embed/fetch (see
mre.pdf_adapter) but not generation: all this function does for a pdf is
attach an LLM-generated mre_xml to an already-existing pdf. The LLM never
authors the pdf itself, so the code path is otherwise identical to the
other formats.
"""

from dataclasses import dataclass, field
from pathlib import Path

import openai
from bs4 import BeautifulSoup

from mre.format_detect import DocFormat, detect_format
from mre.generation import call_llm_chunked_async
from mre.html_site_adapter import HTMLSiteAdapter, compute_adapter_fingerprint, get_site_adapter
from mre.llm_util import MODEL_CTX, _merge_stats, _new_stats
from mre.opc_adapter import get_opc_adapter
from mre.pdf_adapter import embed_mre_pdf, parse_pdf
from mre.repair import _MISALIGN_MAX_RETRIES, repair_misaligned_alignment_async
from mre.xml_builder import build_mre_xml


@dataclass
class MREGenerationResult:
    format: DocFormat
    title: str
    mre_xml: str
    embedded_html: str | None = None    # only set when format=HTML
    embedded_path: Path | None = None   # only set when format=HWPX/DOCX/PDF (same path as source, updated in place)
    stats: dict = field(default_factory=dict)


async def generate_mre(
    source: str | Path,
    *,
    client: openai.AsyncOpenAI,
    model: str,
    title: str,
    url: str | None = None,
    fmt: DocFormat | None = None,
    html_fallback_adapter: HTMLSiteAdapter | None = None,
    guided: bool = True,
    model_ctx: int = MODEL_CTX,
    repair: bool = True,
    repair_misaligned: bool = False,
    misalign_retries: int = _MISALIGN_MAX_RETRIES,
    embed: bool = True,
) -> MREGenerationResult:
    """Generate MRE XML from source.

    Parameters
    ----------
    source : the raw HTML text (str) when format=html; a file path (str | Path)
             for hwpx/docx/pdf.
    client : the OpenAI-compatible async client used for LLM calls. Can be the
             official openai SDK or any OpenAI-compatible backend like vLLM —
             only the base_url differs.
    model  : the model name passed to client. The library does not enforce a
             default, so the caller must always specify it.
    title  : the document title that goes into MRE's <metadata><title>. Not
             auto-extracted — the caller supplies it (title placement varies
             too much across document sources).
    url    : the document's original URL, used to pick a per-site parsing
             adapter when format=html. Since an html source is raw text (str),
             passing it straight to detect_format would be mistaken for a
             "file path" (see mre.format_detect for details), so when fmt is
             not given, supplying url alone is enough to settle fmt=HTML.
    fmt    : specify the format directly to skip auto-detection. When omitted,
             resolution order is (HTML if url is given) -> detect_format(source).
    html_fallback_adapter : fallback HTML adapter used when url's domain isn't
             registered.
    repair : whether to run the tail-gap repair pass (headings/keywords arrays
             shorter than the paragraph count, leaving the tail empty). On by
             default (matches mre_generator3.py CLI's default).
    repair_misaligned : whether to also repair keyword-grounding failures
             (entities from an adjacent paragraph bleeding in). Off by default —
             mre_generator3.py found this regeneration erases cross-paragraph
             retrieval signal and hurts retrieval precision.
    embed  : if True (default), actually inserts the result into the document —
             for html this returns the new embedded string, for hwpx/docx it
             updates the source file in place. If False, only builds mre_xml
             without touching the document.
    """
    if fmt is None:
        fmt = DocFormat.HTML if url is not None else detect_format(source)

    if fmt is DocFormat.HTML:
        if url is None:
            raise ValueError("format=html generation requires url (used to pick the site-specific adapter).")
        if not isinstance(source, str):
            raise TypeError(
                f"format=html generation requires source to be raw HTML text (str), got: {type(source)}"
            )
        html = source
        site_adapter = get_site_adapter(url, fallback=html_fallback_adapter)
        soup = BeautifulSoup(html, "lxml")
        if site_adapter.preprocess is not None:
            # Cleans up soup in place (e.g. dropping appendix sections, merging
            # short sections). embed always operates on the original html
            # string below, never on this soup.
            site_adapter.preprocess(soup)
        raw_nodes = site_adapter.extract(soup)
        if site_adapter.assign_ids is not None:
            # Rewrites paragraph ids in place (e.g. Wikipedia's title-initial
            # prefix, to reduce cross-document id collisions; same rule as
            # mre_generator3.py).
            site_adapter.assign_ids(raw_nodes, title)
        nodes = site_adapter.strip(raw_nodes)
    elif fmt in (DocFormat.HWPX, DocFormat.DOCX):
        path = Path(source)
        opc_adapter = get_opc_adapter(fmt)
        nodes = opc_adapter.strip(opc_adapter.extract(path))
    elif fmt is DocFormat.PDF:
        path = Path(source)
        nodes = parse_pdf(path)
    else:
        raise NotImplementedError(f"No adapter for {fmt.value} yet (supported: html, hwpx, docx, pdf)")

    llm_data, gen_stats = await call_llm_chunked_async(
        client, title, nodes, model=model, guided=guided, model_ctx=model_ctx,
    )
    stats = _new_stats()
    _merge_stats(stats, gen_stats)

    if repair:
        llm_data, repair_stats, _counts = await repair_misaligned_alignment_async(
            client, title, nodes, llm_data,
            model=model, guided=guided, model_ctx=model_ctx,
            max_retries=misalign_retries, include_misaligned=repair_misaligned,
        )
        _merge_stats(stats, repair_stats)

    if fmt is DocFormat.HTML:
        # Stamped into the document so a later fetch_block() can recompute
        # and compare it. This is the only way to tell, when site_adapter.fetch()
        # runs later, whether the adapter's parsing logic changed in the
        # meantime (which could have shifted the id-to-paragraph mapping).
        mre_xml = build_mre_xml(
            llm_data, nodes, title=title,
            generator=site_adapter.name,
            generator_fingerprint=compute_adapter_fingerprint(site_adapter),
        )
    else:
        mre_xml = build_mre_xml(llm_data, nodes, title=title)
    result = MREGenerationResult(format=fmt, title=title, mre_xml=mre_xml, stats=stats)

    if embed:
        if fmt is DocFormat.HTML:
            result.embedded_html = site_adapter.embed(html, mre_xml)
        elif fmt is DocFormat.PDF:
            embed_mre_pdf(path, mre_xml)
            result.embedded_path = path
        else:
            opc_adapter.embed(path, mre_xml)
            result.embedded_path = path

    return result
