"""Top-level entry point combining format detection (or an explicit
override), site/format adapter dispatch, and LLM generation — ``generate_mre()``.

Model choice is never hardcoded by this library: the caller passes an
``openai.AsyncOpenAI`` client and a model name directly. An OpenAI-compatible
backend like vLLM is the same client type with a different ``base_url``,
so this function never needs to know what backend it's talking to — it
always deals in just the ``(client, model)`` pair.

Supported formats: html, hwpx, docx. pdf/hwp have no adapter yet and raise
``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import openai
from bs4 import BeautifulSoup

from mre.format_detect import DocFormat, detect_format
from mre.generation import call_llm_chunked_async
from mre.html_site_adapter import HTMLSiteAdapter, compute_adapter_fingerprint, get_site_adapter
from mre.llm_util import MODEL_CTX, _merge_stats, _new_stats
from mre.opc_adapter import get_opc_adapter
from mre.repair import _MISALIGN_MAX_RETRIES, repair_misaligned_alignment_async
from mre.xml_builder import build_mre_xml


@dataclass
class MREGenerationResult:
    format: DocFormat
    title: str
    mre_xml: str
    embedded_html: str | None = None    # populated only when format=HTML
    embedded_path: Path | None = None   # populated only when format=HWPX/DOCX (same path as source, updated in place)
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
    """Generate MRE XML from ``source``.

    Args:
        source: Raw HTML text (``str``) when ``format=html``; a file path
            (``str | Path``) for hwpx/docx.
        client: OpenAI-compatible async client for LLM calls — the
            official SDK or a vLLM/other OpenAI-compatible server both
            work, only ``base_url`` differs.
        model: Model name to pass to ``client``. The library sets no
            default, so the caller must always specify one.
        title: Document title, placed in MRE's ``<metadata><title>``. Not
            auto-extracted — the caller supplies it, since where a title
            lives varies by document source.
        url: The document's original URL, used to pick a site-specific
            parsing adapter when ``format=html``. Since HTML source is
            raw text (``str``), passing it straight to ``detect_format``
            would be mistaken for a "file path" (see ``mre.format_detect``
            for why) — so when ``fmt`` isn't given, supplying ``url``
            settles ``fmt=HTML`` on its own.
        fmt: Explicit format, skipping auto-detection. When omitted, it's
            resolved as HTML (if ``url`` is given) -> ``detect_format(source)``.
        html_fallback_adapter: A fallback HTML adapter to use when
            ``url``'s domain isn't registered.
        repair: Whether to run the regeneration pass for tail-truncated
            paragraphs (where the headings/keywords arrays came back
            shorter than the paragraph count). On by default.
        repair_misaligned: Whether to also regenerate paragraphs that fail
            keyword grounding (entities bled in from an adjacent
            paragraph). Off by default — this regeneration was found to
            strip cross-paragraph search signal and hurt retrieval
            precision.
        embed: If ``True`` (default), inserts the result into the
            document itself — for HTML, the new string is returned; for
            hwpx/docx, the ``source`` file is updated in place. If
            ``False``, only ``mre_xml`` is produced and the document is
            left untouched.

    Returns:
        The generation result.
    """
    if fmt is None:
        fmt = DocFormat.HTML if url is not None else detect_format(source)

    if fmt is DocFormat.HTML:
        if url is None:
            raise ValueError("format=html generation requires url (to pick a site-specific adapter).")
        html = source
        site_adapter = get_site_adapter(url, fallback=html_fallback_adapter)
        soup = BeautifulSoup(html, "lxml")
        if site_adapter.preprocess is not None:
            # Cleans up the soup in-place (e.g. removing appendix sections,
            # merging short sections). embed always operates on the
            # original html string, not this soup (see below).
            site_adapter.preprocess(soup)
        raw_nodes = site_adapter.extract(soup)
        if site_adapter.assign_ids is not None:
            # Rewrites paragraph ids in-place (e.g. Wikipedia's
            # first-letter-of-title prefix, which reduces cross-document
            # id collisions).
            site_adapter.assign_ids(raw_nodes, title)
        nodes = site_adapter.strip(raw_nodes)
    elif fmt in (DocFormat.HWPX, DocFormat.DOCX):
        path = Path(source)
        opc_adapter = get_opc_adapter(fmt)
        nodes = opc_adapter.strip(opc_adapter.extract(path))
    else:
        raise NotImplementedError(f"No adapter yet for {fmt.value} (supported: html, hwpx, docx)")

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
        # Stamped into the document so fetch_block() can later recompute
        # and compare this value — the only way to tell whether the
        # adapter's parsing logic (and thus its id-to-paragraph mapping)
        # has changed since generation.
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
        else:
            opc_adapter.embed(path, mre_xml)
            result.embedded_path = path

    return result
