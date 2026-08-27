from __future__ import annotations

"""
포맷 감지(또는 명시적 지정) + 사이트/포맷 어댑터 dispatch + LLM 생성을 하나로
묶은 최상위 진입점 — generate_mre().

모델 선택은 라이브러리가 강제하지 않는다: 호출자가 openai.AsyncOpenAI 클라이언트와
model 이름을 직접 넘긴다. vLLM 같은 OpenAI-호환 백엔드도 base_url만 다른 동일한
클라이언트 타입이라 이 함수는 백엔드가 무엇인지 몰라도 된다 — 항상 (client, model)
두 인자로만 다룬다.

지원 포맷: html, hwpx, docx, pdf. hwp는 아직 어댑터가 없어 NotImplementedError.
pdf는 파싱/embed/fetch는 되지만(mre.pdf_adapter 참조) "생성"은 여전히 안 된다 —
여기서 하는 일은 LLM으로 새로 만든 mre_xml을 이미 존재하는 pdf에 첨부하는 것뿐,
LLM이 pdf 자체를 authoring하는 게 아니라서 다른 포맷과 코드 경로는 동일하다.
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
    embedded_html: str | None = None    # format=HTML일 때만 채워짐
    embedded_path: Path | None = None   # format=HWPX/DOCX/PDF일 때만 채워짐 (source와 동일 경로, in-place 갱신됨)
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
            raise ValueError("format=html 생성에는 url이 필요합니다 (사이트별 어댑터 선택용).")
        if not isinstance(source, str):
            raise TypeError(
                f"format=html 생성에는 source가 원문 HTML 텍스트(str)여야 합니다: {type(source)}"
            )
        html = source
        site_adapter = get_site_adapter(url, fallback=html_fallback_adapter)
        soup = BeautifulSoup(html, "lxml")
        if site_adapter.preprocess is not None:
            # in-place로 soup를 정리(예: 부록 section 제거, 짧은 section 통합).
            # embed는 이 soup가 아니라 항상 원본 html 문자열에 대해 수행한다 (아래 참조).
            site_adapter.preprocess(soup)
        raw_nodes = site_adapter.extract(soup)
        if site_adapter.assign_ids is not None:
            # in-place로 문단 id 를 재작성 (예: Wikipedia 의 제목 첫 글자 접두어 —
            # cross-document id collision 완화, mre_generator3.py 와 동일 규칙).
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
        raise NotImplementedError(f"{fmt.value} 어댑터는 아직 없음 (지원: html, hwpx, docx, pdf)")

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
        # fetch_block() 이 나중에 이 값을 재계산해 비교할 수 있도록 문서에 새겨둔다 —
        # site_adapter.fetch() 를 나중에 실행할 때 그 사이 어댑터 파싱 로직이 바뀌었는지
        # (id-to-paragraph 매핑이 어긋났을 수 있는지) 알아내는 유일한 방법.
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
