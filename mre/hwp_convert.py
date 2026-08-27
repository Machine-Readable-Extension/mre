from __future__ import annotations

"""
Best-effort HWP -> DOCX/PDF conversion via an externally-installed LibreOffice
(+ H2Orestart extension, https://github.com/ebandal/H2Orestart), so an mre.xml
can be embedded into the *converted* file using the already-solid docx/pdf
embed pipelines -- mre.hwp_adapter itself stays read-only (see its module
docstring for why: no maintained pure-Python OLE2/CFB writer exists that this
library is willing to trust with a user's real document).

**Deliberately NOT wired into generate_mre()'s fmt dispatch, and never called
implicitly.** Two independent reasons:

1. Hard external dependency that isn't pip/conda-installable as part of this
   library: LibreOffice itself (a full desktop office suite, hundreds of MB)
   plus a third-party community extension. Every other generate_mre() branch
   needs nothing beyond this library's own pip dependencies. Making fmt=HWP
   silently spawn a subprocess and require a system binary would be a
   surprising, platform-fragile exception to that.
2. Measured, real content loss -- not just theoretical risk. Tested against
   mre/tests/fixtures/construction_safety_cost_notice.hwp (a real 27-table
   government notice): the opening prose paragraphs matched
   mre.hwp_adapter.parse_hwp()'s own output character-for-character, and
   ~86% of total characters survived, but a revision-history entry ("고시
   제88 - 13호") was not found anywhere in the converted document at all --
   genuine loss from the H2Orestart import filter itself, not merely
   mre.opc_adapter's documented table-cell-paragraph exclusion. This is a
   community reverse-engineered filter, not Hancom's own converter, and it
   should be treated as best-effort: verify important documents' output
   before trusting it, especially anything table-heavy.

**Usage is a two-step, explicit pipeline** -- this module produces a file,
it does not itself embed anything:

    from mre import DocFormat, convert_hwp, generate_mre

    docx_path = convert_hwp("report.hwp", target=DocFormat.DOCX)
    result = await generate_mre(docx_path, client=client, model=model,
                                 title="...", fmt=DocFormat.DOCX)

**Setup** (not a pip dependency of this library): install LibreOffice, then
the H2Orestart .oxt via `unopkg add path/to/H2Orestart.oxt` so HWP import
works, and confirm `soffice` resolves on PATH (or pass soffice_bin=).
"""

import logging
import subprocess
from pathlib import Path

from mre.format_detect import DocFormat

logger = logging.getLogger(__name__)

_TARGET_EXTENSIONS = {
    DocFormat.DOCX: "docx",
    DocFormat.PDF: "pdf",
}


class LibreOfficeNotAvailableError(RuntimeError):
    """Raised when soffice_bin can't be run from PATH (or the given path)."""


class HwpConversionError(RuntimeError):
    """Raised when the soffice conversion itself fails (non-zero exit code, timeout, no output file, etc.)."""


def convert_hwp(
    path: str | Path,
    target: DocFormat = DocFormat.DOCX,
    *,
    outdir: str | Path | None = None,
    soffice_bin: str = "soffice",
    timeout: float = 120.0,
) -> Path:
    """Convert path (.hwp) to target (DOCX or PDF) via LibreOffice (+H2Orestart)
    and return the resulting file's path. Best-effort — see the module
    docstring for measured real-world content loss.

    Logs at WARNING level on every call (even with no handler configured,
    Python's default lastResort handler prints to stderr, so it's visible
    without any logging setup).

    Parameters
    ----------
    target  : only DocFormat.DOCX (default) or DocFormat.PDF are supported.
    outdir  : directory to write the conversion result to. Defaults to the same directory as path.
    soffice_bin : the soffice executable. Pass an absolute path if it's not on PATH.
    timeout : how long to wait for the conversion subprocess, in seconds.

    Raises
    ------
    ValueError : if target is not DOCX/PDF.
    FileNotFoundError : if path does not exist.
    LibreOfficeNotAvailableError : if soffice_bin cannot be run.
    HwpConversionError : if the conversion subprocess failed (non-zero exit
        code), timed out, or produced no output file.
    """
    if target not in _TARGET_EXTENSIONS:
        raise ValueError(f"target은 DocFormat.DOCX 또는 DocFormat.PDF만 지원합니다: {target!r}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = _TARGET_EXTENSIONS[target]
    logger.warning(
        "convert_hwp(): %r -> .%s 변환은 LibreOffice+H2Orestart(커뮤니티 리버스엔지니어링 "
        "필터) 기반 best-effort입니다. 실제 정부 문서 테스트에서 표 안 콘텐츠 일부가 "
        "누락되는 것을 확인했습니다 — 중요 문서는 변환 결과를 직접 검증하세요.",
        str(path), ext,
    )

    outdir = Path(outdir) if outdir is not None else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            [
                soffice_bin, "--headless", "--norestore",
                "--convert-to", ext, "--outdir", str(outdir), str(path),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as e:
        raise LibreOfficeNotAvailableError(
            f"{soffice_bin!r}를 실행할 수 없습니다. LibreOffice(+H2Orestart 확장, "
            "https://github.com/ebandal/H2Orestart)가 설치되어 있어야 합니다 — "
            "mre/README.md의 'Legacy HWP' 섹션 참조."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise HwpConversionError(f"{path} -> {ext} 변환이 {timeout}초 안에 끝나지 않았습니다.") from e

    out_path = outdir / f"{path.stem}.{ext}"
    if proc.returncode != 0 or not out_path.exists():
        raise HwpConversionError(
            f"{path} -> {ext} 변환 실패 (returncode={proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )
    return out_path
