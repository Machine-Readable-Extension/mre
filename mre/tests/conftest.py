import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def david_lightfoot_html() -> str:
    """Real Wikipedia article HTML (from data_v2gen/knowledge_base.db), small
    but with multiple <h2> sections and 5 paragraphs — enough to exercise
    heading/paragraph mixing, letter-prefix ids, and multi-paragraph fetch."""
    return (FIXTURES_DIR / "david_lightfoot.html").read_text(encoding="utf-8")


@pytest.fixture
def sample_docx(tmp_path) -> Path:
    """Copied into tmp_path so embed()-ing mre.xml into it doesn't mutate the fixture."""
    dst = tmp_path / "sample.docx"
    shutil.copy(FIXTURES_DIR / "sample.docx", dst)
    return dst


@pytest.fixture
def sample_hwpx(tmp_path) -> Path:
    """Copied into tmp_path so embed()-ing mre.xml into it doesn't mutate the fixture."""
    dst = tmp_path / "sample.hwpx"
    shutil.copy(FIXTURES_DIR / "sample.hwpx", dst)
    return dst


@pytest.fixture
def animal_shelter_hwp() -> Path:
    """Real, compressed legacy HWP (Gangwon-do public-data animal-shelter status
    table) -- unlike the synthetic fixtures built by _ole2_builder.py, this
    exercises the real BodyText compression + inline-control-character mix a
    hand-crafted record stream can't fully stand in for. Read-only (hwp_adapter
    has no embed path), so no tmp_path copy needed."""
    return FIXTURES_DIR / "animal_shelter_status.hwp"


@pytest.fixture
def construction_safety_cost_notice_hwp() -> Path:
    """Real, compressed legacy HWP (a 2017 Ministry of Employment and Labor
    notice, prose/legal-citation text rather than animal_shelter_hwp's table
    layout) -- covers a second, structurally different real-world document:
    long sentences, revision-history entries, and CJK bracket punctuation
    (｢｣「」·․) instead of table cells. Read-only, so no tmp_path copy needed."""
    return FIXTURES_DIR / "construction_safety_cost_notice.hwp"


@pytest.fixture
def sample_prose_pdf(tmp_path) -> Path:
    """2-page PDF (built with reportlab) with paragraphs separated by extra
    vertical leading -- exercises the layout-mode blank-line paragraph split
    and cross-page id numbering. Copied into tmp_path so embed_mre_pdf()-ing
    mre.xml into it doesn't mutate the fixture (same reasoning as sample_docx/
    sample_hwpx)."""
    dst = tmp_path / "sample_prose.pdf"
    shutil.copy(FIXTURES_DIR / "sample_prose.pdf", dst)
    return dst


@pytest.fixture
def no_paragraph_breaks_pdf() -> Path:
    """1-page PDF where every line uses identical leading -- no vertical gap
    anywhere, so layout-mode extraction never emits a blank line. Exercises
    the whole-page-as-one-paragraph fallback."""
    return FIXTURES_DIR / "no_paragraph_breaks.pdf"
