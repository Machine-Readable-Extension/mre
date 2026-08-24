from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def david_lightfoot_html() -> str:
    """Real Wikipedia article HTML (from data_v2gen/knowledge_base.db), small
    but with multiple <h2> sections and 5 paragraphs — enough to exercise
    heading/paragraph mixing, letter-prefix ids, and multi-paragraph fetch."""
    return (FIXTURES_DIR / "david_lightfoot.html").read_text(encoding="utf-8")
