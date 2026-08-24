from __future__ import annotations

"""
Minimal reference implementation of an `mre.site_adapters` entry-point plugin.

This is what a site owner (or a company managing a domain) publishes to add
support for their own site to `mre`, without touching `mre`'s own source —
`mre` discovers `ADAPTER` below purely via the entry point declared in this
package's pyproject.toml:

    [project.entry-points."mre.site_adapters"]
    example = "mre_example_adapter:ADAPTER"

The parsing convention here is intentionally toy (every <p> under <article>
is one paragraph) — a real adapter matches whatever markup the target site
actually uses, the way mre's own built-in Wikipedia adapter matches
Wikipedia's `mw-heading` div structure.
"""

import re

from bs4 import BeautifulSoup, Tag

from mre import HTMLSiteAdapter


def _paragraphs(soup: BeautifulSoup) -> list[Tag]:
    """<article> 아래 <p> 만 문단으로 취급 (toy convention for example.com)."""
    container = soup.find("article") or soup
    return [p for p in container.find_all("p") if p.get_text(strip=True)]


def extract(soup: BeautifulSoup) -> list[dict]:
    nodes: list[dict] = []
    for i, p in enumerate(_paragraphs(soup), start=1):
        nodes.append({
            "type": "paragraph",
            "id": f"p{i}",
            "text": p.get_text(separator=" ", strip=True),
        })
    return nodes


def strip(nodes: list[dict]) -> list[dict]:
    return nodes  # 이미 LLM 전송용 형태 — 추가 정리 불필요


def embed(html: str, mre_xml: str) -> str:
    tag = f'\n<script type="application/mre+xml">\n{mre_xml}\n</script>\n'
    idx = html.lower().find("</head>")
    if idx != -1:
        return html[:idx] + tag + html[idx:]
    return tag + html


def fetch(html: str, node_id: str) -> str:
    """extract()와 동일한 규칙(<article> 아래 <p>)으로 다시 걸어서 node_id 문단의
    전체 텍스트를 가져온다 — 접두 알파벳은 무시하고 끝의 숫자만 본다(id 스킴이
    바뀌어도 안전). id="full"이면 문서 전체 텍스트를 이어붙여 반환."""
    soup = BeautifulSoup(html, "lxml")
    paragraphs = _paragraphs(soup)

    if node_id == "full":
        return "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    m = re.match(r"^[A-Za-z]*(\d+)$", node_id)
    if not m:
        return ""
    idx = int(m.group(1))
    if not (1 <= idx <= len(paragraphs)):
        return ""
    return paragraphs[idx - 1].get_text(separator=" ", strip=True)


ADAPTER = HTMLSiteAdapter(
    name="example",
    domains=("example.com",),
    extract=extract,
    strip=strip,
    embed=embed,
    fetch=fetch,
)
