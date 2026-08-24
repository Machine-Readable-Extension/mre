from __future__ import annotations

"""
Minimal MRE XML 조립 — LLM 출력(summary + headings/keywords 병렬 배열) + 원본 node 트리를
결합해 <mre> 문서를 만든다. LLM 호출 없이 순수 문자열 조립만 한다.

data_utils/mre_generator3.py의 build_mre_xml2를 이 라이브러리 배포 경계 안으로 옮겨왔다
(이름은 build_mre_xml로 정리 — "2" 접미사는 mre_generator3.py 내부에서 v1과 구분하려던
버전 표식이었는데, 이 라이브러리엔 v1이 없어 접미사가 그냥 잡음이 된다).
"""

from xml.sax.saxutils import escape as xml_escape


def build_mre_xml(
    llm_data: dict,
    original_nodes: list[dict],
    title: str = "Untitled",
    *,
    generator: str | None = None,
    generator_fingerprint: str | None = None,
) -> str:
    """Assemble minimal MRE XML — no <tags>, no lang attribute, no <resources>.

    Structure:
        <mre version="1.0" generator="wikipedia" generator-fingerprint="a3f9c21b">
          <metadata>
            <title>...</title>
            <summary>...</summary>
          </metadata>
          <tree>
            <node id="pN">
              <desc>LLM heading</desc>
              <keys>LLM keywords</keys>
            </node>
          </tree>
        </mre>

    Paragraph-granularity(이 라이브러리가 지원하는 유일한 granularity) 는 항상 flat —
    <section> 은 생성하지 않는다. heading 타입 노드는 LLM 프롬프트 입력(build_user_prompt)
    에는 문맥으로 노출되지만, 최종 XML 트리 조립에서는 건너뛴다. 이 함수는
    data_utils/mre_generator3.py의 build_mre_xml2(paragraph-granularity 전용)를 그대로
    옮겨온 것(이름만 build_mre_xml로 정리) — mre_generator3.py 가 기준(reference)이다.
    단, generator/generator_fingerprint 두
    파라미터는 mre_generator3.py 에는 없는 이 라이브러리만의 확장이다 — mre_generator3.py
    는 어댑터 플러그인 개념 자체가 없어 대응되는 게 없다(html_site_adapter.py의
    compute_adapter_fingerprint() 참조: 이 문서를 만든 어댑터를 나중에 fetch 시점에
    식별해, 그 사이 어댑터 파싱 로직이 바뀌었는지 감지하는 용도).

    generator/generator_fingerprint 가 둘 다 주어지면 <mre> 루트 태그에
    generator/generator-fingerprint 속성으로 기록한다. 하나라도 None 이면(hwpx/docx 등
    어댑터 개념이 없는 포맷, 또는 호출자가 안 넘긴 경우) 두 속성 다 생략한다.
    """
    title_esc = xml_escape(title)
    # headings/keywords 두 parallel array 를 입력 단락 순서에 위치 기반으로 매핑.
    # 길이 불일치 시 짧은 쪽 기준으로 매핑하고 나머지는 빈 값으로 fallback.
    summary_esc  = xml_escape((llm_data.get("summary") or "").strip())
    llm_headings: list[str] = list(llm_data.get("headings", []))
    llm_keywords: list[str] = list(llm_data.get("keywords", []))

    pad = "    "
    tree_lines: list[str] = []

    para_idx = 0  # paragraph 카운터 — LLM array 의 인덱스
    for node in original_nodes:
        ntype = node.get("type", "paragraph")
        if ntype == "heading":
            continue  # flat tree — heading 은 프롬프트 문맥에만 쓰이고 XML 에는 반영 안 함
        safe_id = xml_escape(node["id"])
        heading = llm_headings[para_idx] if para_idx < len(llm_headings) else ""
        kws     = llm_keywords[para_idx] if para_idx < len(llm_keywords) else ""
        desc = xml_escape(heading)
        keys = xml_escape(kws)
        para_idx += 1
        tree_lines.append(f'{pad}<node id="{safe_id}">')
        if desc:
            tree_lines.append(f"{pad}  <desc>{desc}</desc>")
        if keys:
            tree_lines.append(f"{pad}  <keys>{keys}</keys>")
        tree_lines.append(f"{pad}</node>")

    tree_body = "\n".join(tree_lines)

    # 명시적 조립 (textwrap.dedent 금지) — dedent + 멀티라인 {tree_body} 조합은 tree 의
    # 첫 줄에만 소스 앞 공백이 붙어 인덴트가 어긋났다. tree_body 는 이미 pad+nesting 인덴트를
    # 갖고 있으므로 컬럼 0 에 그대로 삽입한다.
    summary_xml = (
        f'    <summary>\n      {summary_esc}\n    </summary>\n'
        if summary_esc else ''
    )
    root_attrs = '<mre version="1.0"'
    if generator is not None and generator_fingerprint is not None:
        root_attrs += f' generator="{xml_escape(generator)}"'
        root_attrs += f' generator-fingerprint="{xml_escape(generator_fingerprint)}"'
    root_attrs += '>'
    mre = (
        f'{root_attrs}\n'
        f'  <metadata>\n'
        f'    <title>{title_esc}</title>\n'
        f'{summary_xml}'
        f'  </metadata>\n'
        f'  <tree>\n'
        f'{tree_body}\n'
        f'  </tree>\n'
        f'</mre>'
    )
    return mre
