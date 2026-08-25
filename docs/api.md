# API Reference

## Package layout

| Module | Responsibility |
|---|---|
| `generate.py` | `generate_mre()` — the top-level entry point |
| `format_detect.py` | magic-byte format detection (html/pdf/hwp/hwpx/docx) |
| `html_site_adapter.py` | per-domain HTML adapter registry + `mre.site_adapters` entry-point plugin discovery (Wikipedia built in) |
| `opc_adapter.py` | HWPX/DOCX parsing and zip-based embedding |
| `nodes.py` | format-agnostic node normalization shared by all adapters |
| `appendix.py` | Wikipedia appendix-section stripping / short-section merging |
| `generation.py` | LLM prompt, schema, and chunked calling |
| `repair.py` | missing/misaligned paragraph detection and targeted regeneration |
| `xml_builder.py` | pure string assembly of the final `<mre>` XML |
| `reader.py` | reads a `<mre>` header back out of embedded HTML (`extract_mre_xml`) |
| `hwp_adapter.py` | legacy HWP (OLE2) parsing, read-only (`parse_hwp`) |
| `agent/` | opt-in agentic RAG loop — see [Agentic RAG](agentic-rag.md) |

## Generation

::: mre.generate_mre

::: mre.MREGenerationResult

## Format detection

::: mre.DocFormat

::: mre.detect_format

::: mre.FormatDetectionError

## HTML site adapters

::: mre.fetch_block

::: mre.HTMLSiteAdapter

::: mre.register_site

::: mre.get_site_adapter

::: mre.detect_site

::: mre.registered_sites

::: mre.parse_html

::: mre.compute_adapter_fingerprint

::: mre.UnknownSiteError

::: mre.FetchNotSupportedError

::: mre.GeneratorFingerprintMismatch

## HWPX / DOCX (OPC) adapters

::: mre.OPCAdapter

::: mre.parse_opc

::: mre.embed_mre_opc

::: mre.fetch_opc

::: mre.get_opc_adapter

::: mre.extract_mre_xml_opc

## Legacy HWP (parsing-only)

::: mre.parse_hwp

::: mre.hwp_adapter.build_structure_tree_hwp

## Reading a header back out

::: mre.extract_mre_xml

## Plugin discovery

::: mre.discover_plugin_adapters

## Agent (`mre.agent`)

::: mre.agent.run_agent

::: mre.agent.AgentResult

::: mre.agent.MRENotFoundError

::: mre.agent.BlockFetchError

::: mre.agent.build_progressive_action_schema

::: mre.agent.metadata_view

Smaller pieces `run_agent()` is built from, for wiring MRE into a
different agent loop instead:

| Piece | What it does |
|---|---|
| `mre.agent.SYSTEM_PROMPT` | static system prompt describing the four tools |
| `mre.agent.ANSWER_FORMAT` | small `.format(query=...)` template for the answer instruction |
| `mre.agent.CHECK_SUFFICIENCY_SCHEMA` | schema for the one-turn "is this answer actually supported by what was retrieved?" check |
| `mre.agent.MAX_TURNS` | default turn limit passed to `run_agent()` |
| `mre.agent.MAX_DOCS_PER_TURN` | cap on documents requested per `expand_document`/`fetch_doc` call |
| `mre.agent.MAX_PIDS_PER_DOC` | cap on paragraph ids requested per document per `fetch_blocks` call |
