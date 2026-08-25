__version__ = "1.1.0"

from mre.format_detect import DocFormat, FormatDetectionError, detect_format
from mre.html_site_adapter import (
    FetchNotSupportedError,
    GeneratorFingerprintMismatch,
    HTMLSiteAdapter,
    UnknownSiteError,
    compute_adapter_fingerprint,
    detect_site,
    discover_plugin_adapters,
    fetch_block,
    get_site_adapter,
    parse_html,
    register_site,
    registered_sites,
)
from mre.opc_adapter import (
    OPCAdapter,
    embed_mre_opc,
    extract_mre_xml_opc,
    fetch_opc,
    get_opc_adapter,
    parse_opc,
)
from mre.generate import MREGenerationResult, generate_mre
from mre.hwp_adapter import parse_hwp
from mre.hwp_convert import (
    HwpConversionError,
    LibreOfficeNotAvailableError,
    convert_hwp,
)
from mre.pdf_adapter import (
    embed_mre_pdf,
    extract_mre_xml_pdf,
    fetch_pdf,
    mre_xml_exists_pdf,
    parse_pdf,
)
from mre.reader import extract_mre_xml

# 내장 어댑터(Wikipedia 등)는 mre.html_site_adapter 모듈 로드 시 이미 등록됐다. 플러그인
# 발견은 여기, mre 패키지의 다른 모든 export 가 이미 바인딩된 뒤에 실행해야 한다 — 플러그인이
# 관례대로 `from mre import HTMLSiteAdapter` 로 임포트할 때 순환 임포트가 나지 않도록.
discover_plugin_adapters()

__all__ = [
    "__version__",
    "DocFormat",
    "FormatDetectionError",
    "detect_format",
    "FetchNotSupportedError",
    "GeneratorFingerprintMismatch",
    "HTMLSiteAdapter",
    "UnknownSiteError",
    "compute_adapter_fingerprint",
    "detect_site",
    "discover_plugin_adapters",
    "fetch_block",
    "get_site_adapter",
    "parse_html",
    "register_site",
    "registered_sites",
    "OPCAdapter",
    "embed_mre_opc",
    "extract_mre_xml_opc",
    "fetch_opc",
    "get_opc_adapter",
    "parse_opc",
    "MREGenerationResult",
    "generate_mre",
    "parse_hwp",
    "HwpConversionError",
    "LibreOfficeNotAvailableError",
    "convert_hwp",
    "embed_mre_pdf",
    "extract_mre_xml_pdf",
    "fetch_pdf",
    "mre_xml_exists_pdf",
    "parse_pdf",
    "extract_mre_xml",
]
