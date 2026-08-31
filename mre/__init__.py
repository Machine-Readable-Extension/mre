__version__ = "1.1.1"

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

# Built-in adapters (Wikipedia, etc.) are already registered by the time
# mre.html_site_adapter is imported above. Plugin discovery has to run here,
# after every other export of this package is bound, so that a plugin
# importing `from mre import HTMLSiteAdapter` (the documented convention)
# doesn't hit a circular import.
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
