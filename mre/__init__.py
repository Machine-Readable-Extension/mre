__version__ = "1.0.0"

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
    fetch_opc,
    get_opc_adapter,
    parse_opc,
)
from mre.generate import MREGenerationResult, generate_mre
from mre.reader import extract_mre_xml

# Built-in adapters (Wikipedia, etc.) are already registered by the time
# mre.html_site_adapter is loaded. Plugin discovery must run here, after
# every other export of the mre package is already bound, so that a plugin
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
    "fetch_opc",
    "get_opc_adapter",
    "parse_opc",
    "MREGenerationResult",
    "generate_mre",
    "extract_mre_xml",
]
