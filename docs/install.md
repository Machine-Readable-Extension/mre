# Install

```bash
pip install machine-readable-extension
```

The PyPI distribution is named `machine-readable-extension` (`mre` itself was
already taken by an unrelated project) — the import name is unaffected:
`import mre` either way, the same split `beautifulsoup4`/`bs4` or
`pyyaml`/`yaml` use.

The keyword-grounding repair pass (see [Repair](quickstart.md#repair)) has an
optional fuzzy-matching fallback:

```bash
pip install "machine-readable-extension[fuzzy]"
```

To install from source instead (for development, or to track `master`):

```bash
git clone https://github.com/Machine-Readable-Extension/mre.git
cd mre
pip install -e .
```
