# Install

```bash
pip install py-mre
```

The PyPI distribution is named `py-mre` (`mre` itself was already taken by an
unrelated project, and `py-` is a common PyPI convention for this situation —
`py-cpuinfo`, `py-spy`, ...) — the import name is unaffected: `import mre`
either way, the same split `beautifulsoup4`/`bs4` or `pyyaml`/`yaml` use.

The keyword-grounding repair pass (see [Repair](quickstart.md#repair)) has an
optional fuzzy-matching fallback:

```bash
pip install "py-mre[fuzzy]"
```

To install from source instead (for development, or to track `master`):

```bash
git clone https://github.com/Machine-Readable-Extension/py-mre.git
cd mre
pip install -e .
```
