# Changelog

All notable changes to `py-mre` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

See [Specification § Root element attributes](https://machine-readable-extension.github.io/py-mre/spec/#root-element-attributes)
for how the MRE *schema* version (`<mre version="1.0">`) relates to this
library's own version below — they're independent axes, not the same number.

## [Unreleased]

### Added

- "Adding your site" promoted from a `formats.md` subsection to its own
  top-level docs page, for discoverability by site owners.
- `<mre>` root element attributes (`version`, `generator`,
  `generator-fingerprint`) documented in the specification.

### Changed

- All remaining Korean code comments and private/internal docstrings
  translated to English.
- Public API docstrings (already translated) touched up for style
  consistency.

### Fixed

- `mkdocstrings` `docstring_style` corrected from `google` to `numpy`
  (all docstrings use NumPy-style `Parameters`/`Returns` sections; the
  wrong style setting was rendering them as unformatted text on the API
  reference page).

## [1.1.1] - 2026-08-26

### Fixed

- Crash on real Wikipedia pages whose appendix sections
  (`References`/`Bibliography`/etc.) contain nested `<section>` tags.
- Stale `Machine-Readable-Extension/mre` URLs left over from the repo
  rename to `py-mre`, in `pyproject.toml`'s `[project.urls]` and
  `mkdocs.yml`'s `repo_url`/`repo_name`.

### Added

- README and docs section inviting parsing-only community site adapters
  for sites that haven't adopted MRE headers themselves.

## [1.1.0] - 2026-08-25

First published release (split out of the `Machine-Readable-Extension`
monorepo's in-tree `mre/` package into this standalone library).

### Changed

- PyPI distribution renamed to `py-mre` (`mre` was already taken by an
  unrelated package). Import name is unaffected: `import mre` either way.
- Author email switched from a personal address to the GitHub noreply
  address already used for commits.
