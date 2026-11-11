# Changelog

## 0.2.0 - 2026-11-12

### Added
- English and Korean screen language. CLI: `--lang en|ko` (anywhere on the command line) > `RAG_LANG` > `LC_ALL` > `LC_MESSAGES` > `LANG`; anything that is not Korean is English. Web UI: an `English / 한국어` menu in the header, stored in `localStorage` (key `rag-rerank-lang`), decided from the browser language only on the first visit; `?lang=en|ko` also works.
- `src/rag_rerank/i18n.py` holds all CLI/server messages in one table per language; `static/i18n.js` holds the web table. Tests check that both languages have the same keys and placeholders.
- Library errors carry a `code` and `params` (`RagError`) and are rendered by the CLI or the server in the chosen language. API error `detail` texts follow `?lang=` or `Accept-Language`.

### Changed
- The default language now follows the environment, so human-readable CLI output is English unless the locale (or `--lang` / `RAG_LANG`) says Korean. Machine-readable output is unchanged: `--json` output, `/api/*` JSON keys and shapes, exit codes, cache files.
- Error and notice texts are English by default (`폴더가 아닙니다: <path>` is now `Not a folder: <path>`; use `--lang ko` for the old text).

### Fixed
- `index <folder>` / `ask --docs <folder>` no longer end with a `PermissionError` traceback when a document is unreadable (or the cache folder is not writable); they print one line and exit with code 1.

## 0.1.2 - 2026-11-10

### Fixed
- CLI `index <folder>` and `ask --docs <folder>` no longer print a traceback for a missing folder. They print `폴더가 아닙니다: <경로>` to stderr and exit with code 1.
- A folder with no `.md`/`.txt` files is now rejected with `.md 또는 .txt 문서를 찾지 못했습니다.` (exit 1), matching the web API.

### Changed
- Package version metadata now matches the release tag.

## 0.1.1 - 2026-11-08

### Added
- Practice (mock) mode can show the rerank lesson without a model.

## 0.1.0 - 2026-11-08

- First release.
