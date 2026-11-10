# Changelog

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
