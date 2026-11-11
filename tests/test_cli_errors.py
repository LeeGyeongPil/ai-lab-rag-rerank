"""CLI 오류 경로: 트레이스백 없이 화면 언어(en/ko)로 한 줄 + 종료 코드 1.

폴더 없음·문서 없음·권한 없는 파일·쓸 수 없는 캐시 폴더를 모두 두 언어로 확인한다.
하위 프로세스 테스트는 stderr '줄 수'가 아니라 'Traceback 이 없고 우리 오류 문구가 들어 있다'로 검사한다(import 경고 줄 대비).
"""

import os
import subprocess
import sys

import pytest

from rag_rerank.cli import main


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))


def _cmds(folder, lang):
    return [["index", str(folder), "--mock", "--lang", lang], ["ask", "q", "--docs", str(folder), "--mock", "--lang", lang]]


MISSING = {"en": "Not a folder: {path}", "ko": "폴더가 아닙니다: {path}"}
NO_DOCS = {"en": "No .md or .txt documents were found.", "ko": ".md 또는 .txt 문서를 찾지 못했습니다."}
UNREADABLE = {"en": "Cannot access {path}", "ko": "{path}에 접근할 수 없습니다"}


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_missing_folder(tmp_path, capsys, lang):
    missing = tmp_path / "nope"
    for argv in _cmds(missing, lang):
        assert main(argv) == 1
        out, err = capsys.readouterr()
        assert MISSING[lang].format(path=missing) in err
        assert "Traceback" not in err
        assert out == ""


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_empty_folder(tmp_path, capsys, lang):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "a.png").write_bytes(b"x")
    for argv in _cmds(empty, lang):
        assert main(argv) == 1
        out, err = capsys.readouterr()
        assert NO_DOCS[lang] in err
        assert "Traceback" not in err
        assert out == ""


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_unreadable_file_is_one_line_error(tmp_path, capsys, monkeypatch, lang):
    """읽기 권한이 없는 문서(PermissionError)가 있어도 트레이스백 없이 한 줄 오류 + 종료 코드 1."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# t\n\nhello\n", encoding="utf-8")
    from pathlib import Path

    real = Path.read_text

    def deny(self, *args, **kwargs):  # root 로 돌려도 확실히 실패하게 직접 막는다
        if self.name == "a.md":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)
    for argv in _cmds(d, lang):
        assert main(argv) == 1
        out, err = capsys.readouterr()
        assert UNREADABLE[lang].format(path=d / "a.md") in err
        assert "Permission denied" in err
        assert "Traceback" not in err
        assert out == ""


def test_real_chmod_000_file(tmp_path):
    """진짜 chmod 000 파일로 하위 프로세스를 돌려 본다(권한이 실제로 막히는 환경에서만)."""
    d = tmp_path / "docs"
    d.mkdir()
    locked = d / "a.md"
    locked.write_text("# t\n\nhello\n", encoding="utf-8")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("this account can read chmod 000 files (e.g. root)")
        for lang in ("en", "ko"):
            proc = _run(["index", str(d), "--mock"], lang, tmp_path)
            assert proc.returncode == 1
            assert "Traceback" not in proc.stderr
            assert UNREADABLE[lang].format(path=locked) in proc.stderr
    finally:
        locked.chmod(0o644)


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_unwritable_cache_dir_is_one_line_error(tmp_path, capsys, monkeypatch, lang):
    """색인을 쓰는 중 PermissionError(캐시 폴더 권한 없음)도 트레이스백 없이 로캘 한 줄 오류."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# t\n\nhello\n", encoding="utf-8")
    from pathlib import Path

    real = Path.mkdir

    def deny(self, *args, **kwargs):
        if self.name == "index":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny)
    assert main(["index", str(d), "--mock", "--lang", lang]) == 1
    out, err = capsys.readouterr()
    assert "Permission denied" in err  # 어느 경로가 왜 안 되는지 사유를 그대로 알려 준다
    assert ("Cannot access " if lang == "en" else "에 접근할 수 없습니다") in err  # not "read": the failing step was a write
    assert "Traceback" not in err
    assert out == ""


def _run(argv, lang, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("LC_ALL", "LC_MESSAGES", "LANG")}
    env.update(RAG_LANG=lang, RAG_CACHE_DIR=str(tmp_path / "cache"), RAG_MODE="mock")
    return subprocess.run([sys.executable, "-m", "rag_rerank", *argv], capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_subprocess_missing_folder(tmp_path, lang):
    """실제 프로세스로: Traceback 이 없고 문구가 언어에 맞다(다른 경고 줄이 섞여도 통과)."""
    missing = tmp_path / "nope"
    proc = _run(["index", str(missing)], lang, tmp_path)
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert MISSING[lang].format(path=missing) in proc.stderr


def test_good_folder_still_ok(tmp_path, capsys):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# 제목\n\n회의는 매주 월요일에 합니다.\n", encoding="utf-8")
    assert main(["index", str(d), "--mock"]) == 0
    assert "1 documents" in capsys.readouterr().out
    assert main(["index", str(d), "--mock", "--lang", "ko"]) == 0
    assert "문서 1개" in capsys.readouterr().out
