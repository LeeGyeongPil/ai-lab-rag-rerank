"""CLI index / ask --docs: 없는 폴더·빈 폴더는 트레이스백 없이 한 줄 오류 + 종료 코드 1."""

import pytest

from rag_rerank.cli import main


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))


def _cmds(folder):
    return [["index", str(folder), "--mock"], ["ask", "질문", "--docs", str(folder), "--mock"]]


def test_missing_folder(tmp_path, capsys):
    missing = tmp_path / "nope"
    for argv in _cmds(missing):
        assert main(argv) == 1
        out, err = capsys.readouterr()
        assert f"폴더가 아닙니다: {missing}" in err
        assert "Traceback" not in err
        assert out == ""


def test_empty_folder(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "a.png").write_bytes(b"x")
    for argv in _cmds(empty):
        assert main(argv) == 1
        out, err = capsys.readouterr()
        assert ".md 또는 .txt 문서를 찾지 못했습니다." in err
        assert "Traceback" not in err
        assert out == ""


def test_good_folder_still_ok(tmp_path, capsys):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# 제목\n\n회의는 매주 월요일에 합니다.\n", encoding="utf-8")
    assert main(["index", str(d), "--mock"]) == 0
    assert "문서 1개" in capsys.readouterr().out
