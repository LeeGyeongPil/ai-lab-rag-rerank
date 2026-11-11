import asyncio
import json
import socket

import pytest

from rag_rerank.chunking import ChunkOptions
from rag_rerank.config import ModeInfo, Settings, resolve_mode
from rag_rerank.pipeline import compare, create_engine
from rag_rerank.store import build_index, load_folder


def _run_once(cache_dir) -> str:
    settings = Settings.from_env()
    engine = create_engine(settings, ModeInfo("mock", "test"))
    docs = load_folder(settings.samples_dir)
    index = asyncio.run(build_index("samples", docs, ChunkOptions(), engine.embedder, cache_dir))
    r = asyncio.run(compare(engine, index, "중학생도 공유자전거를 탈 수 있나요?", "hybrid", 8))
    return json.dumps(r, ensure_ascii=False, sort_keys=True)


def test_mock_is_deterministic_and_offline(tmp_path):
    """AC-2: 네트워크를 막은 상태에서 compare() 두 번 → JSON 이 바이트 단위로 같다."""
    with pytest.raises(Exception):  # 정말 막혀 있는지 먼저 확인
        socket.create_connection(("127.0.0.1", 11434), timeout=0.1)
    first = _run_once(tmp_path / "a")
    second = _run_once(tmp_path / "b")  # 캐시도 새로(임베딩부터 다시)
    third = _run_once(tmp_path / "a")  # 캐시 재사용
    assert first.encode() == second.encode() == third.encode()
    assert '"mode": "mock"' in first


def test_auto_falls_back_to_mock(monkeypatch):
    """auto 모드에서 Ollama 에 못 붙으면 mock 으로 내려가고 이유·설치 명령을 준다."""
    monkeypatch.setenv("RAG_MODE", "auto")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    info = resolve_mode(Settings.from_env())
    assert info.mode == "mock"
    assert "Could not connect to Ollama" in info.reason
    assert "연결하지 못했습니다" in info.render("ko")[0]  # 같은 이유를 한국어로도 만들 수 있다
    assert any("ollama pull bge-m3" in h for h in info.hints)


def test_forced_mock_does_not_probe(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "auto")
    assert resolve_mode(Settings.from_env(), force_mock=True).mode == "mock"
