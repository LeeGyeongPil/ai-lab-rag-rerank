"""테스트 공통: mock 모드 고정 + 네트워크 차단.

real 표시(@pytest.mark.real)가 없는 테스트는 모두
- RAG_MODE=mock, 캐시는 임시 폴더
- 인터넷 소켓 연결을 막는다(연결을 시도하면 바로 실패) → mock 이 정말 네트워크를 안 쓰는지 검증
"""

import asyncio
import socket

import pytest

from rag_rerank.chunking import ChunkOptions
from rag_rerank.config import ModeInfo, Settings
from rag_rerank.pipeline import create_engine
from rag_rerank.store import build_index, load_folder


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def mock_and_offline(request, monkeypatch, tmp_path):
    if request.node.get_closest_marker("real"):
        yield
        return
    monkeypatch.setenv("RAG_MODE", "mock")
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAG_NO_BROWSER", "1")

    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(f"테스트 중 네트워크 연결 시도: {address}")
        return real_connect(self, address)

    def blocked(*args, **kwargs):
        raise NetworkBlocked("테스트 중 네트워크 연결 시도")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield


@pytest.fixture
def engine():
    settings = Settings.from_env()
    return create_engine(settings, ModeInfo("mock", "test"))


@pytest.fixture
def make_index(engine):
    """make_index(strategy='paragraph', title_prefix=True) → 샘플 문서 색인."""

    def _make(strategy="paragraph", title_prefix=True):
        docs = load_folder(engine.settings.samples_dir)
        return asyncio.run(build_index("samples", docs, ChunkOptions(strategy, title_prefix),
                                       engine.embedder, engine.settings.cache_dir))

    return _make
