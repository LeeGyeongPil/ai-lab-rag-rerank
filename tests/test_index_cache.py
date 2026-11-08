"""색인 캐시 파일(.cache/index/*.sqlite) 회귀 테스트.

예전 버그: 메모리에는 색인 6개를 두는데 디스크 파일은 5개만 남겨서, 아직 쓰고 있는 파일이 지워졌다.
그 뒤 벡터 검색이 빈 파일(0바이트)을 새로 만들어 'no such table: vec' 로 계속 실패했다(재시작해도).
"""

import asyncio
import itertools
import sqlite3

import pytest
from fastapi.testclient import TestClient
from schema_check import assert_compare_schema

from rag_rerank import server, store
from rag_rerank.chunking import ChunkOptions
from rag_rerank.server import create_app
from rag_rerank.store import build_index, load_folder

BASE = "http://127.0.0.1"  # 서버는 127.0.0.1/localhost 주소로 온 요청만 받는다
QUESTION = "책을 늦게 돌려주면 벌금을 내나요?"
COMBOS = list(itertools.product(["paragraph", "recursive", "heading"], [True, False]))  # 청킹 3 × 제목 2 = 6


def _compare(client, chunking="paragraph", title_prefix=True, retriever="hybrid"):
    body = {"question": QUESTION, "retriever": retriever, "top_k": 8,
            "chunking": chunking, "title_prefix": title_prefix}
    return client.post("/api/compare", json=body)


def test_all_combinations_then_default_still_works_after_restart():
    """6가지 청킹×제목 조합을 다 돌린 뒤에도 기본 비교가 되고, 서버를 다시 켜도 된다."""
    with TestClient(create_app(), base_url=BASE) as client:
        for chunking, title in COMBOS:
            r = _compare(client, chunking, title, "vector")
            assert r.status_code == 200, (chunking, title, r.text)
        r = _compare(client)  # 기본 비교(문단 + 제목 붙임, 하이브리드)
        assert r.status_code == 200, r.text
        assert_compare_schema(r.json())

    # 같은 캐시 폴더로 서버를 다시 켠다(앱 재시작)
    with TestClient(create_app(), base_url=BASE) as client:
        for chunking, title in COMBOS:
            assert _compare(client, chunking, title, "vector").status_code == 200
        assert _compare(client).status_code == 200


def test_disk_cap_covers_memory_cap():
    # 디스크에 남기는 파일 수가 메모리 색인 수보다 적으면, 쓰는 중인 파일이 지워질 수 있다
    assert store.MAX_CACHED_INDEXES >= server.MAX_INDEXES


def test_prune_never_deletes_files_in_use(engine):
    """디스크 상한이 아주 작아도, 메모리 색인이 쓰는 파일(in_use)은 지우지 않는다."""
    docs = load_folder(engine.settings.samples_dir)
    cache = engine.settings.cache_dir

    async def scenario():
        first = await build_index("samples", docs, ChunkOptions("paragraph", True), engine.embedder, cache)
        in_use = {first.db_path}
        for chunking, title in COMBOS[1:]:
            idx = await build_index("samples", docs, ChunkOptions(chunking, title), engine.embedder, cache,
                                    in_use=in_use, keep=2)
            in_use.add(idx.db_path)
        return first

    first = asyncio.run(scenario())
    assert first.db_path.exists() and first.db_path.stat().st_size > 0
    q = asyncio.run(engine.embedder.embed([QUESTION]))[0]
    assert first.vector_search(q, 3)  # 지워지지 않았으니 검색이 된다
    # 쓰지 않는 파일은 상한(2개)에 맞춰 정리된다 → 무한히 쌓이지 않는다
    files = list((cache / "index").glob("*.sqlite"))
    assert len(files) == len(COMBOS)  # 6개 모두 in_use 라 남아 있음


def test_prune_respects_cap_when_nothing_in_use(engine):
    docs = load_folder(engine.settings.samples_dir)
    cache = engine.settings.cache_dir

    async def scenario():
        for chunking, title in COMBOS:
            await build_index("samples", docs, ChunkOptions(chunking, title), engine.embedder, cache, keep=2)

    asyncio.run(scenario())
    assert len(list((cache / "index").glob("*.sqlite"))) == 2


@pytest.mark.parametrize("damage", ["empty", "garbage", "no_vec_table"])
def test_broken_cache_file_is_rebuilt(engine, damage):
    """빈 파일·깨진 파일·vec 표가 없는 파일이 캐시 자리에 있으면 버리고 다시 만든다."""
    docs = load_folder(engine.settings.samples_dir)
    opts = ChunkOptions("paragraph", True)
    cache = engine.settings.cache_dir
    good = asyncio.run(build_index("samples", docs, opts, engine.embedder, cache))
    path = good.db_path
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "garbage":
        path.write_bytes(b"not a sqlite file" * 10)
    else:
        path.unlink()
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, chunk_id TEXT, content TEXT)")
        db.close()

    rebuilt = asyncio.run(build_index("samples", docs, opts, engine.embedder, cache))
    q = asyncio.run(engine.embedder.embed([QUESTION]))[0]
    assert len(rebuilt.vector_search(q, 3)) == 3


def test_missing_file_raises_instead_of_creating_empty_one(engine, make_index):
    """파일이 사라졌을 때 검색이 빈 파일을 새로 만들지 않는다(읽기 전용으로 연다)."""
    index = make_index()
    index.db_path.unlink()
    q = asyncio.run(engine.embedder.embed([QUESTION]))[0]
    with pytest.raises(sqlite3.OperationalError):
        index.vector_search(q, 3)
    assert not index.db_path.exists()


def test_server_rebuilds_when_file_disappears():
    """메모리에 있는 색인의 파일이 밖에서 지워져도(다른 프로세스·사용자가 지움) 서버가 다시 만든다."""
    with TestClient(create_app(), base_url=BASE) as client:
        assert _compare(client).status_code == 200
        for idx in client.app.state.rag.indexes.values():
            idx.db_path.unlink()
        r = _compare(client)
        assert r.status_code == 200, r.text
