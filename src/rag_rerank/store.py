"""색인 저장소: sqlite-vec(벡터 검색 확장)을 얹은 SQLite 파일 하나.

- chunks 테이블 : 청크 원문
- vec 가상 테이블 : 청크 벡터(vec0, 코사인 거리)
- 같은 문서·청킹 설정·임베딩 모델이면 파일을 재사용한다(.cache/index/<해시>.sqlite).
- 캐시 파일은 최근 16개만 남긴다(무한히 쌓이지 않게). 단, 지금 메모리 색인이 쓰는 파일은 절대 지우지 않는다.
- 캐시 파일이 비었거나 깨졌으면(중간에 지워짐 등) 버리고 다시 만든다.
- SQLite 연결은 쓸 때마다 열고 바로 닫는다(with closing) — 연결이 새는 일이 없다.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from .bm25 import BM25
from .chunking import Chunk, ChunkOptions, Document, chunk_documents

# 디스크에 남길 색인 파일 수. 서버 메모리 색인 상한(server.MAX_INDEXES=6)보다 넉넉해야 한다
# (샘플·내 문서 2 × 청킹 3 × 제목 2 = 12 조합 + 여유). 테스트가 두 값의 관계를 검사한다.
MAX_CACHED_INDEXES = 16


def connect(path: Path | str, readonly: bool = False) -> sqlite3.Connection:
    """sqlite-vec 확장을 불러온 연결. (macOS 기본 파이썬은 확장 로딩이 막혀 있어 uv 파이썬을 쓴다)

    readonly=True: 파일이 없으면 빈 파일을 새로 만들지 않고 바로 오류를 낸다(검색할 때 쓴다).
    """
    if readonly:
        db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    else:
        db = sqlite3.connect(str(path))
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except Exception:
        db.close()  # 확장을 못 불러오면 연결을 닫고 오류를 그대로 올린다(연결이 새지 않게)
        raise
    return db


@dataclass
class Index:
    name: str  # 컬렉션 이름: samples | mine
    chunks: list[Chunk]
    db_path: Path
    embedder_name: str
    options: ChunkOptions
    bm25: BM25  # 키워드 검색용(메모리에 둠, 청크 수천 개까지 충분)

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """질문 벡터와 가장 가까운 청크 k개 → [(청크 번호, 코사인 유사도)]."""
        k = min(k, len(self.chunks))
        if k <= 0:
            return []
        with closing(connect(self.db_path, readonly=True)) as db:  # 읽기 전용: 없는 파일을 새로 만들지 않는다
            rows = db.execute(
                "SELECT id, distance FROM vec WHERE emb MATCH ? AND k = ? ORDER BY distance",
                (np.asarray(query_vec, np.float32).tobytes(), k),
            ).fetchall()
        # 코사인 거리 = 1 - 코사인 유사도. 동점이면 청크 순번으로 순서를 고정한다.
        hits = [(int(i), round(1.0 - float(d), 6)) for i, d in rows]
        return sorted(hits, key=lambda h: (-h[1], h[0]))


def _cache_key(docs: list[Document], opts: ChunkOptions, embedder_name: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([[d.filename, d.text] for d in docs], ensure_ascii=False).encode())
    h.update(f"|{opts.key()}|{embedder_name}".encode())
    return h.hexdigest()[:24]


def _is_valid(path: Path, n_chunks: int) -> bool:
    """캐시 파일을 그대로 써도 되는지: 비어 있지 않고, chunks·vec 표가 있고, 청크 수가 맞아야 한다."""
    try:
        if path.stat().st_size == 0:
            return False
        with closing(connect(path, readonly=True)) as db:
            n = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            m = db.execute("SELECT count(*) FROM vec").fetchone()[0]
        return n == m == n_chunks
    except (OSError, sqlite3.Error):  # 파일 없음·SQLite 파일 아님·표 없음 → 다시 만든다
        return False


def _prune(index_dir: Path, keep: int = MAX_CACHED_INDEXES, in_use: Iterable[Path] = ()) -> None:
    """오래된 색인 파일을 지워 최근 keep 개만 남긴다. in_use(지금 쓰는 파일)는 세지도, 지우지도 않는다."""
    protected = {Path(p).resolve() for p in in_use}
    files = []
    for p in index_dir.glob("*.sqlite"):
        try:
            files.append((p.stat().st_mtime, p))
        except FileNotFoundError:  # 다른 프로세스가 방금 지운 파일
            continue
    files.sort(key=lambda t: t[0], reverse=True)  # 최근에 쓴 것부터
    spare = [p for _, p in files if p.resolve() not in protected]
    n_protected = len(files) - len(spare)  # 이 폴더에 실제로 있는 '쓰는 중' 파일 수
    for old in spare[max(0, keep - n_protected):]:
        old.unlink(missing_ok=True)


async def build_index(name: str, docs: list[Document], opts: ChunkOptions, embedder, cache_dir: Path,
                      in_use: Iterable[Path] = (), keep: int = MAX_CACHED_INDEXES) -> Index:
    """문서 → 청크 → 임베딩 → SQLite 파일. 같은 조건의 멀쩡한 파일이 있으면 임베딩을 건너뛴다.

    in_use: 호출한 쪽이 아직 쓰고 있는 색인 파일들(정리할 때 지우지 않는다).
    """
    chunks = chunk_documents(docs, opts)
    index_dir = Path(cache_dir) / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / f"{_cache_key(docs, opts, embedder.name)}.sqlite"

    if _is_valid(path, len(chunks)):
        os.utime(path)  # 최근 사용 표시(정리할 때 오래된 것부터 지운다)
    elif chunks:
        path.unlink(missing_ok=True)  # 빈 파일·깨진 파일이 있으면 버린다
        vectors = await embedder.embed([c.content for c in chunks])
        tmp = path.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        with closing(connect(tmp)) as db:
            db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, chunk_id TEXT, content TEXT)")
            db.execute(f"CREATE VIRTUAL TABLE vec USING vec0(id INTEGER PRIMARY KEY, emb float[{vectors.shape[1]}] distance_metric=cosine)")
            db.executemany("INSERT INTO chunks VALUES(?,?,?)", [(i, c.chunk_id, c.content) for i, c in enumerate(chunks)])
            db.executemany("INSERT INTO vec(id, emb) VALUES(?,?)", [(i, v.tobytes()) for i, v in enumerate(vectors)])
            db.commit()
        tmp.replace(path)  # 다 쓴 뒤에 이름을 바꿔, 중간에 끊겨도 깨진 캐시가 남지 않게 한다
        _prune(index_dir, keep, in_use=[*in_use, path])  # 방금 만든 파일과 쓰는 중인 파일은 남긴다
    return Index(name, chunks, path, embedder.name, opts, BM25([c.content for c in chunks]))


def load_folder(folder: Path, max_files: int = 200, max_bytes: int = 1_000_000) -> list[Document]:
    """폴더 안의 .md/.txt 파일을 읽는다(하위 폴더 포함, 파일 수·크기 상한)."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise ValueError(f"폴더가 아닙니다: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".md", ".txt") and p.is_file())
    docs = []
    for p in files[:max_files]:
        if p.stat().st_size > max_bytes:
            continue
        rel = p.relative_to(folder).as_posix()
        docs.append(Document(rel.rsplit(".", 1)[0], rel, p.read_text(encoding="utf-8", errors="replace")))
    return docs
