import sqlite3
from contextlib import closing

import pytest

from rag_rerank.bm25 import BM25, grams
from rag_rerank.retrieve import keyword


def test_grams_two_syllables():
    assert grams("반납하면 돼요") == ["반납", "납하", "하면", "돼요"]
    assert grams("Q 코드") == ["q", "코드"]  # 1글자 단어는 그대로


def test_two_syllable_korean_term(make_index):
    """AC-5: 2글자 검색어 '연체'로 연체 문단이 검색된다(FTS5 trigram 함정 회귀 방지)."""
    index = make_index("paragraph", False)
    hits = keyword(index, "연체", 8)
    assert len(hits) >= 1
    assert all("연체" in index.chunks[i].text for i, _ in hits)
    assert "반납 예정일을 넘기면 연체가 시작됩니다" in index.chunks[hits[0][0]].text or len(hits) >= 3


def test_fts5_trigram_misses_two_syllables(make_index):
    """왜 BM25 를 직접 구현했나: SQLite FTS5 trigram 은 2글자 검색어를 못 찾는다."""
    index = make_index("paragraph", False)
    with closing(sqlite3.connect(":memory:")) as db:
        try:
            db.execute("CREATE VIRTUAL TABLE f USING fts5(t, tokenize='trigram')")
        except sqlite3.OperationalError:
            pytest.skip("이 SQLite 에는 FTS5 trigram 이 없음")
        db.executemany("INSERT INTO f(t) VALUES(?)", [(c.text,) for c in index.chunks])
        two = db.execute("SELECT count(*) FROM f WHERE f MATCH '\"연체\"'").fetchone()[0]
        three = db.execute("SELECT count(*) FROM f WHERE f MATCH '\"연체료\"'").fetchone()[0]
    assert two == 0 and three >= 1
    assert len(keyword(index, "연체", 8)) > 0  # 우리 BM25 는 찾는다


def test_bm25_prefers_rare_terms():
    bm = BM25(["도서관 운영 시간", "도서관 연체 규정", "도서관 시설"])
    scores = bm.scores("연체")
    assert scores.index(max(scores)) == 1 and scores[0] == 0
