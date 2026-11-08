import asyncio

import pytest

from rag_rerank.retrieve import HYBRID_POOL, hybrid, keyword, rrf, vector


def test_rrf_hand_computed():
    """AC-6: k=60, 목록1=[A,B,C], 목록2=[C,A,D] 를 손으로 계산한 값과 같다."""
    A, B, C, D = 0, 1, 2, 3
    fused = rrf([[A, B, C], [C, A, D]], k=60)
    expected = {
        A: 1 / 61 + 1 / 62,  # 1위 + 2위 = 0.032522
        C: 1 / 63 + 1 / 61,  # 3위 + 1위 = 0.032266
        B: 1 / 62,           # 0.016129
        D: 1 / 63,           # 0.015873
    }
    assert [i for i, _ in fused] == [A, C, B, D]
    for i, score in fused:
        assert score == pytest.approx(expected[i], abs=1e-6)


def test_hybrid_is_fusion_of_both(engine, make_index):
    index = make_index()
    q = "중학생도 공유자전거를 탈 수 있나요?"
    qv = asyncio.run(engine.embedder.embed([q]))[0]
    kw = {i for i, _ in keyword(index, q, HYBRID_POOL)}
    vec = {i for i, _ in vector(index, qv, HYBRID_POOL)}
    hy = hybrid(index, q, qv, 8)
    assert len(hy) == 8
    assert {i for i, _ in hy} <= kw | vec
    expected = rrf([[i for i, _ in keyword(index, q, HYBRID_POOL)], [i for i, _ in vector(index, qv, HYBRID_POOL)]])[:8]
    assert hy == expected


def test_vector_scores_are_cosine(engine, make_index):
    index = make_index()
    qv = asyncio.run(engine.embedder.embed(["도서관 운영 시간"]))[0]
    hits = vector(index, qv, 5)
    assert len(hits) == 5
    assert all(-1.0 <= s <= 1.0 for _, s in hits)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
