"""1단계 검색기 세 가지: 벡터 / 키워드 / 하이브리드(RRF).

1단계는 '빠르지만 거친' 검색이다. 여기서 뽑힌 후보 top-k 안에서만 리랭커가 순서를 고친다.
그래서 1단계가 정답을 후보에 못 올리면 리랭커도 살릴 수 없다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .i18n import RagError
from .store import Index

RETRIEVERS = ("vector", "keyword", "hybrid")
RRF_K = 60  # RRF 관례값. 클수록 '1위와 10위의 차이'를 덜 크게 본다.
HYBRID_POOL = 20  # 하이브리드는 두 검색기에서 각각 20개씩 받아 섞는다

# 열마다 점수의 뜻(척도)이 다르다 — 같은 막대 길이로 비교하면 안 된다는 표시
SCALES = {
    "vector": "코사인 유사도(-1~1)",
    "keyword": "BM25 점수(상한 없음)",
    "hybrid": "RRF 점수(Σ 1/(60+순위))",
}


def keyword(index: Index, query: str, k: int) -> list[tuple[int, float]]:
    """BM25 점수가 0보다 큰(겹치는 토큰이 있는) 청크만 높은 순으로."""
    scores = index.bm25.scores(query)
    hits = [(i, round(s, 6)) for i, s in enumerate(scores) if s > 0]
    return sorted(hits, key=lambda h: (-h[1], h[0]))[:k]


def vector(index: Index, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    return index.vector_search(query_vec, k)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: 점수는 버리고 '순위'만으로 합친다.

    점수(i) = Σ_목록 1 / (k + i의 순위)   (순위는 1부터)
    척도가 다른 BM25 점수와 코사인 점수를 그대로 더할 수 없어서 순위를 쓴다.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return sorted(((i, round(s, 6)) for i, s in fused.items()), key=lambda h: (-h[1], h[0]))


def hybrid(index: Index, query: str, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    pool = max(HYBRID_POOL, k)
    kw = [i for i, _ in keyword(index, query, pool)]
    vec = [i for i, _ in vector(index, query_vec, pool)]
    return rrf([kw, vec])[:k]


def search(index: Index, retriever: str, query: str, query_vec: np.ndarray | None, k: int) -> list[tuple[int, float]]:
    if retriever == "keyword":
        return keyword(index, query, k)
    if retriever == "vector":
        return vector(index, query_vec, k)
    if retriever == "hybrid":
        return hybrid(index, query, query_vec, k)
    raise RagError("unknown_retriever", name=retriever)
