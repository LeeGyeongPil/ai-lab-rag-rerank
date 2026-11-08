"""한국어 키워드 검색: 한글 2글자 단위 토큰 + BM25. 외부 의존성 없음.

왜 2글자 단위인가?
- 한국어는 조사가 붙어서('반납하면', '반납한') 띄어쓰기 단위로는 같은 말이 잘 안 맞는다.
- 2글자씩 잘라 두면('반납', '납하', '하면') '반납'이 들어간 모든 형태가 서로 맞는다.
- SQLite FTS5 의 trigram(3글자) 토크나이저는 '연체' 같은 2글자 검색어를 아예 못 찾는다.
"""

from __future__ import annotations

import math
import re
from collections import Counter

WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def grams(text: str) -> list[str]:
    """'반납하면 됩니다' → ['반납', '납하', '하면', '됩니', '니다'] (1글자 단어는 그대로 둔다)."""
    out: list[str] = []
    for w in WORD.findall(text.lower()):
        out += [w] if len(w) < 2 else [w[i : i + 2] for i in range(len(w) - 1)]
    return out


class BM25:
    """고전 BM25(k1=1.5, b=0.75). 문서 길이를 보정한 '드문 단어가 많이 겹칠수록 높은 점수'."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(grams(d)) for d in docs]  # 문서별 토큰 빈도
        self.lengths = [sum(t.values()) for t in self.tf]
        self.n_docs = len(docs)
        self.avg_len = (sum(self.lengths) / self.n_docs) if self.n_docs else 0.0
        # df: 그 토큰이 나온 문서 수. 적게 나올수록(드물수록) idf 가 커진다.
        self.df = Counter(g for t in self.tf for g in t)
        self.idf = {g: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for g, n in self.df.items()}

    def scores(self, query: str) -> list[float]:
        q = grams(query)
        out = []
        for tf, length in zip(self.tf, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            out.append(sum(self.idf[g] * tf[g] * (self.k1 + 1) / (tf[g] + norm) for g in q if g in tf))
        return out
