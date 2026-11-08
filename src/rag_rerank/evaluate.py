"""성적표: 정답이 달린 질문 세트로 리랭킹 전/후를 숫자로 비교한다.

- hit@1 : 정답 청크가 1위인 질문 수
- MRR   : 정답 순위의 역수 평균(1위=1, 2위=0.5, 3위=0.33 …, 후보 밖=0). 1에 가까울수록 좋다.

질문 파일 형식(samples/questions.json):
  [{"q": "질문", "gold": "정답 청크에 들어 있는 문자열", "note": "(선택) 설명"}]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from . import retrieve
from .store import Index


def load_questions(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _explain(before: int | None, after: int | None) -> str:
    if before is None:
        return "1단계 후보 밖 — 리랭커가 살릴 수 없음"
    if after < before:
        return f"{before}위 → {after}위로 올라옴"
    if after > before:
        return f"{before}위 → {after}위로 내려감"
    return f"{before}위 그대로"


async def evaluate(engine, index: Index, questions: list[dict], top_k: int = 8,
                   retrievers: tuple[str, ...] = ("keyword", "vector", "hybrid")) -> dict:
    reranker, state = engine.reranker_for(index)
    if reranker is None:
        raise RuntimeError(f"리랭커가 아직 준비되지 않았습니다(상태: {state}).")

    # 질문마다 정답 청크(들)의 번호. gold 문자열을 품은 청크가 여러 개면 그중 가장 높은 순위를 쓴다.
    gold = [{i for i, c in enumerate(index.chunks) if q["gold"] in c.text} for q in questions]
    vectors = await engine.embedder.embed([q["q"] for q in questions])  # 질문 임베딩은 한 번에
    score_cache: dict[tuple[int, int], float] = {}  # (질문, 청크) 리랭커 점수 — 검색기 3종이 같은 쌍을 다시 쓰지 않게

    def rank_of(order: list[int], golds: set[int]) -> int | None:
        return next((r for r, i in enumerate(order, start=1) if i in golds), None)

    table, rows, changed = {}, [{"q": q["q"], "gold": q["gold"], "ranks": {}, "explain": {}} for q in questions], 0
    for qi, q in enumerate(questions):
        if q.get("note"):
            rows[qi]["note"] = q["note"]
    for name in retrievers:
        h1b = h1a = 0
        mrr_b = mrr_a = 0.0
        moved = 0
        for qi, q in enumerate(questions):
            cand = [i for i, _ in retrieve.search(index, name, q["q"], vectors[qi], top_k)]
            todo = [i for i in cand if (qi, i) not in score_cache]
            if todo:
                for i, s in zip(todo, await asyncio.to_thread(reranker.score, q["q"], [index.chunks[i].content for i in todo])):
                    score_cache[(qi, i)] = s
            after = sorted(cand, key=lambda i: (-score_cache[(qi, i)], cand.index(i)))
            b, a = rank_of(cand, gold[qi]), rank_of(after, gold[qi])
            moved += cand != after
            h1b += b == 1
            h1a += a == 1
            mrr_b += 1 / b if b else 0
            mrr_a += 1 / a if a else 0
            rows[qi]["ranks"][name] = [b, a]
            rows[qi]["explain"][name] = _explain(b, a)
        n = len(questions) or 1
        table[name] = {"hit1_before": h1b, "hit1_after": h1a,
                       "mrr_before": round(mrr_b / n, 4), "mrr_after": round(mrr_a / n, 4), "orders_changed": moved}
        changed += moved
    return {
        "mode": engine.mode,
        "reranker": reranker.name,
        "chunking": {"strategy": index.options.strategy, "title_prefix": index.options.title_prefix},
        "top_k": top_k,
        "n_questions": len(questions),
        "table": table,
        "orders_changed": changed,  # 리랭킹으로 순서가 바뀐 (질문 × 검색기) 조합 수
        "questions": rows,
    }
