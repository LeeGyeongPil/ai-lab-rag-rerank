"""파이프라인의 심장: 같은 질문으로 '리랭킹 전'과 '리랭킹 후'를 만들어 나란히 담는다.

처음 읽는다면 이 파일 → retrieve.py → reranker.py 순서를 추천한다.

    질문 ─┬─ 임베딩 ─ 1단계 검색(벡터/키워드/하이브리드) ─ 후보 top-k ─┬─ "리랭킹 전" (1단계 점수 순)
          │                                                         └─ 리랭커 ─ "리랭킹 후" (리랭커 점수 순)
          └─ 답변 A = 전 상위 3개 근거 / 답변 B = 후 상위 3개 근거
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from . import retrieve
from .config import ModeInfo, Settings
from .embedder import MockEmbedder, OllamaEmbedder
from .generator import NO_EVIDENCE, MockGenerator, OllamaGenerator
from .highlight import best_sentence, overlap_spans, split_sentences
from .reranker import MockReranker, RerankerHub
from .store import Index

ANSWER_CACHE_MAX = 64  # 같은 (질문, 근거) 답변은 최근 64개까지만 기억한다(무한 축적 방지)
CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Engine:
    """모드에 맞는 임베더·리랭커·생성기 묶음. 서버 하나(또는 CLI 실행 한 번)에 하나만 만든다."""

    settings: Settings
    mode_info: ModeInfo
    embedder: object
    generator: object
    hub: RerankerHub | None = None  # 실제 모드에서만
    client: httpx.AsyncClient | None = None  # Ollama 와 말하는 HTTP 클라이언트(하나를 재사용)
    answer_cache: OrderedDict = field(default_factory=OrderedDict)

    @property
    def mode(self) -> str:
        return self.mode_info.mode

    def reranker_for(self, index: Index):
        """(리랭커, 상태). 실제 모드에서 모델을 받는 중이면 (None, 'downloading')."""
        if self.mode == "mock":
            return MockReranker(index.bm25.df, index.bm25.n_docs), "ready"
        model = self.hub.get()
        return (model, "ready") if model else (None, self.hub.state)

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.hub is not None:
            self.hub.close()
        self.answer_cache.clear()


def create_engine(settings: Settings, mode_info: ModeInfo) -> Engine:
    if mode_info.mode == "mock":
        return Engine(settings, mode_info, MockEmbedder(), MockGenerator())
    client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=httpx.Timeout(30.0, read=120.0))
    return Engine(
        settings, mode_info,
        embedder=OllamaEmbedder(client, settings.embed_model),
        generator=OllamaGenerator(client, settings.chat_model),
        hub=RerankerHub(settings),
        client=client,
    )


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def top2_ratio(scores: list[float]) -> float | None:
    """'2위 점수 ÷ 1위 점수'(0~1). 1위가 나머지와 얼마나 떨어져 있는지 보는 잣대.

    1에 가까우면 1·2위가 거의 붙어 있고(누가 정답인지 확신 없음), 0에 가까우면 1위가 확 앞선다.
    비율이라 척도가 달라도(RRF 0.03 안팎, 리랭커 0~1) 두 열을 같은 잣대로 비교할 수 있다.
    """
    top = sorted(scores, reverse=True)
    if len(top) < 2 or top[0] <= 0:
        return None
    return round(max(top[1], 0.0) / top[0], 4)


def _card(index: Index, i: int, rank: int, score: float, question: str, digits: int) -> dict:
    c = index.chunks[i]
    return {
        "chunk_id": c.chunk_id,
        "rank": rank,
        "score": round(score, digits),
        "text": c.text,
        "source": c.source,
        "highlights": overlap_spans(question, c.text),  # 질문과 겹치는 표현(노랑)
        "evidence": None,  # 리랭커가 가장 높게 본 문장(밑줄) — 아래에서 채운다
    }


async def compare(engine: Engine, index: Index, question: str, retriever: str = "hybrid", top_k: int = 8) -> dict:
    """리랭킹 전/후 비교 결과(dict). 웹 /api/compare 와 CLI --json 이 같은 모양을 쓴다."""
    question = question.strip()
    timings: dict[str, int] = {}

    # 1) 질문 임베딩(키워드 검색만 쓸 때는 필요 없음)
    t = time.perf_counter()
    query_vec = None
    if retriever != "keyword":
        query_vec = (await engine.embedder.embed([question]))[0]
    timings["embed"] = _ms(t)

    # 2) 1단계 검색: 후보 top-k 와 1단계 점수
    t = time.perf_counter()
    hits = retrieve.search(index, retriever, question, query_vec, top_k)
    timings["retrieve"] = _ms(t)
    digits = 5 if retriever == "hybrid" else 4  # RRF 점수는 차이가 작아 한 자리 더 보여 준다
    before = [_card(index, i, rank, s, question, digits) for rank, (i, s) in enumerate(hits, start=1)]

    # 3) 리랭킹: 후보를 (질문, 청크) 쌍으로 다시 읽고 0~1 점수 → 점수 순으로 다시 줄 세우기
    reranker, state = engine.reranker_for(index)
    after, top_score, scores = None, None, []
    t = time.perf_counter()
    if reranker is not None:
        chunks = [index.chunks[i] for i, _ in hits]
        scores = await asyncio.to_thread(reranker.score, question, [c.content for c in chunks])

        # 근거 문장: 청크마다 문장을 나눠 한꺼번에 점수를 매기고 가장 높은 문장을 고른다
        sentences = [split_sentences(c.text) for c in chunks]
        flat = [c.text[s:e] for c, spans in zip(chunks, sentences) for s, e in spans]
        flat_scores = await asyncio.to_thread(reranker.score, question, flat) if flat else []
        pos = 0
        for card, spans in zip(before, sentences):
            card["evidence"] = best_sentence(card["text"], flat_scores[pos : pos + len(spans)])
            pos += len(spans)

        order = sorted(range(len(hits)), key=lambda j: (-scores[j], j))  # 동점이면 1단계 순서 유지
        after = []
        for new_rank, j in enumerate(order, start=1):
            move = before[j]["rank"] - new_rank  # 양수 = 올라감
            before[j].update(rank_after=new_rank, move=move, rerank_score=scores[j])
            after.append({**before[j], "rank": new_rank, "score": scores[j], "rank_before": j + 1,
                          "first_stage_score": before[j]["score"], "move": move})
        for card in after:
            card.pop("rank_after", None)
            card.pop("rerank_score", None)
        top_score = max(scores) if scores else None
    timings["rerank"] = _ms(t)

    # 4) 근거 부족 판단: 후보가 없거나, 리랭커 최고 점수가 문턱값보다 낮으면
    no_evidence = not hits or (top_score is not None and top_score < reranker.min_score)

    n = engine.settings.answer_evidence
    return {
        "mode": engine.mode,
        "question": question,
        "collection": index.name,
        "retriever": retriever,
        "top_k": top_k,
        "chunking": {"strategy": index.options.strategy, "title_prefix": index.options.title_prefix},
        "reranker": getattr(reranker, "name", None),
        "reranker_state": state,
        "scales": {"before": retrieve.SCALES[retriever], "after": "리랭커 점수(0~1)"},
        # 1위와 2위 차이(2위 ÷ 1위): 점수 분포 그림과 CLI 요약이 같은 값을 쓴다
        "top2_ratio": {"before": top2_ratio([s for _, s in hits]),
                       "after": top2_ratio(scores) if after is not None else None},
        "before": before,
        "after": after,
        "no_evidence": no_evidence,
        "min_score": getattr(reranker, "min_score", None),
        # mock 모드는 같은 입력 → 같은 출력(바이트 단위)이어야 하므로 시간을 0으로 고정한다
        "timings_ms": timings if engine.mode == "real" else {k: 0 for k in timings},
        "answer_evidence": {
            "A": [c["chunk_id"] for c in before[:n]],
            "B": [c["chunk_id"] for c in after[:n]] if after is not None else None,
        },
    }


async def answer_stream(engine: Engine, index: Index, result: dict, side: str) -> AsyncIterator[dict]:
    """답변 A(전 상위 3개) 또는 B(후 상위 3개)를 이벤트로 흘려보낸다. 마지막은 {'type': 'end', ...}."""
    ids = result["answer_evidence"][side]
    if ids is None:  # 리랭커가 아직 준비되지 않음
        yield {"type": "notice", "text": "리랭커 모델을 준비하는 중이라 답변 B는 잠시 뒤에 만들 수 있습니다."}
        yield {"type": "end", "text": "", "citations": [], "evidence": [], "cached": False}
        return
    if not ids or (side == "B" and result["no_evidence"]):  # 근거가 없으면 모델을 부르지 않는다
        yield {"type": "content", "text": NO_EVIDENCE}
        yield {"type": "end", "text": NO_EVIDENCE, "citations": [], "evidence": ids, "cached": False}
        return

    by_id = {c.chunk_id: c for c in index.chunks}
    evidence = [by_id[i].content for i in ids]
    key = (getattr(engine.generator, "name", ""), result["question"], tuple(evidence))
    cached = engine.answer_cache.get(key)
    if cached is not None:
        engine.answer_cache.move_to_end(key)
        yield {"type": "content", "text": cached}
        text, from_cache, clean = cached, True, True
    else:
        parts, clean = [], True
        # aclosing: 여기서 소비를 멈추면(클라이언트가 끊으면) 안쪽 생성기도 즉시 닫혀 업스트림 스트림이 정리된다
        async with aclosing(engine.generator.stream(result["question"], evidence)) as stream:
            async for event in stream:
                if event["type"] == "content":
                    parts.append(event["text"])
                elif event["type"] == "notice":
                    clean = False
                yield event
        text, from_cache = "".join(parts).strip(), False
        if clean and text:
            engine.answer_cache[key] = text
            while len(engine.answer_cache) > ANSWER_CACHE_MAX:
                engine.answer_cache.popitem(last=False)

    citations = []
    for m in CITATION.finditer(text):
        k = int(m.group(1))
        if 1 <= k <= len(ids) and k not in [c["n"] for c in citations]:
            citations.append({"n": k, "chunk_id": ids[k - 1]})
    yield {"type": "end", "text": text, "citations": citations, "evidence": ids, "cached": from_cache}
