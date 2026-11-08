"""실제 모드 스모크(AC-10 일부). Ollama·모델·리랭커 파일이 없으면 건너뛴다 — CI 에서는 항상 건너뜀.

답변까지 확인하려면(느림, 1~2분): RAG_REAL_ANSWER=1 uv run pytest -m real
답변이 시간 상한(120초)에 걸려 '근거 문장을 옮긴 대체 답'이 나오면 인용 검사는 의미가 없으므로 건너뛴다(skip).
"""

import asyncio
import os
from contextlib import aclosing

import pytest

from rag_rerank.chunking import ChunkOptions
from rag_rerank.config import ModeInfo, Settings, has_model, probe_ollama
from rag_rerank.pipeline import answer_stream, compare, create_engine
from rag_rerank.reranker import RerankerHub
from rag_rerank.store import build_index, load_folder

pytestmark = pytest.mark.real
GOLD = "만 13세 이상이면 누구나"


def model_answer_end(events: list[dict]) -> dict:
    """답변 이벤트에서 마지막 'end'를 꺼낸다. 안내(notice)가 섞였으면 모델이 끝까지 답한 게 아니므로 skip.

    notice 는 시간 초과로 대체 답(추출형, 늘 [1]을 인용)을 썼거나 답을 중간에 멈춘 경우에만 나온다.
    그 상태로 인용을 검사하면 모델의 근거 답변을 확인하지 못한 채 통과해 버린다(QA Q-4).
    """
    notices = [e["text"] for e in events if e["type"] == "notice"]
    if notices:
        pytest.skip("모델 답변이 끝까지 나오지 않아 인용 검사를 건너뜀: " + " / ".join(notices))
    ends = [e for e in events if e["type"] == "end"]
    assert len(ends) == 1 and ends[0]["text"], "답변 끝 이벤트가 하나 있어야 한다"
    return ends[0]


def test_real_smoke(tmp_path):
    settings = Settings.from_env()
    ok, names, _ = probe_ollama(settings)
    if not ok or not all(has_model(names, m) for m in (settings.embed_model, settings.chat_model)):
        pytest.skip("Ollama 또는 모델 없음")
    if RerankerHub(settings).local_paths() is None:
        pytest.skip("리랭커 모델 파일 없음 — uv run rag-rerank doctor --download")

    async def scenario():
        engine = create_engine(settings, ModeInfo("real", "test"))
        try:
            await asyncio.to_thread(engine.hub.prepare_blocking, False)
            docs = load_folder(settings.samples_dir)
            index = await build_index("samples", docs, ChunkOptions(), engine.embedder, tmp_path)
            r = await compare(engine, index, "중학생도 공유자전거를 탈 수 있나요?", "hybrid", 8)
            events = None
            if os.environ.get("RAG_REAL_ANSWER"):
                events = []  # 답변 B 이벤트를 전부 모은다(notice 포함)
                async with aclosing(answer_stream(engine, index, r, "B")) as stream:
                    async for ev in stream:
                        events.append(ev)
            return r, events
        finally:
            await engine.aclose()

    r, events = asyncio.run(scenario())
    before = next(c["rank"] for c in r["before"] if GOLD in c["text"])
    after = next(c for c in r["after"] if GOLD in c["text"])
    assert before >= 2 and after["rank"] == 1
    # 점수 분포의 요약 값: 리랭커가 1위를 1단계보다 더 뚜렷하게 앞세운다(2위 ÷ 1위가 더 작다)
    assert r["top2_ratio"]["after"] < r["top2_ratio"]["before"]
    if events is not None:
        answer_b = model_answer_end(events)  # 대체 답이면 여기서 skip
        assert after["chunk_id"] in [c["chunk_id"] for c in answer_b["citations"]]
