import asyncio
from contextlib import aclosing

import pytest
from schema_check import assert_compare_schema

from rag_rerank.pipeline import answer_stream, compare


async def _collect(engine, index, result, side):
    events = []
    async with aclosing(answer_stream(engine, index, result, side)) as stream:
        async for ev in stream:
            events.append(ev)
    return events


@pytest.mark.parametrize("retriever", ["vector", "keyword", "hybrid"])
def test_compare_schema(engine, make_index, retriever):
    """AC-3: 전/후·이동·척도·인용이 서로 맞는다."""
    index = make_index()
    r = asyncio.run(compare(engine, index, "책을 늦게 돌려주면 벌금을 내나요?", retriever, 8))
    assert r["retriever"] == retriever and r["mode"] == "mock"
    # 키워드 검색은 겹치는 토큰이 있는 청크만 돌려주므로 8개보다 적을 수 있다
    assert 0 < len(r["before"]) <= 8 and (retriever == "keyword" or len(r["before"]) == 8)
    assert_compare_schema(r)
    # 세 검색기 모두 이 질문의 순서가 실제로 바뀐다(리랭커가 1단계와 다른 신호 — 뜻 사전 — 를 쓴다는 증거)
    assert any(c["move"] != 0 for c in r["after"])

    for side in ("A", "B"):
        events = asyncio.run(_collect(engine, index, r, side))
        end = events[-1]
        assert end["type"] == "end"
        assert any(e["type"] == "content" for e in events)
        assert end["citations"], "답변에 [n] 인용이 1개 이상"
        for c in end["citations"]:
            assert 1 <= c["n"] <= 3
            assert c["chunk_id"] == r["answer_evidence"][side][c["n"] - 1]  # 인용 번호 → 실제 근거 청크


def test_answer_cache_is_bounded(engine, make_index):
    from rag_rerank import pipeline

    index = make_index()
    for i in range(pipeline.ANSWER_CACHE_MAX + 10):
        q = f"도서관 질문 {i}번 연체"
        r = asyncio.run(compare(engine, index, q, "keyword", 4))
        asyncio.run(_collect(engine, index, r, "A"))
    assert len(engine.answer_cache) == pipeline.ANSWER_CACHE_MAX


def test_top2_ratio_separates_even_with_tiny_reranker_scores():
    """점수 분포 요약은 '2위 ÷ 1위'를 쓴다 — 리랭커 절대값이 아주 작아도 1위가 앞서는 정도가 드러난다.

    숫자는 실제 모드 실측(중학생 질문, 하이브리드): RRF 0.03279/0.03252…, 리랭커 0.0395/0.0072/0.0011.
    예전 그림은 리랭커를 0~1 고정 축에 찍어 오히려 '더 몰려 보였다'(QA Q-2).
    """
    from rag_rerank.pipeline import top2_ratio

    before = top2_ratio([0.03279, 0.03252, 0.03200, 0.03042, 0.02900])
    after = top2_ratio([0.0395, 0.0072, 0.0011, 0.0004, 0.0001])
    assert before > 0.95  # 1단계: 2위가 1위와 거의 같다(몰려 있음)
    assert after < 0.2  # 리랭커: 2위가 1위의 18% → 1위가 확 앞선다
    assert top2_ratio([0.5]) is None and top2_ratio([0.0, 0.0]) is None  # 비교할 수 없으면 None
    assert top2_ratio([0.4, -0.1]) == 0.0  # 음수 점수는 0으로 본다(0~1 범위 유지)


@pytest.mark.parametrize("retriever", ["vector", "keyword", "hybrid"])
def test_compare_reports_top2_ratio(engine, make_index, retriever):
    index = make_index()
    r = asyncio.run(compare(engine, index, "책을 늦게 돌려주면 벌금을 내나요?", retriever, 8))
    for side in ("before", "after"):
        ratio = r["top2_ratio"][side]
        assert ratio is not None and 0.0 <= ratio <= 1.0
    # after 열은 점수 내림차순이므로 2위/1위 그대로다
    a = r["after"]
    assert r["top2_ratio"]["after"] == pytest.approx(a[1]["score"] / a[0]["score"], abs=1e-3)


def test_distribution_chart_uses_per_column_top_score():
    """화면 그림이 리랭커를 0~1 고정 축에 찍지 않고, 열마다 1위 점수 기준으로 그린다(문구와 그림이 같은 값)."""
    from pathlib import Path

    import rag_rerank

    js = (Path(rag_rerank.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    fn = js[js.index("function renderDistribution"): js.index("// ---------- 답변 A/B")]
    assert "max: 1" not in fn  # 예전: 리랭커 열만 0~1 고정 축
    assert "top2_ratio" in fn and "c.score / top" in fn
    assert "폭에 퍼져" not in fn  # 예전 요약 문장(리랭커가 더 좁게 퍼졌다고 읽히던 문장)
