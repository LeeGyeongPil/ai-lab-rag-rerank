import asyncio

from rag_rerank.highlight import merge_spans, overlap_spans, split_sentences
from rag_rerank.pipeline import compare


def _valid(spans, text):
    for s, e in spans:
        assert 0 <= s < e <= len(text)
    for (_, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 < s2  # 정렬되어 있고 겹치거나 맞닿지 않음


def test_spans_valid_and_evidence_sentence(engine, make_index):
    """AC-4: 모든 구간이 범위 안·비겹침, 정답 청크의 근거 문장이 맞게 선택된다."""
    index = make_index()
    q = "일요일에는 도서관이 몇 시에 문을 닫나요?"
    r = asyncio.run(compare(engine, index, q, "hybrid", 8))
    for card in r["before"] + r["after"]:
        _valid(card["highlights"], card["text"])
        s, e = card["evidence"]
        assert 0 <= s < e <= len(card["text"])
    gold = next(c for c in r["after"] if "토요일과 일요일에는 오전 9시에" in c["text"])
    s, e = gold["evidence"]
    assert gold["text"][s:e] == "토요일과 일요일에는 오전 9시에 문을 열어 오후 5시에 닫습니다."
    assert gold["rank"] == 1


def test_merge_and_overlap():
    assert merge_spans([(0, 2), (1, 4), (6, 8), (8, 9)]) == [[0, 4], [6, 9]]
    text = "도서관 운영 시간은 평일 도서관 기준입니다."
    spans = overlap_spans("도서관은 언제 여나요", text)
    assert [text[s:e] for s, e in spans] == ["도서관", "도서관"]


def test_split_sentences():
    text = "첫 문장입니다. 둘째 문장! 셋째"
    assert [text[s:e] for s, e in split_sentences(text)] == ["첫 문장입니다.", "둘째 문장!", "셋째"]
