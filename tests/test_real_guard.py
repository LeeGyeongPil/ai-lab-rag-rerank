"""실제 모드 테스트의 안전장치를 mock 으로 검사한다: 대체 답(notice)이 섞이면 인용 검사를 통과시키지 않고 skip."""

import pytest
from test_real import model_answer_end

from rag_rerank.generator import extractive_answer

END = {"type": "end", "text": "답 [1]", "citations": [{"n": 1, "chunk_id": "x#0"}], "evidence": ["x#0"], "cached": False}


def test_fallback_notice_skips_citation_check():
    # OllamaGenerator 가 120초를 넘겼을 때 흘리는 이벤트 모양 그대로
    events = [
        {"type": "thinking", "tokens": 900},
        {"type": "notice", "text": "생각이 120초를 넘어 중단하고, 근거 문장을 그대로 옮긴 답으로 대신합니다."},
        {"type": "content", "text": extractive_answer("질문", ["근거 문장입니다."])},
        END,
    ]
    with pytest.raises(pytest.skip.Exception, match="120초"):
        model_answer_end(events)


def test_clean_model_answer_passes_through():
    events = [{"type": "thinking", "tokens": 10}, {"type": "content", "text": "답 [1]"}, END]
    assert model_answer_end(events) is END
