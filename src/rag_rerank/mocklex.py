"""mock 모드 전용 '뜻 사전': 같은 뜻인데 글자가 다른 표현을 손으로 이어 둔 작은 표.

실제 모델은 '중학생'과 '만 13세 이상'이 관련 있다는 것을 학습으로 익힌다. mock 모드의 가짜 모델은
글자(2글자 토큰)만 보므로 이것을 알 수 없다. 그래서 샘플 문서에 나오는 몇 가지 표현만 이 표로 이어 준다.

- 가짜 임베딩(1단계)은 이 표를 '약하게' 쓴다 → 뜻이 통하는 문단을 후보 안에는 올리지만, 대개 1위까지는 올리지 못한다.
- 가짜 리랭커(2단계)는 이 표를 '제대로' 쓴다 → 후보를 다시 읽을 때 뜻이 통하는 문단을 위로 올린다.

이렇게 '1단계는 대충, 2단계는 꼼꼼히'라는 실제 모델의 차이를 흉내 낸다. 검색 품질과는 상관없는 연습용 장치다.
일부러 넣지 않은 표현도 있다: '한 달 ↔ 30일'(1단계가 놓치는 한계 사례), '날씨'(문서에 없는 질문)
"""

from __future__ import annotations

from .bm25 import grams

# 질문에 나오는 말 → 샘플 문서에서 같은 뜻으로 쓰인 표현들
PARAPHRASES: dict[str, tuple[str, ...]] = {
    "중학생": ("만 13세", "18세 미만"),
    "초등학생": ("14세 미만", "어린이"),
    "벌금": ("연체료",),
    "늦게 돌려": ("연체",),
    "강아지": ("반려동물",),
    "넘어져": ("사고",),
    "다쳤": ("사고", "치료비"),
    "물어내": ("배상", "변상"),
    "자리가 없": ("가득 차",),
    "돈을 더": ("초과 요금",),
}


def concepts(query: str) -> list[tuple[str, tuple[str, ...]]]:
    """질문에 들어 있는 사전 항목들: [(질문 쪽 표현, 문서 쪽 표현들), ...] (사전 순서 그대로)."""
    return [(word, alts) for word, alts in PARAPHRASES.items() if word in query]


def concept_keys(text: str) -> set[str]:
    """이 글이 건드리는 사전 항목 이름들(질문 쪽 표현 또는 문서 쪽 표현이 들어 있으면)."""
    return {word for word, alts in PARAPHRASES.items() if word in text or any(a in text for a in alts)}


def overlap(query: str, text: str) -> int:
    """질문과 글이 겹치는 정도: 겹치는 2글자 토큰 수 + 뜻 사전으로 통하는 표현의 토큰 수.

    mock 답변이 근거에서 '어느 문장'을 옮길지 고를 때 쓴다(예: '중학생' 질문 → '만 13세 …' 문장).
    """
    q = set(grams(query))
    n = len(q & set(grams(text)))
    for word, alts in concepts(query):
        if word not in text and any(a in text for a in alts):
            n += len(set(grams(word)))
    return n
