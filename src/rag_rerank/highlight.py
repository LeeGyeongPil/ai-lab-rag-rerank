"""근거 하이라이트.

- overlap_spans : 질문과 겹치는 표현(한글 2글자 토큰) 위치 → 화면의 옅은 노랑
- split_sentences / best_sentence : 청크를 문장으로 나누고, 리랭커가 가장 높게 본 문장 → 밑줄+굵게

구간은 모두 [시작, 끝) 글자 위치(파이썬 문자열 인덱스)이며 서로 겹치지 않게 합쳐서 돌려준다.
"""

from __future__ import annotations

import re

from .bm25 import grams

SENTENCE = re.compile(r"[^.!?。\n]+[.!?。]?")


def merge_spans(spans: list[tuple[int, int]]) -> list[list[int]]:
    """겹치거나 맞닿은 구간을 하나로 합친다. [(0,2),(1,4),(6,8)] → [[0,4],[6,8]]"""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap_spans(query: str, text: str) -> list[list[int]]:
    """질문의 2글자 토큰이 청크 안에 나오는 모든 위치."""
    lowered = text.lower()
    haystack = lowered if len(lowered) == len(text) else text  # 소문자로 바꿔 길이가 달라지면 원문에서 찾는다
    spans = []
    for g in set(grams(query)):
        if len(g) < 2:  # 1글자 토큰은 너무 흔해서 표시하지 않는다
            continue
        start = haystack.find(g)
        while start != -1:
            spans.append((start, start + len(g)))
            start = haystack.find(g, start + 1)
    return merge_spans(spans)


def split_sentences(text: str) -> list[list[int]]:
    """문장 구간 목록(앞뒤 공백 제외)."""
    spans = []
    for m in SENTENCE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append([s, e])
    return spans


def best_sentence(text: str, sentence_scores: list[float]) -> list[int] | None:
    """문장별 리랭커 점수 중 가장 높은 문장 구간(동점이면 앞 문장)."""
    spans = split_sentences(text)
    if not spans or len(spans) != len(sentence_scores):
        return None
    best = max(range(len(spans)), key=lambda i: (sentence_scores[i], -i))
    return spans[best]
