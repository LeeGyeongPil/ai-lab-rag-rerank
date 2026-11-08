"""답변 생성: 근거 청크 3개만 보고 답하고 [1]처럼 출처 번호를 단다(근거 기반 답변).

- OllamaGenerator : Ollama /api/chat + think:true + 스트리밍 — 실제 모드
  · qwen3:4b 는 '생각 전용' 모델이라 생각을 끌 수 없다. think:true 로 받으면 Ollama 가
    생각(message.thinking)과 답(message.content)을 나눠 준다 → 생각은 토큰 수만 보여 주고 답만 쓴다.
- MockGenerator   : 근거 1번에서 질문과 가장 많이 겹치는(뜻 사전 포함) 문장을 그대로 옮기는 추출형 답 — mock 모드

두 생성기 모두 같은 이벤트를 흘려보낸다(비동기 제너레이터).
  {"type": "thinking", "tokens": 120}   생각 진행(토큰 수)
  {"type": "content",  "text": "..."}   답 조각
  {"type": "notice",   "text": "..."}   안내(시간 초과로 대체 답을 쓴 경우 등)
"""

from __future__ import annotations

import json
import re
import time
from typing import AsyncIterator, Callable

import httpx

from .bm25 import grams
from .highlight import split_sentences
from .mocklex import overlap as lexicon_overlap

PREFIX = re.compile(r"^\[[^\]]*\]\s*")
NO_EVIDENCE = "문서에서 근거를 찾지 못했습니다."
SYSTEM_PROMPT = (
    "주어진 근거만 사용해 한국어로 한두 문장으로 답하고, 사용한 근거 번호를 [1]처럼 붙이세요. "
    f"근거에 답이 없으면 '{NO_EVIDENCE}'라고만 답하세요."
)


def build_messages(question: str, evidence: list[str]) -> list[dict]:
    context = "\n".join(f"[{i}] {text}" for i, text in enumerate(evidence, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n질문: {question}"},
    ]


def token_overlap(question: str, sentence: str) -> int:
    """질문과 문장이 함께 가진 2글자 토큰 수."""
    return len(set(grams(question)) & set(grams(sentence)))


def extractive_answer(question: str, evidence: list[str],
                      overlap: Callable[[str, str], int] = token_overlap) -> str:
    """근거 1번에서 질문과 가장 많이 겹치는 문장을 골라 '[1]'을 붙인다(모델 없이 만드는 답)."""
    if not evidence:
        return NO_EVIDENCE
    text = PREFIX.sub("", evidence[0])  # 청크 앞의 "[문서 > 절 제목] " 접두는 답에 넣지 않는다
    sentences = [text[s:e] for s, e in split_sentences(text)] or [text]
    best = max(sentences, key=lambda s: (overlap(question, s), -sentences.index(s)))  # 동점이면 앞 문장
    return f"근거에 따르면 {best} [1]"


class MockGenerator:
    """결정적 가짜 생성기: 가짜 '생각' 토큰 몇 개 → 추출형 답을 몇 글자씩 흘려보낸다."""

    name = "mock-extractive"

    async def stream(self, question: str, evidence: list[str]) -> AsyncIterator[dict]:
        for n in (4, 8, 12):
            yield {"type": "thinking", "tokens": n}
        # mock 리랭커와 같은 뜻 사전으로 문장을 고른다('중학생' 질문 → '만 13세 …' 문장)
        answer = extractive_answer(question, evidence, overlap=lexicon_overlap)
        for i in range(0, len(answer), 12):
            yield {"type": "content", "text": answer[i : i + 12]}


class OllamaGenerator:
    """Ollama 채팅 스트림. 클라이언트가 끊거나 시간이 넘으면 업스트림 응답을 반드시 닫는다."""

    def __init__(self, client: httpx.AsyncClient, model: str, num_predict: int = 2048, time_limit: float = 120.0):
        self.client, self.model = client, model
        self.num_predict, self.time_limit = num_predict, time_limit
        self.name = f"ollama-{model}"

    async def stream(self, question: str, evidence: list[str]) -> AsyncIterator[dict]:
        payload = {
            "model": self.model,
            "messages": build_messages(question, evidence),
            "think": True,  # 생각과 답을 나눠 받는다
            "stream": True,
            "options": {"temperature": 0, "num_predict": self.num_predict},  # 생각이 끝없이 길어지지 않게 상한
        }
        request = self.client.build_request(
            "POST", "/api/chat", json=payload, timeout=httpx.Timeout(10.0, read=self.time_limit)
        )
        response = await self.client.send(request, stream=True)
        started, thinking, timed_out, got_answer = time.monotonic(), 0, False, False
        try:
            response.raise_for_status()
            async for line in response.aiter_lines():  # 한 줄 = JSON 하나(토큰 1개 안팎)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                part = msg.get("message", {})
                if part.get("thinking"):
                    thinking += 1
                    if thinking % 16 == 1:  # 화면 갱신은 16토큰마다 한 번이면 충분
                        yield {"type": "thinking", "tokens": thinking}
                if part.get("content"):
                    got_answer = True
                    yield {"type": "content", "text": part["content"]}
                if msg.get("done"):
                    break
                if time.monotonic() - started > self.time_limit:
                    timed_out = True
                    break
        finally:
            # 정상 종료·시간 초과·클라이언트 연결 끊김(GeneratorExit) 모두 여기로 온다 → 업스트림 스트림을 닫는다.
            # 닫으면 Ollama 도 생성을 멈춘다(다음 답변이 대기열에서 기다리지 않게).
            await response.aclose()
        yield {"type": "thinking", "tokens": thinking}
        if timed_out and not got_answer:
            yield {"type": "notice", "text": f"생각이 {int(self.time_limit)}초를 넘어 중단하고, 근거 문장을 그대로 옮긴 답으로 대신합니다."}
            yield {"type": "content", "text": extractive_answer(question, evidence)}
        elif timed_out:
            yield {"type": "notice", "text": f"{int(self.time_limit)}초를 넘어 답을 중간에 멈췄습니다."}
