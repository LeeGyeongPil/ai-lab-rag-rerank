"""임베딩: 글 → 의미 벡터(숫자 목록). 의미가 가까운 글일수록 벡터 방향이 비슷해진다.

- OllamaEmbedder : 로컬 Ollama 의 bge-m3(1024차원, 다국어) — 실제 모드
- MockEmbedder   : 한글 2글자 토큰을 256칸에 해싱해 세는 가짜 벡터 — mock 모드
"""

from __future__ import annotations

import zlib

import httpx
import numpy as np

from .bm25 import grams


class MockEmbedder:
    """결정적 가짜 임베딩. 같은 글이면 어느 컴퓨터·어느 실행에서든 같은 벡터가 나온다."""

    name = "mock-hash256"
    dim = 256

    async def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for row, text in enumerate(texts):
            for g in grams(text):
                # 파이썬 hash()는 실행마다 값이 바뀐다 → 항상 같은 값을 주는 crc32 를 쓴다
                out[row, zlib.crc32(g.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)  # 길이 1로 맞춤(코사인 비교용)


class OllamaEmbedder:
    """Ollama /api/embed 에 여러 글을 한 번에(배치) 보내 벡터를 받는다."""

    def __init__(self, client: httpx.AsyncClient, model: str, batch: int = 32):
        self.client, self.model, self.batch = client, model, batch
        self.name = f"ollama-{model}"

    async def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = await self.client.post(
                "/api/embed", json={"model": self.model, "input": texts[i : i + self.batch]}, timeout=120
            )
            r.raise_for_status()
            vectors += r.json()["embeddings"]
        return np.asarray(vectors, dtype=np.float32)
