"""리랭커(2단계): 후보 청크를 질문과 '함께' 다시 읽고 관련도 점수(0~1)를 매겨 순서를 고친다.

- OnnxCrossEncoder : bge-reranker-v2-m3 int8 ONNX 를 onnxruntime(CPU)으로 돌린다 — 실제 모드
  · 크로스 인코더: (질문, 청크) 한 쌍을 하나의 입력으로 넣는다. 1단계(따로 벡터로 만들어 거리 비교)보다
    느리지만 '만 13세 이상' 같은 세부 조건까지 함께 읽는다. 그래서 후보 수십 개에만 쓴다.
- MockReranker     : 질문 2글자 토큰이 청크에 얼마나 들어 있는지(드문 토큰일수록 가중) — mock 모드
- RerankerHub      : 모델 파일 내려받기(백그라운드 스레드 하나, 진행률)와 지연 로딩을 맡는다.

onnxruntime·tokenizers·huggingface_hub 는 실제 모드에서만 import 한다(mock·CI 는 필요 없음).
"""

from __future__ import annotations

import math
import os
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from .bm25 import grams
from .config import RERANKER_FILES, RERANKER_REPO, Settings


class MockReranker:
    """결정적 가짜 리랭커. 1단계(해시 벡터·BM25)와 다른 신호라서 순서가 실제로 바뀐다."""

    name = "mock-overlap"
    min_score = 0.1  # 겹치는 토큰이 거의 없으면(포함률 7.5% 미만) 근거 부족으로 본다

    def __init__(self, df: Counter, n_docs: int):
        self.df, self.n_docs = df, n_docs

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(grams(query))
        weight = {g: math.log(1 + self.n_docs / (1 + self.df[g])) for g in q}  # 드문 토큰일수록 무겁게
        total = sum(weight.values()) or 1.0
        out = []
        for text in texts:
            present = set(grams(text))
            cover = sum(w for g, w in weight.items() if g in present) / total  # 질문 토큰 포함률(가중)
            out.append(round(1 / (1 + math.exp(-8 * (cover - 0.35))), 4))  # 시그모이드로 0~1
        return out


class OnnxCrossEncoder:
    """bge-reranker-v2-m3(int8 ONNX). torch 없이 onnxruntime + tokenizers 만으로 돈다."""

    name = "bge-reranker-v2-m3-int8"

    def __init__(self, model_path: str, tokenizer_path: str, min_score: float):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.min_score = min_score
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(512)  # 질문+청크 합쳐 512토큰까지
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")  # XLM-R 계열은 패딩 id 가 1 (0이면 점수가 틀어짐)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self._lock = threading.Lock()  # 토크나이저 패딩 설정을 여러 요청이 동시에 건드리지 않게

    def score(self, query: str, texts: list[str], batch: int = 16) -> list[float]:
        out: list[float] = []
        with self._lock:
            for i in range(0, len(texts), batch):
                enc = self.tok.encode_batch([(query, t) for t in texts[i : i + batch]])  # (질문, 청크) 쌍
                feed = {
                    "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                }
                feed = {k: v for k, v in feed.items() if k in self.input_names}
                logits = self.session.run(None, feed)[0].reshape(-1)
                out += [round(float(1 / (1 + np.exp(-x))), 4) for x in logits]  # 시그모이드 → 0~1
        return out


class _NullWriter:
    """진행 막대 글자를 버리는 출력(서버에서는 진행률을 화면 API 로만 보여 준다)."""

    def write(self, _s: str) -> None: ...

    def flush(self) -> None: ...


class RerankerHub:
    """리랭커 모델을 한 번만 내려받고(single-flight), 한 번만 불러 와(싱글턴) 나눠 쓴다.

    상태: idle → downloading → loading → ready  (실패하면 error, '다시 시도'는 최대 3회)
    """

    MAX_ATTEMPTS = 3

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = "idle"
        self.error: str | None = None
        self.done_bytes = 0
        self.total_bytes: int | None = None
        self.attempts = 0
        self._model: OnnxCrossEncoder | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ----- 파일 위치 -----
    def _cache_dir(self) -> str | None:
        # HF_HOME / HF_HUB_CACHE 를 직접 지정했다면 그 캐시를 그대로 쓴다(이미 받은 모델 재사용).
        # 아니면 프로젝트 안 .cache/models 에 받는다(홈 폴더에 2GB 가 쌓이지 않게).
        if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
            return None
        return str(Path(self.settings.cache_dir) / "models")

    def local_paths(self) -> list[str] | None:
        """이미 받아 둔 파일이 있으면 그 경로들, 없으면 None (네트워크를 쓰지 않음)."""
        from huggingface_hub import hf_hub_download

        try:
            return [
                hf_hub_download(RERANKER_REPO, f, revision=self.settings.reranker_revision,
                                cache_dir=self._cache_dir(), local_files_only=True)
                for f in RERANKER_FILES
            ]
        except Exception:
            return None

    # ----- 상태 -----
    def status(self) -> dict:
        pct = None
        if self.total_bytes:
            pct = round(100 * min(self.done_bytes, self.total_bytes) / self.total_bytes, 1)
        return {
            "state": self.state,
            "model": f"{RERANKER_REPO}@{self.settings.reranker_revision[:7]} (int8)",
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "percent": pct,
            "error": self.error,
            "attempts": self.attempts,
            "can_retry": self.state == "error" and self.attempts < self.MAX_ATTEMPTS,
        }

    def get(self) -> OnnxCrossEncoder | None:
        return self._model if self.state == "ready" else None

    # ----- 내려받기 + 불러오기 -----
    def start_background(self) -> bool:
        """백그라운드 스레드 하나로 준비를 시작한다. 이미 진행 중이거나 시도 횟수를 넘겼으면 False."""
        with self._lock:
            busy = self._thread is not None and self._thread.is_alive()
            if busy or self.state == "ready" or self.attempts >= self.MAX_ATTEMPTS:
                return False
            self.attempts += 1
            self._thread = threading.Thread(target=self._prepare, args=(False,), name="reranker-download", daemon=True)
            self._thread.start()
            return True

    def prepare_blocking(self, show_progress: bool = True) -> OnnxCrossEncoder:
        """CLI 용: 지금 이 자리에서 내려받고(터미널 진행 막대) 불러온다."""
        if self.state != "ready":
            self.attempts += 1
            self._prepare(show_progress)
        if self._model is None:
            raise RuntimeError(f"리랭커 준비 실패: {self.error}")
        return self._model

    def _prepare(self, show_progress: bool) -> None:
        try:
            paths = self.local_paths()
            if paths is None:
                self.state = "downloading"
                paths = self._download(show_progress)
            self.state = "loading"
            self._model = OnnxCrossEncoder(paths[0], paths[1], self.settings.min_rerank_score)
            self.state, self.error = "ready", None
        except Exception as e:  # 네트워크 끊김 등: 자동 재시도하지 않고 '다시 시도' 버튼에 맡긴다
            self.state, self.error = "error", f"{type(e).__name__}: {e}"[:300]

    def _download(self, show_progress: bool) -> list[str]:
        from huggingface_hub import hf_hub_download
        from tqdm.std import tqdm

        rev, cache = self.settings.reranker_revision, self._cache_dir()
        # 전체 크기를 먼저 물어본다(dry_run: 내려받지 않고 정보만)
        sizes = [hf_hub_download(RERANKER_REPO, f, revision=rev, cache_dir=cache, dry_run=True).file_size or 0
                 for f in RERANKER_FILES]
        self.total_bytes, self.done_bytes = sum(sizes) or None, 0
        hub = self

        paths, finished = [], 0
        for f, size in zip(RERANKER_FILES, sizes):
            offset = finished

            class Progress(tqdm):  # 진행 막대가 갱신될 때마다 hub 에 바이트 수를 알려 준다
                def __init__(self, *args, **kwargs):
                    kwargs.pop("name", None)
                    if not show_progress:
                        kwargs["file"] = _NullWriter()
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    out = super().update(n)
                    hub.done_bytes = offset + int(self.n)
                    return out

            paths.append(hf_hub_download(RERANKER_REPO, f, revision=rev, cache_dir=cache, tqdm_class=Progress))
            finished += size
            self.done_bytes = finished
        return paths

    def close(self) -> None:
        """서버 종료 시 ONNX 세션을 놓아 메모리를 돌려준다(내려받기 스레드는 daemon 이라 함께 끝난다)."""
        self._model = None
        if self.state == "ready":
            self.state = "idle"
