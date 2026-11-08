"""환경변수 → 설정값, 그리고 real/mock 모드 결정.

모드는 세 가지다.
- real : 로컬 Ollama(bge-m3 임베딩, qwen3:4b 답변) + ONNX 리랭커를 실제로 쓴다.
- mock : 네트워크·모델 없이 가짜(결정적) 임베딩·리랭커·답변을 쓴다. 테스트와 CI가 쓴다.
- auto : 시작할 때 Ollama를 2초 안에 확인해 real 또는 mock 으로 정한다(기본값).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# 리랭커 모델: 다국어 크로스 인코더 bge-reranker-v2-m3 의 ONNX 변환본, int8 양자화 파일
RERANKER_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
RERANKER_REVISION = "6f5ff65298512715a1e669753bc754d2bc8f367b"  # 커밋 해시로 고정(재현성)
RERANKER_FILES = ("onnx/model_int8.onnx", "tokenizer.json")

# 저장소 루트(src/rag_rerank/config.py 기준 두 단계 위) — 샘플 문서 위치를 찾는 데 쓴다.
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    mode: str = "auto"  # auto | real | mock
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    chat_model: str = "qwen3:4b"
    reranker_revision: str = RERANKER_REVISION
    cache_dir: Path = Path(".cache")
    host: str = "127.0.0.1"  # 내 컴퓨터에서만 접속 가능하게 묶는다
    port: int = 8765
    min_rerank_score: float = 0.01  # 리랭커 최고 점수가 이보다 낮으면 "근거 부족"
    top_k: int = 8  # 1단계 후보 수(기본)
    answer_evidence: int = 3  # 답변에 넣을 근거 수(전/후 각각 상위 3개)
    samples_dir: Path = field(default_factory=lambda: REPO_ROOT / "samples")

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        samples = REPO_ROOT / "samples"
        if not samples.is_dir():  # 저장소 밖에서 설치해 실행한 경우: 현재 폴더 기준
            samples = Path.cwd() / "samples"
        return cls(
            mode=env.get("RAG_MODE", "auto").strip().lower() or "auto",
            ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            embed_model=env.get("RAG_EMBED_MODEL", "bge-m3"),
            chat_model=env.get("RAG_CHAT_MODEL", "qwen3:4b"),
            reranker_revision=env.get("RAG_RERANKER_REVISION", RERANKER_REVISION),
            cache_dir=Path(env.get("RAG_CACHE_DIR", ".cache")),
            port=int(env.get("RAG_PORT", "8765")),
            min_rerank_score=float(env.get("RAG_MIN_RERANK_SCORE", "0.01")),
            samples_dir=samples,
        )


@dataclass
class ModeInfo:
    """실제로 쓰게 된 모드와 그 이유(화면 상단과 doctor 에 그대로 보여 준다)."""

    mode: str  # real | mock
    reason: str
    hints: list[str] = field(default_factory=list)  # 사용자가 따라 할 설치 명령


def probe_ollama(settings: Settings, timeout: float = 2.0) -> tuple[bool, list[str], str]:
    """Ollama 가 떠 있는지, 설치된 모델 목록이 무엇인지 확인한다. (연결됨?, 모델들, 오류 설명)"""
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return True, names, ""
    except Exception as e:  # 연결 거부·시간 초과 등 — auto 모드에서는 mock 으로 내려간다
        return False, [], f"{type(e).__name__}"


def has_model(names: list[str], wanted: str) -> bool:
    """'bge-m3' 와 'bge-m3:latest' 를 같은 것으로 본다."""
    want = wanted if ":" in wanted else f"{wanted}:latest"
    return any(n == wanted or n == want for n in names)


def resolve_mode(settings: Settings, force_mock: bool = False) -> ModeInfo:
    """설정과 Ollama 상태를 보고 real/mock 중 하나를 고른다."""
    if force_mock or settings.mode == "mock":
        return ModeInfo("mock", "--mock 플래그(또는 RAG_MODE=mock)로 mock 모드를 켰습니다.")
    ok, names, err = probe_ollama(settings)
    hints = []
    if not ok:
        reason = f"Ollama({settings.ollama_base_url})에 연결하지 못했습니다({err})."
        hints = ["https://ollama.com 에서 Ollama 설치 후 실행",
                 f"ollama pull {settings.embed_model}", f"ollama pull {settings.chat_model}"]
    else:
        missing = [m for m in (settings.embed_model, settings.chat_model) if not has_model(names, m)]
        if not missing:
            return ModeInfo("real", "Ollama와 모델이 준비되어 실제 모드로 실행합니다.")
        reason = "Ollama에 모델이 없습니다: " + ", ".join(missing)
        hints = [f"ollama pull {m}" for m in missing]
    if settings.mode == "real":  # real 을 강제했는데 준비가 안 됐으면 바로 알린다
        raise SystemExit(f"[real 모드 불가] {reason}\n  해결: " + " / ".join(hints))
    return ModeInfo("mock", reason + " 그래서 mock 모드로 실행합니다.", hints)
