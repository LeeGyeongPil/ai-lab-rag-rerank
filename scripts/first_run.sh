#!/bin/bash
# 깨끗한 상태(새 가상환경 + 빈 모델·색인 캐시)에서 '첫 비교 결과가 나올 때까지' 걸리는 시간을 잰다.
#   포함: 의존성 설치(uv sync) → 리랭커 모델 내려받기(약 588MB) → 샘플 색인(bge-m3) → 첫 비교
#   제외: Ollama 설치와 `ollama pull`(이미 되어 있다고 가정), uv 자체의 패키지 캐시
# 사용: scripts/first_run.sh            (임시 폴더는 끝나면 지움)
#       KEEP=1 scripts/first_run.sh     (임시 폴더를 남김)
set -euo pipefail
cd "$(dirname "$0")/.."

WORK="$(mktemp -d)"
cleanup() { [ "${KEEP:-}" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

# HF_HOME 이 잡혀 있으면 이미 받은 모델을 재사용하게 되므로, 이 측정에서는 지운다.
unset HF_HOME HF_HUB_CACHE
export UV_PROJECT_ENVIRONMENT="$WORK/venv"   # 새 가상환경
export RAG_CACHE_DIR="$WORK/cache"           # 빈 모델·색인 캐시
export RAG_MODE=real

start=$(date +%s)
uv sync --locked --quiet
uv run --quiet rag-rerank ask "중학생도 공유자전거를 탈 수 있나요?" --retriever hybrid
end=$(date +%s)

uv run --quiet rag-rerank doctor || true
echo "FIRST_RUN_SECONDS=$((end - start))"
