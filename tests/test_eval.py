import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from rag_rerank.evaluate import evaluate, load_questions

GOLDEN = Path(__file__).parent / "golden" / "eval_mock.json"


def test_eval_mock_matches_golden(engine, make_index):
    """AC-7: 성적표가 골든 값과 같고, 순서가 바뀐 조합이 1개 이상."""
    index = make_index()
    questions = load_questions(engine.settings.samples_dir / "questions.json")
    result = asyncio.run(evaluate(engine, index, questions, 8))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert result == golden
    assert result["orders_changed"] > 0
    assert set(result["table"]) == {"keyword", "vector", "hybrid"}


def test_eval_cli_json_matches_golden(tmp_path):
    """명령 'rag-rerank eval --mock --json' 의 출력도 골든과 같다(별도 프로세스)."""
    env = {**os.environ, "RAG_CACHE_DIR": str(tmp_path), "RAG_MODE": "auto"}
    out = subprocess.run([sys.executable, "-m", "rag_rerank", "eval", "--mock", "--json"],
                         capture_output=True, text=True, env=env, timeout=60, check=True)
    assert json.loads(out.stdout) == json.loads(GOLDEN.read_text(encoding="utf-8"))
