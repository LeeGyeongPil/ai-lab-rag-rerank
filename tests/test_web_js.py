"""Run the real web script (static/app.js) in node with a tiny fake DOM.

Regression for a local variable named ``tr`` that shadowed the translation function and
stopped the Experiment table after three rows. The old tests only checked keys, not behaviour.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_rerank.server import create_app

ROOT = Path(__file__).parent
STATIC = ROOT.parent / "src" / "rag_rerank" / "static"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed (needed to run app.js)")


@pytest.fixture()
def eval_fixture(tmp_path) -> Path:
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        data = client.post("/api/eval", json={"top_k": 8}).json()
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_experiment_table_draws_every_question(lang, eval_fixture):
    data = json.loads(eval_fixture.read_text(encoding="utf-8"))
    assert any(r is None or r[0] is None for q in data["questions"] for r in q["ranks"].values()), \
        "fixture must contain a rank outside the candidates (the case that used to crash)"
    out = subprocess.run(["node", str(ROOT / "js" / "render_eval.mjs"), str(STATIC), lang, str(eval_fixture)],
                         capture_output=True, text=True, timeout=60, check=True)
    result = json.loads(out.stdout)
    assert result["thrown"] is None
    assert result["rows"] == len(data["questions"]) == 14
    assert result["summaryRows"] == len(data["table"])
    assert result["outside"] is True
    assert result["lang"] == lang
    assert "Error" not in result["status"] and "오류" not in result["status"]


def test_no_local_named_like_the_translation_function():
    """Cheap guard that also works without node."""
    import re
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"\b(const|let|var)\s+tr\b|\(\s*tr\s*\)\s*=>", src)
