"""Regression: CI once ran for 6 hours. Keep the workflow and the test suite from waiting forever."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_workflow_job_and_steps_have_time_limits():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^    timeout-minutes: \d+", text, re.M), "job needs timeout-minutes"
    assert len(re.findall(r"^\s+- run:", text, re.M)) == len(re.findall(r"^\s+timeout-minutes: \d+", text, re.M)) - 1


def test_workflow_runs_on_pull_requests_and_pushes():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^  push:", text, re.M) and re.search(r"^  pull_request:", text, re.M)


def test_pytest_has_a_per_test_timeout(pytestconfig):
    assert int(pytestconfig.getini("timeout")) > 0
