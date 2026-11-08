"""mock(연습) 모드만으로도 README 의 핵심 장면이 그대로 재현되는지 확인한다.

Ollama 가 없는 방문자가 README 를 따라 해도 '정답이 리랭킹 전에는 아래에 있다가, 리랭킹 뒤 1위로 올라오고,
답변 B 가 그 문단을 인용한다'를 볼 수 있어야 한다. README 의 'mock 모드로 따라 하기' 출력과 같은 숫자다.
"""

import asyncio
import os
import selectors
import socket
import subprocess
import sys
import time
from contextlib import aclosing, closing

import pytest

from rag_rerank.cli import browser_url, build_parser
from rag_rerank.pipeline import answer_stream, compare
from rag_rerank.server import FEATURED

FEATURED_GOLD = "만 13세 이상이면 누구나"


def _gold_rank(cards, gold):
    return next((c["rank"] for c in cards if gold in c["text"]), None)


async def _answer(engine, index, result, side):
    async with aclosing(answer_stream(engine, index, result, side)) as stream:
        async for ev in stream:
            if ev["type"] == "end":
                return ev


@pytest.mark.parametrize(
    "question, gold, retriever, before, after",
    [
        # 웹 화면 1단계: 첫 번째 예시 질문, 기본값(하이브리드·top-k 8) → 6위가 1위로(▲5)
        (FEATURED, FEATURED_GOLD, "hybrid", 6, 1),
        # README 의 터미널 예시: 키워드 검색 4위 → 1위
        ("책을 늦게 돌려주면 벌금을 내나요?", "우리 도서관은 연체료를 받지 않습니다", "keyword", 4, 1),
    ],
)
def test_mock_reproduces_readme_lesson(engine, make_index, question, gold, retriever, before, after):
    index = make_index()
    r = asyncio.run(compare(engine, index, question, retriever, 8))
    assert r["mode"] == "mock"
    rank_before, rank_after = _gold_rank(r["before"], gold), _gold_rank(r["after"], gold)
    assert rank_before is not None and rank_before > 1, "정답이 후보 안에 있되 1위는 아니어야 한다"
    assert rank_after == 1, "리랭킹 뒤에는 정답이 1위"
    assert (rank_before, rank_after) == (before, after)  # README 에 붙여 둔 숫자와 같다
    assert not r["no_evidence"]

    # 답변 B 는 1위(정답) 문단을 [1]로 인용하고, 그 문단의 핵심 문장을 옮긴다
    b = asyncio.run(_answer(engine, index, r, "B"))
    gold_id = next(c["chunk_id"] for c in r["after"] if gold in c["text"])
    assert {"n": 1, "chunk_id": gold_id} in b["citations"]
    assert gold in b["text"]


def test_featured_answer_b_quotes_age_sentence(engine, make_index):
    """첫 예시 질문의 답변 B 는 '가입 방법' 문장이 아니라 '만 13세 이상' 문장을 옮긴다(뜻 사전으로 문장 고르기)."""
    index = make_index()
    r = asyncio.run(compare(engine, index, FEATURED, "hybrid", 8))
    b = asyncio.run(_answer(engine, index, r, "B"))
    assert "만 13세 이상이면 누구나 가입할 수 있고" in b["text"] and b["text"].endswith("[1]")
    a = asyncio.run(_answer(engine, index, r, "A"))
    assert FEATURED_GOLD not in a["text"]  # 답변 A 는 엉뚱한 근거를 옮긴다 — 근거가 틀리면 답도 틀린다


def test_mock_keeps_limit_and_no_evidence_cases(engine, make_index):
    """뜻 사전이 한계 사례와 근거 부족 사례를 없애 버리지 않는다(6·7단계도 mock 에서 그대로 보인다)."""
    index = make_index()
    limit = asyncio.run(compare(engine, index, "책 반납이 한 달 넘게 밀리면 어떻게 되나요?", "hybrid", 8))
    assert _gold_rank(limit["before"], "장기 연체가 30일을") is None  # 후보 밖 → 리랭커도 못 살림
    weather = asyncio.run(compare(engine, index, "오늘 서울 날씨는 어때요?", "hybrid", 8))
    assert weather["no_evidence"] is True and weather["min_score"] == 0.1  # mock 은 고정 문턱값 0.1


def test_wildcard_host_prints_usable_url():
    """0.0.0.0 은 브라우저 주소가 아니다 → 이 컴퓨터 주소로 바꿔 보여 준다."""
    assert browser_url("0.0.0.0", 8765) == "http://127.0.0.1:8765"
    assert browser_url("::", 8765) == "http://[::1]:8765"
    assert browser_url("127.0.0.1", 8766) == "http://127.0.0.1:8766"


def test_no_browser_flag_works_with_and_without_subcommand():
    parser = build_parser()
    assert parser.parse_args(["--mock", "--no-browser"]).no_browser is True
    assert parser.parse_args(["serve", "--no-browser"]).no_browser is True
    assert not getattr(parser.parse_args(["--mock"]), "no_browser", False)


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_startup_lines_show_up_without_a_terminal(tmp_path):
    """stdout 이 터미널이 아닐 때(도커 로그·파이프)도 '화면: 주소' 줄이 서버가 떠 있는 동안 바로 보인다."""
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}  # 버퍼링을 끄는 환경변수 없이도 보여야 한다
    env.update(RAG_CACHE_DIR=str(tmp_path), RAG_NO_BROWSER="1")
    proc = subprocess.Popen([sys.executable, "-m", "rag_rerank", "serve", "--mock", "--no-browser", "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    out = b""
    try:
        # 파이썬 쪽 버퍼를 거치지 않고 파이프에서 바로 읽는다(서버가 아직 떠 있는 동안 보이는지가 핵심)
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and "화면:".encode() not in out and proc.poll() is None:
                if sel.select(timeout=0.5):
                    out += os.read(proc.stdout.fileno(), 4096)
        assert proc.poll() is None, "서버가 떠 있는 상태여야 한다"
        lines = out.decode().splitlines()
        assert lines[0].startswith("[MOCK]")
        assert f"화면: http://127.0.0.1:{port}  (끝내려면 Ctrl+C)" in lines
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()
