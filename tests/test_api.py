import asyncio
import json
from contextlib import aclosing

import httpx
from fastapi.testclient import TestClient
from schema_check import assert_compare_schema

from rag_rerank.generator import OllamaGenerator
from rag_rerank.server import create_app


def _events(resp) -> list[dict]:
    return [json.loads(line) for line in resp.iter_lines() if line.strip()]


def test_index_page_status_compare():
    """AC-8: 화면·상태·비교·답변 스트림이 mock 에서 동작한다."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:  # with: 서버 시작·종료(lifespan)까지 실행
        page = client.get("/")
        assert page.status_code == 200
        assert 'id="col-before"' in page.text and 'id="col-after"' in page.text
        assert client.get("/static/app.js").status_code == 200

        st = client.get("/api/status").json()
        assert st["mode"] == "mock"
        assert st["questions"][0]["q"] == "중학생도 공유자전거를 탈 수 있나요?"
        assert len(st["questions"]) == 14

        body = {"question": "책을 늦게 돌려주면 벌금을 내나요?", "retriever": "hybrid", "top_k": 8}
        r = client.post("/api/compare", json=body)
        assert r.status_code == 200
        assert_compare_schema(r.json())

        with client.stream("POST", "/api/answer", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/x-ndjson")
            events = _events(resp)
        assert events[0]["type"] == "start"
        assert events[-1] == {"type": "done"}
        for side in ("A", "B"):
            mine = [e for e in events if e.get("side") == side]
            assert any(e["type"] == "content" for e in mine)
            assert any(e["type"] == "thinking" for e in mine)
            assert mine[-1]["type"] == "end" and mine[-1]["citations"]

        ev = client.post("/api/eval", json={"top_k": 8})
        assert ev.status_code == 200 and ev.json()["n_questions"] == 14


def test_bad_requests_are_rejected():
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        assert client.post("/api/compare", json={"question": ""}).status_code == 422
        assert client.post("/api/compare", json={"question": "x", "top_k": 99}).status_code == 422
        assert client.post("/api/compare", json={"question": "x", "collection": "mine"}).status_code == 400
        assert client.post("/api/index", json={}).status_code == 400


class _UpstreamStream(httpx.AsyncByteStream):
    """가짜 Ollama 스트림: 생각 토큰을 끝없이 흘리고, 닫혔는지 기록한다."""

    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'{"message":{"content":"A"}}\n'
        for _ in range(10_000):
            yield b'{"message":{"thinking":"t"}}\n'
            await asyncio.sleep(0)

    async def aclose(self):
        self.closed = True


def test_generator_closes_upstream_when_consumer_stops():
    """클라이언트가 중간에 끊으면(소비 중단) Ollama 응답 스트림을 반드시 닫는다."""
    upstream = _UpstreamStream()

    async def scenario():
        transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=upstream))
        async with httpx.AsyncClient(transport=transport, base_url="http://ollama.test") as client:
            gen = OllamaGenerator(client, "fake-model")
            async with aclosing(gen.stream("질문", ["근거"])) as stream:
                async for ev in stream:
                    if ev["type"] == "content":
                        break  # 브라우저가 '중지'를 누른 상황
        return upstream.closed

    assert asyncio.run(scenario()) is True


def test_only_loopback_host_header_is_accepted():
    """DNS 리바인딩 방어: Host 가 127.0.0.1/localhost(포트 무관)가 아니면 400."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        for host in ("127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:8766"):
            assert client.get("/api/status", headers={"host": host}).status_code == 200, host
        for host in ("evil.example", "evil.example:8765", "127.0.0.1.evil.example", "192.168.0.10:8765"):
            r = client.post("/api/index", json={"folder": "/"}, headers={"host": host})
            assert r.status_code == 400 and "Invalid host" in r.text, host


def test_ask_answer_help_matches_measured_time():
    """--answer 도움말의 시간 안내가 실측(답 하나에 수십 초~2분, 120초 상한)과 맞는다."""
    from rag_rerank.cli import build_parser
    from rag_rerank.generator import OllamaGenerator

    sub = next(a for a in build_parser()._actions if a.dest == "cmd")
    help_text = next(a.help for a in sub.choices["ask"]._actions if a.dest == "answer")
    assert "45~50초" not in help_text
    assert "2분" in help_text
    # 도움말의 '2분'은 생성기의 시간 상한과 같은 값이어야 한다
    assert OllamaGenerator(None, "m").time_limit == 120.0


def test_all_interfaces_mode_accepts_lan_host():
    # serve --host 0.0.0.0 은 다른 기기 접속을 직접 고른 것이므로 Host 검사를 풀어 둔다(extra_hosts=["*"])
    with TestClient(create_app(extra_hosts=["*"]), base_url="http://192.168.0.10:8765") as client:
        assert client.get("/api/status").status_code == 200
