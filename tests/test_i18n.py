"""화면 언어(en/ko): 메시지 표 키 일치, 언어 우선순위, 언어 전환, 기계용 출력 불변."""

import json
import re
import string
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_rerank import i18n
from rag_rerank.cli import main
from rag_rerank.config import ModeInfo, Settings, resolve_mode
from rag_rerank.i18n import MESSAGES, RagError, resolve_lang, t
from rag_rerank.server import create_app

STATIC = Path(__file__).parent.parent / "src" / "rag_rerank" / "static"


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


# ---------- 메시지 표 ----------
def test_python_tables_have_same_keys_and_placeholders():
    en, ko = MESSAGES["en"], MESSAGES["ko"]
    assert set(en) == set(ko), sorted(set(en) ^ set(ko))
    for key in en:
        assert _placeholders(en[key]) == _placeholders(ko[key]), key
        assert en[key].strip() and ko[key].strip(), key


def _web_tables() -> dict[str, dict[str, str]]:
    src = (STATIC / "i18n.js").read_text(encoding="utf-8")
    tables: dict[str, dict[str, str]] = {}
    current = None
    for line in src.splitlines():
        head = re.match(r"^  (en|ko): \{$", line)
        if head:
            current = head.group(1)
            tables[current] = {}
        elif current and line == "  },":
            current = None
        elif current:
            m = re.match(r'^    "([^"]+)": "((?:[^"\\]|\\.)*)",$', line)
            assert m, f"unparsable line in i18n.js: {line!r}"
            tables[current][m.group(1)] = m.group(2)
    return tables


def test_web_tables_have_same_keys_and_placeholders():
    tables = _web_tables()
    assert set(tables) == {"en", "ko"}
    en, ko = tables["en"], tables["ko"]
    assert len(en) > 80
    assert set(en) == set(ko), sorted(set(en) ^ set(ko))
    for key in en:
        assert set(re.findall(r"\{(\w+)\}", en[key])) == set(re.findall(r"\{(\w+)\}", ko[key])), key


def test_every_key_used_by_the_page_exists_in_the_web_table():
    keys = set(_web_tables()["en"])
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    used = set(re.findall(r'data-i18n(?:-placeholder|-aria)?="([^"]+)"', html))
    used |= set(re.findall(r'\btr\("([\w.]+)"', js))
    used |= {f"scale.{r}" for r in ("vector", "keyword", "hybrid")}
    assert used <= keys, sorted(used - keys)
    assert 'id="lang-select"' in html and 'value="en"' in html and 'value="ko"' in html  # 눈에 보이는 언어 선택


def test_web_page_has_no_hardcoded_korean_outside_the_table():
    html = re.sub(r"<!--.*?-->", "", (STATIC / "index.html").read_text(encoding="utf-8"), flags=re.S)
    assert re.findall(r"[가-힣]+", html) == ["한국어"]  # 언어 선택 항목의 자기 이름만


# ---------- 언어 우선순위 ----------
@pytest.mark.parametrize("cli,env,expected", [
    ("en", {"RAG_LANG": "ko", "LANG": "ko_KR.UTF-8"}, "en"),            # --lang 이 가장 강하다
    ("ko", {"RAG_LANG": "en", "LANG": "en_US.UTF-8"}, "ko"),
    (None, {"RAG_LANG": "ko", "LC_ALL": "en_US.UTF-8"}, "ko"),           # RAG_LANG > LC_ALL
    (None, {"LC_ALL": "ko_KR.UTF-8", "LC_MESSAGES": "en_US", "LANG": "en_US"}, "ko"),
    (None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}, "ko"),  # LC_MESSAGES > LANG
    (None, {"LANG": "ko_KR.UTF-8"}, "ko"),
    (None, {"LANG": "C.UTF-8"}, "en"),                                   # 한국어가 아니면 영어
    (None, {"LANG": "fr_FR.UTF-8"}, "en"),
    (None, {}, "en"),
    (None, {"RAG_LANG": "  ", "LANG": "ko_KR"}, "ko"),                   # 빈 값은 건너뛴다
])
def test_language_priority(cli, env, expected):
    assert resolve_lang(cli, env) == expected


def test_accept_language_parsing():
    assert i18n.from_accept_language("ko-KR,ko;q=0.9,en;q=0.8") == "ko"
    assert i18n.from_accept_language("fr, en-US;q=0.5") == "en"
    assert i18n.from_accept_language("fr") is None and i18n.from_accept_language(None) is None


def test_env_priority_reaches_cli(monkeypatch, capsys):
    monkeypatch.setenv("RAG_LANG", "ko")
    assert main(["index", "/nonexistent-dir", "--mock"]) == 1
    assert "폴더가 아닙니다" in capsys.readouterr().err
    assert main(["index", "/nonexistent-dir", "--mock", "--lang", "en"]) == 1  # 옵션이 환경변수를 이긴다
    assert "Not a folder" in capsys.readouterr().err
    monkeypatch.delenv("RAG_LANG")
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")
    assert main(["index", "/nonexistent-dir", "--mock"]) == 1
    assert "폴더가 아닙니다" in capsys.readouterr().err


def test_invalid_lang_option_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--lang", "fr", "doctor"])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# ---------- 언어 전환이 실제로 바뀌는지 ----------
def test_help_follows_language(capsys):
    for lang, needle in (("en", "Compare RAG retrieval"), ("ko", "RAG 리랭킹 전/후 비교 워크벤치")):
        with pytest.raises(SystemExit) as e:
            main(["--lang", lang, "--help"])
        assert e.value.code == 0
        out = capsys.readouterr().out
        assert needle in out
        assert bool(re.search(r"[가-힣]", out)) == (lang == "ko")  # 영어 도움말에는 한글이 없다


def test_help_language_option_works_after_subcommand(capsys):
    with pytest.raises(SystemExit):
        main(["ask", "--help", "--lang", "ko"])
    assert "터미널에서" not in capsys.readouterr().out  # 하위 명령 도움말은 질문 인자 설명부터 나온다
    with pytest.raises(SystemExit):
        main(["ask", "x", "--lang", "ko", "--help"])
    assert "질문" in capsys.readouterr().out


def test_ask_output_follows_language(capsys):
    assert main(["ask", "중학생도 공유자전거를 탈 수 있나요?", "--mock", "--lang", "en"]) == 0
    out, err = capsys.readouterr()
    assert "Before reranking" in out and "After reranking" in out and "Answer B" in out
    assert "  rank  score" in out and "순위" not in out  # 표 머리글도 영어
    assert "[MOCK] Mock mode was turned on" in err
    assert main(["ask", "중학생도 공유자전거를 탈 수 있나요?", "--mock", "--lang", "ko"]) == 0
    out, err = capsys.readouterr()
    assert "리랭킹 전" in out and "리랭킹 후" in out and "답변 B" in out
    assert "[MOCK] --mock 플래그" in err


def test_eval_table_follows_language_but_json_does_not(capsys):
    assert main(["eval", "--mock", "--lang", "en"]) == 0
    assert "Score card" in capsys.readouterr().out
    assert main(["eval", "--mock", "--lang", "ko"]) == 0
    assert "성적표" in capsys.readouterr().out
    outs = {}
    for lang in ("en", "ko"):
        assert main(["eval", "--mock", "--json", "--lang", lang]) == 0
        outs[lang] = capsys.readouterr().out
    assert outs["en"] == outs["ko"]  # 기계용 출력은 언어와 무관하게 같다
    golden = json.loads((Path(__file__).parent / "golden" / "eval_mock.json").read_text(encoding="utf-8"))
    assert json.loads(outs["en"])["table"] == golden["table"]


def test_ask_json_is_language_independent(capsys):
    outs = []
    for lang in ("en", "ko"):
        assert main(["ask", "중학생도 공유자전거를 탈 수 있나요?", "--mock", "--json", "--lang", lang]) == 0
        data = json.loads(capsys.readouterr().out)
        data.pop("timings_ms", None)
        outs.append(data)
    assert outs[0] == outs[1]


# ---------- 라이브러리 오류(code + params) ----------
def test_library_errors_are_coded():
    from rag_rerank.chunking import ChunkOptions, chunk_documents, Document
    from rag_rerank.retrieve import search

    with pytest.raises(RagError) as e:
        chunk_documents([Document("a", "a.md", "x")], ChunkOptions("nope"))
    assert (e.value.code, e.value.params) == ("unknown_chunking", {"name": "nope"})
    assert str(e.value) == "Unknown chunking strategy: nope"  # str() 은 영어
    assert e.value.render("ko") == "알 수 없는 청킹 방식: nope"
    with pytest.raises(RagError) as e2:
        search(None, "nope", "q", None, 3)
    assert e2.value.code == "unknown_retriever" and e2.value.render("ko") == "알 수 없는 검색기: nope"


def test_reranker_not_ready_error(engine, make_index):
    import asyncio

    from rag_rerank.evaluate import evaluate

    engine.reranker_for = lambda index: (None, "downloading")
    with pytest.raises(RagError) as e:
        asyncio.run(evaluate(engine, make_index(), [{"q": "q", "gold": "g"}]))
    assert e.value.render("en") == "The reranker is not ready yet (state: downloading)."
    assert "아직 준비되지 않았습니다" in e.value.render("ko")


def test_every_error_code_has_both_messages():
    for lang in ("en", "ko"):
        for key in MESSAGES[lang]:
            if key.startswith("err."):
                params = {p: "x" for p in _placeholders(MESSAGES[lang][key])}
                assert RagError(key[4:], **params).render(lang)


def test_notice_events_render_per_language():
    ev = i18n.notice("time_limit_cut", seconds=120)
    assert ev["text"] == "The answer was cut off after 120s."
    assert i18n.render_notice(ev, "ko") == {"type": "notice", "text": "120초를 넘어 답을 중간에 멈췄습니다."}


def test_mode_info_renders_both_languages(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "auto")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    info = resolve_mode(Settings.from_env())
    reason_en, hints_en = info.render("en")
    reason_ko, hints_ko = info.render("ko")
    assert reason_en.startswith("Could not connect to Ollama") and reason_ko.startswith("Ollama(")
    assert hints_en[0] != hints_ko[0] and hints_en[1:] == hints_ko[1:] == ["ollama pull bge-m3", "ollama pull qwen3:4b"]
    assert ModeInfo("mock", "plain").render("ko") == ("plain", [])


def test_forced_real_mode_error_follows_language(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "real")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    with pytest.raises(SystemExit) as en:
        resolve_mode(Settings.from_env(), lang="en")
    with pytest.raises(SystemExit) as ko:
        resolve_mode(Settings.from_env(), lang="ko")
    assert "[real mode unavailable]" in str(en.value) and "[real 모드 불가]" in str(ko.value)


# ---------- 서버: 오류 문구는 요청 언어로, JSON 모양은 그대로 ----------
def test_api_errors_follow_request_language(tmp_path):
    missing = str(tmp_path / "nope")
    with TestClient(create_app(), base_url="http://127.0.0.1") as c:
        r = c.post("/api/index", json={"folder": missing})  # 서버 기본(RAG_LANG=en)
        assert r.status_code == 400 and r.json() == {"detail": f"Not a folder: {missing}"}
        r = c.post("/api/index?lang=ko", json={"folder": missing})
        assert r.status_code == 400 and r.json() == {"detail": f"폴더가 아닙니다: {missing}"}
        r = c.post("/api/index", json={"folder": missing}, headers={"Accept-Language": "ko-KR,ko;q=0.9"})
        assert r.json()["detail"].startswith("폴더가 아닙니다")
        r = c.post("/api/index?lang=en", json={"folder": missing}, headers={"Accept-Language": "ko"})  # 쿼리가 헤더를 이긴다
        assert r.json()["detail"].startswith("Not a folder")
        assert c.post("/api/index?lang=ko", json={}).json() == {"detail": "folder 또는 files 중 하나를 보내세요."}
        assert c.post("/api/compare?lang=ko", json={"question": "q", "collection": "mine"}).json() == {
            "detail": "먼저 '내 문서' 탭에서 문서를 색인하세요."}
        assert c.post("/api/compare", json={"question": "q", "collection": "mine"}).json() == {
            "detail": "Index your documents in the 'My documents' tab first."}


def test_api_unreadable_file_is_400_not_500(tmp_path, monkeypatch):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("hello", encoding="utf-8")
    real = Path.read_text

    def deny(self, *a, **k):
        if self.name == "a.md":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", deny)
    with TestClient(create_app(), base_url="http://127.0.0.1", raise_server_exceptions=False) as c:
        r = c.post("/api/index?lang=ko", json={"folder": str(d)})
    assert r.status_code == 400 and "에 접근할 수 없습니다" in r.json()["detail"]


def test_status_mode_reason_follows_language_and_keeps_keys():
    with TestClient(create_app(mode_info=resolve_mode(Settings.from_env(), force_mock=True)), base_url="http://127.0.0.1") as c:
        en = c.get("/api/status?lang=en").json()
        ko = c.get("/api/status?lang=ko").json()
    assert en["mode_reason"].startswith("Mock mode was turned on") and ko["mode_reason"].startswith("--mock 플래그")
    assert set(en) == set(ko)


def test_server_default_language_comes_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_LANG", "ko")
    with TestClient(create_app(), base_url="http://127.0.0.1") as c:
        assert c.post("/api/index", json={}).json()["detail"].startswith("folder 또는")


def test_compare_json_is_language_independent():
    body = {"question": "중학생도 공유자전거를 탈 수 있나요?"}
    with TestClient(create_app(), base_url="http://127.0.0.1") as c:
        a = c.post("/api/compare?lang=en", json=body).json()
        b = c.post("/api/compare?lang=ko", json=body).json()
    for r in (a, b):
        r.pop("timings_ms")
    assert a == b


def test_web_page_and_script_are_served():
    with TestClient(create_app(), base_url="http://127.0.0.1") as c:
        assert 'id="lang-select"' in c.get("/").text
        assert "MESSAGES" in c.get("/static/i18n.js").text
