"""Screen language (English / Korean): one message table, language resolution, and coded errors.

Language priority for the CLI: ``--lang en|ko`` > ``RAG_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.
A value that does not start with ``ko`` means English. The web UI keeps its own table in
``static/i18n.js`` (same idea, browser side); ``tests/test_i18n.py`` checks both tables.

Library code never builds user-facing sentences. It raises :class:`RagError` with a ``code`` and
``params``; the CLI and the server render it in the chosen language. ``str(error)`` is English.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

LANGS = ("en", "ko")
DEFAULT_LANG = "en"
LANG_ENV_VARS = ("RAG_LANG", "LC_ALL", "LC_MESSAGES", "LANG")

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # ----- errors (RagError codes) -----
        "err.not_a_folder": "Not a folder: {path}",
        "err.no_docs": "No .md or .txt documents were found.",
        "err.file_access": "Cannot access {path}: {reason}",
        "err.os": "File system error: {reason}",
        "err.unknown_chunking": "Unknown chunking strategy: {name}",
        "err.unknown_retriever": "Unknown retriever: {name}",
        "err.reranker_not_ready": "The reranker is not ready yet (state: {state}).",
        "err.reranker_failed": "Reranker setup failed: {error}",
        "err.need_docs_first": "Index your documents in the 'My documents' tab first.",
        "err.need_folder_or_files": "Send either folder or files.",
        "err.bad_request": "Invalid request.",
        # ----- mode (config.py) -----
        "mode.mock_forced": "Mock mode was turned on by the --mock flag (or RAG_MODE=mock).",
        "mode.ollama_down": "Could not connect to Ollama ({url}) ({error}).",
        "mode.models_missing": "Ollama is missing models: {models}",
        "mode.real_ready": "Ollama and the models are ready, running in real mode.",
        "mode.fallback": " So running in mock mode.",
        "mode.real_unavailable": "[real mode unavailable] {reason}\n  Fix: {fix}",
        "hint.install_ollama": "Install and start Ollama from https://ollama.com",
        # ----- notices streamed with answers -----
        "notice.reranker_wait": "The reranker model is still loading, so answer B can be made a little later.",
        "notice.time_limit_fallback": "Thinking ran past {seconds}s and was stopped; showing the evidence sentences instead.",
        "notice.time_limit_cut": "The answer was cut off after {seconds}s.",
        "no_evidence_answer": "No evidence was found in the documents.",
        # ----- server -----
        "server.no_model_mock": "Mock mode has no model to download.",
        "scale.vector": "Cosine similarity (-1 to 1)",
        "scale.keyword": "BM25 score (no upper bound)",
        "scale.hybrid": "RRF score (sum of 1/(60+rank))",
        "scale.after": "Reranker score (0 to 1)",
        # ----- CLI: help -----
        "help.description": "Compare RAG retrieval before and after reranking.",
        "help.mock": "Run with fake (deterministic) models, no Ollama or downloads",
        "help.lang": "Screen language: en or ko (default: RAG_LANG, then LC_ALL/LC_MESSAGES/LANG, else English)",
        "help.no_browser": "Do not open a browser automatically (just print the address)",
        "help.serve": "Web UI (default)",
        "help.ask": "Ask one question in the terminal",
        "help.eval": "Score card (hit@1 and MRR before/after)",
        "help.index": "Index a folder and preview its chunks",
        "help.doctor": "Check whether real mode is ready",
        "help.host": "Address to bind (default 127.0.0.1)",
        "help.port": "Port (default RAG_PORT or 8765)",
        "help.question": "The question to ask",
        "help.retriever": "First-stage retriever",
        "help.docs": "Search this folder's .md/.txt files instead of the samples",
        "help.answer": "Also make answers A/B in real mode (slow: one answer takes tens of seconds to 2 minutes; "
                       "after 2 minutes the evidence sentences are used instead)",
        "help.json": "Print the result as JSON",
        "help.folder": "Folder to index (default: samples/)",
        "help.show": "Number of chunks to preview",
        "help.download": "Download the reranker model now if it is missing",
        "help.chunking": "Chunking strategy (default paragraph)",
        "help.no_title_prefix": "Do not prepend [document > heading] to each chunk",
        "help.top_k": "Number of first-stage candidates (default 8)",
        # ----- CLI: output -----
        "cli.model_download": "Downloading the reranker model: {repo} (about 588MB, first time only)",
        "cli.model_download_doctor": "Downloading the reranker model: {repo} (about 588MB)",
        "cli.question": "Question: {question}",
        "cli.settings": "Retriever: {retriever} | top-k {top_k} | chunking {strategy} ({prefix}) | index {chunks} chunks",
        "cli.prefix_on": "with titles",
        "cli.prefix_off": "no titles",
        "cli.before": "Before reranking",
        "cli.after": "After reranking",
        "cli.head": "  rank  score     move  ",
        "cli.head_prev": "was    ",
        "cli.head_tail": "source / text",
        "cli.ratio": "Gap between #1 and #2: #2 score is {before} of #1 (stage 1) -> {after} (reranker); smaller = clearer winner",
        "cli.no_evidence": "Note: the top reranker score is below the threshold, so this counts as 'no evidence'.",
        "cli.timings": "Time (ms): embed {embed} | stage 1 {retrieve} | rerank {rerank}",
        "cli.answer_a": "Answer A (top 3 evidence before reranking)",
        "cli.answer_b": "Answer B (top 3 evidence after reranking)",
        "cli.thinking": "  (thinking... {tokens} tokens)",
        "cli.cites": "  Citations: {cites}",
        "cli.none": "none",
        "cli.eval_title": "Score card - {n} questions, top-k {top_k}, reranker {reranker}",
        "cli.eval_head": "  retriever  hit@1 before->after  MRR before->after  reordered questions",
        "cli.eval_changed": "(question x retriever) pairs reordered: {count}",
        "cli.index_summary": "{docs} documents -> {chunks} chunks ({strategy})",
        # ----- CLI: doctor -----
        "doctor.sqlite": "SQLite extension + sqlite-vec",
        "doctor.sqlite_fix": "Run with the Python that uv installs (uv run ...)",
        "doctor.ollama": "Ollama connection",
        "doctor.ollama_fix": "Install Ollama from https://ollama.com, then run `ollama serve`",
        "doctor.model": "Model {name}",
        "doctor.present": "present",
        "doctor.absent": "missing",
        "doctor.reranker_file": "Reranker model file",
        "doctor.reranker_absent": "missing (downloaded automatically on first run)",
        # ----- CLI: serve -----
        "serve.port_busy": "Port {port} is already in use. Run on another port:\n"
                           "  uv run rag-rerank serve --port {next_port}   (or RAG_PORT={next_port})",
        "serve.url": "UI: {url}  (Ctrl+C to quit)",
        "serve.lan": "From another machine open http://<this computer's IP>:{port} (with Docker, publish the port with -p {port}:{port}).\n"
                     "Warning: other machines on the same network can reach this server. Use trusted networks only.",
    },
    "ko": {
        "err.not_a_folder": "폴더가 아닙니다: {path}",
        "err.no_docs": ".md 또는 .txt 문서를 찾지 못했습니다.",
        "err.file_access": "{path}에 접근할 수 없습니다 ({reason})",
        "err.os": "파일 시스템 오류: {reason}",
        "err.unknown_chunking": "알 수 없는 청킹 방식: {name}",
        "err.unknown_retriever": "알 수 없는 검색기: {name}",
        "err.reranker_not_ready": "리랭커가 아직 준비되지 않았습니다(상태: {state}).",
        "err.reranker_failed": "리랭커 준비 실패: {error}",
        "err.need_docs_first": "먼저 '내 문서' 탭에서 문서를 색인하세요.",
        "err.need_folder_or_files": "folder 또는 files 중 하나를 보내세요.",
        "err.bad_request": "잘못된 요청입니다.",
        "mode.mock_forced": "--mock 플래그(또는 RAG_MODE=mock)로 mock 모드를 켰습니다.",
        "mode.ollama_down": "Ollama({url})에 연결하지 못했습니다({error}).",
        "mode.models_missing": "Ollama에 모델이 없습니다: {models}",
        "mode.real_ready": "Ollama와 모델이 준비되어 실제 모드로 실행합니다.",
        "mode.fallback": " 그래서 mock 모드로 실행합니다.",
        "mode.real_unavailable": "[real 모드 불가] {reason}\n  해결: {fix}",
        "hint.install_ollama": "https://ollama.com 에서 Ollama 설치 후 실행",
        "notice.reranker_wait": "리랭커 모델을 준비하는 중이라 답변 B는 잠시 뒤에 만들 수 있습니다.",
        "notice.time_limit_fallback": "생각이 {seconds}초를 넘어 중단하고, 근거 문장을 그대로 옮긴 답으로 대신합니다.",
        "notice.time_limit_cut": "{seconds}초를 넘어 답을 중간에 멈췄습니다.",
        "no_evidence_answer": "문서에서 근거를 찾지 못했습니다.",
        "server.no_model_mock": "mock 모드에는 내려받을 모델이 없습니다.",
        "scale.vector": "코사인 유사도(-1~1)",
        "scale.keyword": "BM25 점수(상한 없음)",
        "scale.hybrid": "RRF 점수(Σ 1/(60+순위))",
        "scale.after": "리랭커 점수(0~1)",
        "help.description": "RAG 리랭킹 전/후 비교 워크벤치",
        "help.mock": "Ollama·모델 없이 가짜(결정적) 모델로 실행",
        "help.lang": "화면 언어: en 또는 ko (기본: RAG_LANG, 그다음 LC_ALL/LC_MESSAGES/LANG, 없으면 영어)",
        "help.no_browser": "브라우저를 자동으로 열지 않음(주소만 출력)",
        "help.serve": "웹 화면(기본)",
        "help.ask": "터미널에서 질문 한 번",
        "help.eval": "성적표(hit@1·MRR 전/후)",
        "help.index": "폴더 색인해 보기",
        "help.doctor": "실제 모드 준비 상태 진단",
        "help.host": "바인딩할 주소(기본 127.0.0.1)",
        "help.port": "포트(기본 RAG_PORT 또는 8765)",
        "help.question": "물어볼 질문",
        "help.retriever": "1단계 검색기",
        "help.docs": "샘플 대신 이 폴더의 .md/.txt 로 검색",
        "help.answer": "실제 모드에서도 답변 A/B 생성(느림: 생각 전용 모델이라 답 하나에 수십 초~2분, "
                       "2분을 넘으면 근거 문장을 옮긴 답으로 대신)",
        "help.json": "결과를 JSON 으로 출력",
        "help.folder": "색인할 폴더(생략하면 samples/)",
        "help.show": "미리 볼 청크 수",
        "help.download": "리랭커 모델이 없으면 지금 내려받기",
        "help.chunking": "청킹 방식(기본 paragraph)",
        "help.no_title_prefix": "청크 앞에 [문서 > 절 제목]을 붙이지 않음",
        "help.top_k": "1단계 후보 수(기본 8)",
        "cli.model_download": "리랭커 모델 내려받는 중: {repo} (약 588MB, 처음 한 번만)",
        "cli.model_download_doctor": "리랭커 모델 내려받는 중: {repo} (약 588MB)",
        "cli.question": "질문: {question}",
        "cli.settings": "검색기: {retriever} · top-k {top_k} · 청킹 {strategy}({prefix}) · 색인 {chunks}개 청크",
        "cli.prefix_on": "제목 붙임",
        "cli.prefix_off": "제목 없음",
        "cli.before": "리랭킹 전",
        "cli.after": "리랭킹 후",
        "cli.head": "  순위  점수      이동  ",
        "cli.head_prev": "전순위  ",
        "cli.head_tail": "출처 / 내용",
        "cli.ratio": "1위와 2위 차이: 2위 점수가 1위의 {before}(1단계) → {after}(리랭커) — 작을수록 1위가 확실히 앞섬",
        "cli.no_evidence": "※ 리랭커 최고 점수가 문턱값보다 낮아 '근거 부족'으로 판단했습니다.",
        "cli.timings": "시간(ms): 임베딩 {embed} · 1단계 {retrieve} · 리랭킹 {rerank}",
        "cli.answer_a": "답변 A (리랭킹 전 상위 3개 근거)",
        "cli.answer_b": "답변 B (리랭킹 후 상위 3개 근거)",
        "cli.thinking": "  (생각 중… {tokens}토큰)",
        "cli.cites": "  인용: {cites}",
        "cli.none": "없음",
        "cli.eval_title": "성적표 — 질문 {n}개, top-k {top_k}, 리랭커 {reranker}",
        "cli.eval_head": "  검색기    hit@1 전→후    MRR 전→후        순서 바뀐 질문",
        "cli.eval_changed": "순서가 바뀐 (질문 × 검색기) 조합: {count}",
        "cli.index_summary": "문서 {docs}개 → 청크 {chunks}개 ({strategy})",
        "doctor.sqlite": "SQLite 확장 + sqlite-vec",
        "doctor.sqlite_fix": "uv 가 설치한 Python 으로 실행(uv run …)",
        "doctor.ollama": "Ollama 연결",
        "doctor.ollama_fix": "https://ollama.com 설치 후 `ollama serve`",
        "doctor.model": "모델 {name}",
        "doctor.present": "있음",
        "doctor.absent": "없음",
        "doctor.reranker_file": "리랭커 모델 파일",
        "doctor.reranker_absent": "없음(첫 실행 때 자동으로 받음)",
        "serve.port_busy": "포트 {port} 를 이미 다른 프로그램이 쓰고 있습니다. 다른 포트로 실행하세요:\n"
                           "  uv run rag-rerank serve --port {next_port}   (또는 RAG_PORT={next_port})",
        "serve.url": "화면: {url}  (끝내려면 Ctrl+C)",
        "serve.lan": "다른 기기에서는 http://<이 컴퓨터 IP>:{port} 로 접속하세요(도커라면 -p {port}:{port} 로 포트를 열어 두세요).\n"
                     "주의: 같은 네트워크의 다른 기기에서도 이 서버에 접속할 수 있습니다. 믿을 수 있는 네트워크에서만 쓰세요.",
    },
}


def normalize_lang(value: str | None) -> str | None:
    """'ko_KR.UTF-8' -> 'ko', 'en_US' / 'C.UTF-8' -> 'en', empty or None -> None."""
    if value is None or not value.strip():
        return None
    return "ko" if value.strip().lower().startswith("ko") else "en"


def resolve_lang(cli_lang: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """CLI flag > RAG_LANG > LC_ALL > LC_MESSAGES > LANG; anything not Korean is English."""
    if cli_lang in LANGS:
        return str(cli_lang)
    env = os.environ if env is None else env
    for name in LANG_ENV_VARS:
        found = normalize_lang(env.get(name))
        if found:
            return found
    return DEFAULT_LANG


def from_accept_language(header: str | None) -> str | None:
    """First language of an Accept-Language header that we support (ko or en), else None."""
    for part in (header or "").split(","):
        tag = part.split(";")[0].strip().lower()
        if tag.startswith("ko"):
            return "ko"
        if tag.startswith("en"):
            return "en"
    return None


def t(lang: str, message_id: str, **params: object) -> str:
    """Look up a message in ``lang`` (falls back to English) and fill in the placeholders."""
    table = MESSAGES.get(lang, MESSAGES[DEFAULT_LANG])
    template = table.get(message_id) or MESSAGES[DEFAULT_LANG][message_id]
    return template.format(**params) if params else template


class RagError(Exception):
    """A user-facing error: ``code`` names the message, ``params`` fill it in. ``str()`` is English."""

    def __init__(self, code: str, **params: object) -> None:
        self.code = code
        self.params = params
        super().__init__(t(DEFAULT_LANG, f"err.{code}", **params))

    def render(self, lang: str) -> str:
        return t(lang, f"err.{self.code}", **self.params)


def os_error_to_rag_error(error: OSError) -> RagError:
    """Turn PermissionError / FileNotFoundError / ... into a coded error without a traceback."""
    reason = error.strerror or type(error).__name__
    if error.filename:
        return RagError("file_access", path=str(error.filename), reason=reason)
    return RagError("os", reason=reason)


def notice(code: str, **params: object) -> dict:
    """Answer-stream notice event. ``text`` is English; ``code``/``params`` let CLI and server re-render it."""
    return {"type": "notice", "text": t(DEFAULT_LANG, f"notice.{code}", **params), "code": f"notice.{code}", "params": params}


def render_notice(event: dict, lang: str) -> dict:
    """Return the event with ``text`` in ``lang`` and the internal ``code``/``params`` removed (JSON shape unchanged)."""
    if event.get("type") != "notice" or "code" not in event:
        return event
    return {"type": "notice", "text": t(lang, event["code"], **event.get("params", {}))}
