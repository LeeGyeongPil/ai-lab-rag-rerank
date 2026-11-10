"""명령줄: rag-rerank [serve|ask|eval|index|doctor] [--mock]

  uv run rag-rerank                      웹 화면 띄우기(기본, http://127.0.0.1:8765)
  uv run rag-rerank --mock --no-browser   브라우저를 열지 않고 주소만 출력(화면 없는 서버·컨테이너)
  uv run rag-rerank ask "질문" --mock     터미널에서 전/후 비교 + 답변 A/B
  uv run rag-rerank eval --mock --json    성적표(hit@1·MRR 전/후)
  uv run rag-rerank index 내폴더          내 문서 폴더 색인해 보기
  uv run rag-rerank doctor               실제 모드 준비 상태 진단
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sqlite3
import sys
import threading
import webbrowser
from contextlib import aclosing, closing
from pathlib import Path

from .chunking import STRATEGIES, ChunkOptions
from .config import RERANKER_REPO, Settings, has_model, probe_ollama, resolve_mode
from .evaluate import evaluate, load_questions
from .pipeline import Engine, answer_stream, compare, create_engine
from .retrieve import RETRIEVERS
from .store import build_index, connect, load_folder


# ---------- 공통 ----------
def _engine(args) -> tuple[Settings, Engine]:
    settings = Settings.from_env()
    mode_info = resolve_mode(settings, force_mock=getattr(args, "mock", False))
    print(f"[{mode_info.mode.upper()}] {mode_info.reason}", file=sys.stderr)
    for hint in mode_info.hints:
        print(f"  · {hint}", file=sys.stderr)
    return settings, create_engine(settings, mode_info)


def _prepare_reranker(engine: Engine) -> None:
    """실제 모드: 리랭커가 없으면 지금 내려받는다(터미널 진행 막대, 첫 실행에 한 번 약 588MB)."""
    if engine.hub is not None and engine.hub.get() is None:
        if engine.hub.local_paths() is None:
            print(f"리랭커 모델 내려받는 중: {RERANKER_REPO} (약 588MB, 처음 한 번만)", file=sys.stderr)
        engine.hub.prepare_blocking(show_progress=True)


class _InputError(Exception):
    """사용자가 준 폴더가 잘못됨(없음·문서 없음) — 트레이스백 없이 한 줄로 알린다."""


async def _build(engine: Engine, args, docs_dir: str | None):
    opts = ChunkOptions(args.chunking, not args.no_title_prefix)
    if docs_dir:
        try:
            docs = load_folder(Path(docs_dir))
        except ValueError as e:  # '폴더가 아닙니다: …'
            raise _InputError(str(e)) from e
        if not docs:  # 웹 API(/api/index)와 같은 문구
            raise _InputError(".md 또는 .txt 문서를 찾지 못했습니다.")
        return await build_index("mine", docs, opts, engine.embedder, engine.settings.cache_dir)
    return await build_index("samples", load_folder(engine.settings.samples_dir), opts, engine.embedder,
                             engine.settings.cache_dir)


def _short(text: str, n: int = 38) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _move(m: int) -> str:
    return f"▲{m}" if m > 0 else (f"▼{-m}" if m < 0 else "–")


def _print_table(title: str, scale: str, rows: list[dict], after: bool) -> None:
    print(f"\n{title}  [{scale}]")
    head = "  순위  점수      이동  " + ("전순위  " if after else "") + "출처 / 내용"
    print(head)
    for r in rows:
        extra = f"{r['rank_before']:>4}    " if after else ""
        mv = _move(r["move"]) if "move" in r else ""
        print(f"  {r['rank']:>3}   {r['score']:<8}  {mv:<4}  {extra}{r['source']} | {_short(r['text'])}")


# ---------- ask ----------
async def _ask(args) -> int:
    settings, engine = _engine(args)
    try:
        _prepare_reranker(engine)
        try:
            index = await _build(engine, args, args.docs)
        except _InputError as e:
            print(str(e), file=sys.stderr)
            return 1
        result = await compare(engine, index, args.question, args.retriever, args.top_k)
        want_answer = args.answer or engine.mode == "mock"  # mock 은 답이 즉시 나오므로 늘 보여 준다
        answers = {}
        if not args.json:
            c = result["chunking"]
            print(f"질문: {result['question']}")
            print(f"검색기: {args.retriever} · top-k {args.top_k} · 청킹 {c['strategy']}"
                  f"({'제목 붙임' if c['title_prefix'] else '제목 없음'}) · 색인 {len(index.chunks)}개 청크")
            _print_table("리랭킹 전", result["scales"]["before"], result["before"], after=False)
            if result["after"] is not None:
                _print_table("리랭킹 후", result["scales"]["after"], result["after"], after=True)
            ratio = result["top2_ratio"]
            if ratio["before"] is not None and ratio["after"] is not None:
                # 2위 점수가 1위의 몇 %인지: 작을수록 1위가 나머지보다 확실히 앞선다(화면 '점수 분포'와 같은 값)
                print(f"\n1위와 2위 차이: 2위 점수가 1위의 {ratio['before']:.0%}(1단계) → {ratio['after']:.0%}(리랭커)"
                      " — 작을수록 1위가 확실히 앞섬")
            if result["no_evidence"]:
                print("\n※ 리랭커 최고 점수가 문턱값보다 낮아 '근거 부족'으로 판단했습니다.")
            t = result["timings_ms"]
            print(f"\n시간(ms): 임베딩 {t['embed']} · 1단계 {t['retrieve']} · 리랭킹 {t['rerank']}")
        if want_answer:
            for side, label in (("A", "답변 A (리랭킹 전 상위 3개 근거)"), ("B", "답변 B (리랭킹 후 상위 3개 근거)")):
                text = ""
                async with aclosing(answer_stream(engine, index, result, side)) as stream:
                    async for ev in stream:
                        if ev["type"] == "thinking" and engine.mode == "real" and not args.json:
                            print(f"\r  (생각 중… {ev['tokens']}토큰)", end="", file=sys.stderr, flush=True)
                        elif ev["type"] == "notice" and not args.json:
                            print(f"\n  ※ {ev['text']}", file=sys.stderr)
                        elif ev["type"] == "end":
                            text = ev["text"]
                            answers[side] = {"text": text, "citations": ev["citations"]}
                if not args.json:
                    if engine.mode == "real":
                        print("", file=sys.stderr)
                    print(f"\n{label}: {text}")
                    cites = ", ".join(f"[{c['n']}]={c['chunk_id']}" for c in answers[side]["citations"])
                    print(f"  인용: {cites or '없음'}")
        if args.json:
            if answers:
                result["answers"] = answers
            print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0
    finally:
        await engine.aclose()


# ---------- eval ----------
async def _eval(args) -> int:
    settings, engine = _engine(args)
    try:
        _prepare_reranker(engine)
        index = await _build(engine, args, None)
        result = await evaluate(engine, index, load_questions(settings.samples_dir / "questions.json"), args.top_k)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=1))
            return 0
        n = result["n_questions"]
        print(f"성적표 — 질문 {n}개, top-k {args.top_k}, 리랭커 {result['reranker']}")
        print("  검색기    hit@1 전→후    MRR 전→후        순서 바뀐 질문")
        for name, r in result["table"].items():
            print(f"  {name:<8}  {r['hit1_before']:>2}/{n} → {r['hit1_after']:>2}/{n}   "
                  f"{r['mrr_before']:.2f} → {r['mrr_after']:.2f}      {r['orders_changed']}")
        print(f"\n순서가 바뀐 (질문 × 검색기) 조합: {result['orders_changed']}")
        return 0
    finally:
        await engine.aclose()


# ---------- index ----------
async def _index(args) -> int:
    settings, engine = _engine(args)
    try:
        try:
            index = await _build(engine, args, args.folder)
        except _InputError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"문서 {len({c.doc for c in index.chunks})}개 → 청크 {len(index.chunks)}개 ({index.options.strategy})")
        for c in index.chunks[: args.show]:
            print(f"  {c.chunk_id:<24} {_short(c.content, 60)}")
        return 0
    finally:
        await engine.aclose()


# ---------- doctor ----------
def _doctor(args) -> int:
    settings = Settings.from_env()
    checks: list[tuple[str, bool, str, str]] = []  # (항목, 통과?, 설명, 해결 방법)
    checks.append(("Python 3.14+", sys.version_info >= (3, 14), sys.version.split()[0], "uv python install 3.14"))
    try:
        with closing(connect(":memory:")) as db:
            ver = db.execute("select vec_version()").fetchone()[0]
        checks.append(("SQLite 확장 + sqlite-vec", True, ver, ""))
    except (AttributeError, sqlite3.Error) as e:
        checks.append(("SQLite 확장 + sqlite-vec", False, str(e), "uv 가 설치한 Python 으로 실행(uv run …)"))
    ok, names, err = probe_ollama(settings)
    checks.append(("Ollama 연결", ok, settings.ollama_base_url if ok else err, "https://ollama.com 설치 후 `ollama serve`"))
    for m in (settings.embed_model, settings.chat_model):
        checks.append((f"모델 {m}", ok and has_model(names, m), "있음" if ok and has_model(names, m) else "없음", f"ollama pull {m}"))
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        checks.append(("onnxruntime·tokenizers", True, f"onnxruntime {onnxruntime.__version__}", ""))
    except ImportError as e:
        checks.append(("onnxruntime·tokenizers", False, str(e), "uv sync"))
    from .reranker import RerankerHub

    hub = RerankerHub(settings)
    if args.download and hub.local_paths() is None:
        print(f"리랭커 모델 내려받는 중: {RERANKER_REPO} (약 588MB)", file=sys.stderr)
        hub.prepare_blocking(show_progress=True)
    have = hub.local_paths() is not None
    checks.append(("리랭커 모델 파일", have, "있음" if have else "없음(첫 실행 때 자동으로 받음)",
                   "uv run rag-rerank doctor --download"))
    for name, passed, detail, fix in checks:
        print(f"[{'OK' if passed else '!!'}] {name}: {detail}" + ("" if passed else f"  → {fix}"))
    return 0 if all(c[1] for c in checks) else 1


# ---------- serve ----------
WILDCARD_HOSTS = ("0.0.0.0", "::")  # '모든 주소에서 받기' — 브라우저 주소로는 쓸 수 없다


def browser_url(host: str, port: int) -> str:
    """브라우저에 넣을 수 있는 주소. 0.0.0.0·:: 은 '모든 주소'라는 뜻일 뿐이라 이 컴퓨터 주소로 바꿔 보여 준다."""
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "[::1]"
    return f"http://{host}:{port}"


def _serve(args) -> int:
    import uvicorn

    from .server import create_app

    settings = Settings.from_env()
    mode_info = resolve_mode(settings, force_mock=getattr(args, "mock", False))
    port = args.port or settings.port
    url = browser_url(args.host, port)
    probe_host = "127.0.0.1" if args.host in WILDCARD_HOSTS else args.host
    with closing(socket.socket()) as probe:  # 포트가 이미 쓰이고 있으면 알아듣기 쉬운 말로 알린다
        if probe.connect_ex((probe_host, port)) == 0:
            print(f"포트 {port} 를 이미 다른 프로그램이 쓰고 있습니다. 다른 포트로 실행하세요:\n"
                  f"  uv run rag-rerank serve --port {port + 1}   (또는 RAG_PORT={port + 1})", file=sys.stderr)
            return 1
    # flush=True: 터미널이 아닌 곳(도커 로그, 파이프)으로 내보낼 때도 이 줄들이 바로 보이게 한다
    print(f"[{mode_info.mode.upper()}] {mode_info.reason}", flush=True)
    for hint in mode_info.hints:
        print(f"  · {hint}", flush=True)
    print(f"화면: {url}  (끝내려면 Ctrl+C)", flush=True)
    timer = None
    # --no-browser 는 'rag-rerank --no-browser' 와 'rag-rerank serve --no-browser' 어느 쪽에 써도 된다
    if not getattr(args, "no_browser", False) and not os.environ.get("RAG_NO_BROWSER"):
        timer = threading.Timer(1.5, webbrowser.open, args=(url,))  # 서버가 뜰 시간을 조금 준다
        timer.daemon = True
        timer.start()
    try:
        # --host 로 다른 주소를 직접 고른 경우에만 그 이름을 Host 허용 목록에 더한다(기본은 127.0.0.1·localhost)
        if args.host in ("127.0.0.1", "localhost"):
            extra = []
        elif args.host in WILDCARD_HOSTS:
            # 모든 주소로 열기는 "다른 기기에서도 접속"을 직접 고른 것이므로 Host 검사를 푼다(접속 IP를 미리 알 수 없음)
            extra = ["*"]
            print(f"다른 기기에서는 http://<이 컴퓨터 IP>:{port} 로 접속하세요(도커라면 -p {port}:{port} 로 포트를 열어 두세요).\n"
                  "주의: 같은 네트워크의 다른 기기에서도 이 서버에 접속할 수 있습니다. 믿을 수 있는 네트워크에서만 쓰세요.",
                  flush=True)
        else:
            extra = [args.host]
        uvicorn.run(create_app(settings, mode_info, extra_hosts=extra), host=args.host, port=port, log_level="warning")
    finally:
        if timer is not None:
            timer.cancel()
    return 0


# ---------- 인자 ----------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: 'rag-rerank --mock ask …' 와 'rag-rerank ask … --mock' 둘 다 되게 한다
    common.add_argument("--mock", action="store_true", default=argparse.SUPPRESS,
                        help="Ollama·모델 없이 가짜(결정적) 모델로 실행")

    def chunk_args(p):
        p.add_argument("--chunking", choices=STRATEGIES, default="paragraph", help="청킹 방식(기본 paragraph)")
        p.add_argument("--no-title-prefix", action="store_true", help="청크 앞에 [문서 > 절 제목]을 붙이지 않음")
        p.add_argument("--top-k", type=int, default=8, help="1단계 후보 수(기본 8)")

    parser = argparse.ArgumentParser(prog="rag-rerank", parents=[common],
                                     description="RAG 리랭킹 전/후 비교 워크벤치")
    # 하위 명령 없이 'rag-rerank --no-browser' 로도 쓸 수 있게(화면 없는 서버·컨테이너·SSH)
    parser.add_argument("--no-browser", action="store_true", default=argparse.SUPPRESS,
                        help="브라우저를 자동으로 열지 않음(주소만 출력)")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("serve", parents=[common], help="웹 화면(기본)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true", default=argparse.SUPPRESS,
                   help="브라우저를 자동으로 열지 않음(주소만 출력)")

    p = sub.add_parser("ask", parents=[common], help="터미널에서 질문 한 번")
    p.add_argument("question")
    p.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    p.add_argument("--docs", help="샘플 대신 이 폴더의 .md/.txt 로 검색")
    p.add_argument("--answer", action="store_true", help="실제 모드에서도 답변 A/B 생성(느림: 생각 전용 모델이라 답 하나에 수십 초~2분, "
                        "2분을 넘으면 근거 문장을 옮긴 답으로 대신)")
    p.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    chunk_args(p)

    p = sub.add_parser("eval", parents=[common], help="성적표(hit@1·MRR 전/후)")
    p.add_argument("--json", action="store_true")
    chunk_args(p)

    p = sub.add_parser("index", parents=[common], help="폴더 색인해 보기")
    p.add_argument("folder", nargs="?", help="생략하면 samples/")
    p.add_argument("--show", type=int, default=5, help="미리 볼 청크 수")
    chunk_args(p)

    p = sub.add_parser("doctor", parents=[common], help="실제 모드 준비 상태 진단")
    p.add_argument("--download", action="store_true", help="리랭커 모델이 없으면 지금 내려받기")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd or "serve"
    if cmd == "serve" and args.cmd is None:  # 하위 명령 없이 실행 → serve 기본값 채우기
        args.host, args.port = "127.0.0.1", None
    if cmd == "ask":
        return asyncio.run(_ask(args))
    if cmd == "eval":
        return asyncio.run(_eval(args))
    if cmd == "index":
        return asyncio.run(_index(args))
    if cmd == "doctor":
        return _doctor(args)
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
