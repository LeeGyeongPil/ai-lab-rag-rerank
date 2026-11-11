"""명령줄: rag-rerank [serve|ask|eval|index|doctor] [--mock]

  uv run rag-rerank                      웹 화면 띄우기(기본, http://127.0.0.1:8765)
  uv run rag-rerank --mock --no-browser   브라우저를 열지 않고 주소만 출력(화면 없는 서버·컨테이너)
  uv run rag-rerank ask "질문" --mock     터미널에서 전/후 비교 + 답변 A/B
  uv run rag-rerank eval --mock --json    성적표(hit@1·MRR 전/후)
  uv run rag-rerank index 내폴더          내 문서 폴더 색인해 보기
  uv run rag-rerank doctor               실제 모드 준비 상태 진단
  uv run rag-rerank --lang ko ...        화면 언어(en|ko). 없으면 RAG_LANG > LC_ALL > LC_MESSAGES > LANG, 한국어가 아니면 영어

사람이 읽는 문구는 모두 i18n.py 의 표에 있다. 이 파일은 문구 id 와 값만 넘긴다.
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
from .generator import NO_EVIDENCE
from .i18n import LANGS, RagError, os_error_to_rag_error, render_notice, resolve_lang, t
from .pipeline import Engine, answer_stream, compare, create_engine
from .retrieve import RETRIEVERS
from .store import build_index, connect, load_folder


# ---------- 공통 ----------
def _engine(args) -> tuple[Settings, Engine]:
    settings = Settings.from_env()
    mode_info = resolve_mode(settings, force_mock=getattr(args, "mock", False), lang=args.lang)
    reason, hints = mode_info.render(args.lang)
    print(f"[{mode_info.mode.upper()}] {reason}", file=sys.stderr)
    for hint in hints:
        print(f"  · {hint}", file=sys.stderr)
    return settings, create_engine(settings, mode_info)


def _prepare_reranker(engine: Engine, lang: str) -> None:
    """실제 모드: 리랭커가 없으면 지금 내려받는다(터미널 진행 막대, 첫 실행에 한 번 약 588MB)."""
    if engine.hub is not None and engine.hub.get() is None:
        if engine.hub.local_paths() is None:
            print(t(lang, "cli.model_download", repo=RERANKER_REPO), file=sys.stderr)
        engine.hub.prepare_blocking(show_progress=True)


async def _build(engine: Engine, args, docs_dir: str | None):
    opts = ChunkOptions(args.chunking, not args.no_title_prefix)
    if docs_dir:
        docs = load_folder(Path(docs_dir))  # 잘못된 폴더·읽기 실패는 RagError — main() 이 한 줄로 알린다
        if not docs:  # 웹 API(/api/index)와 같은 오류
            raise RagError("no_docs")
        return await build_index("mine", docs, opts, engine.embedder, engine.settings.cache_dir)
    return await build_index("samples", load_folder(engine.settings.samples_dir), opts, engine.embedder,
                             engine.settings.cache_dir)


def _short(text: str, n: int = 38) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _move(m: int) -> str:
    return f"▲{m}" if m > 0 else (f"▼{-m}" if m < 0 else "–")


def _print_table(lang: str, title: str, scale: str, rows: list[dict], after: bool) -> None:
    print(f"\n{title}  [{scale}]")
    head = t(lang, "cli.head") + (t(lang, "cli.head_prev") if after else "") + t(lang, "cli.head_tail")
    print(head)
    for r in rows:
        extra = f"{r['rank_before']:>4}    " if after else ""
        mv = _move(r["move"]) if "move" in r else ""
        print(f"  {r['rank']:>3}   {r['score']:<8}  {mv:<4}  {extra}{r['source']} | {_short(r['text'])}")


# ---------- ask ----------
async def _ask(args) -> int:
    settings, engine = _engine(args)
    try:
        _prepare_reranker(engine, args.lang)
        index = await _build(engine, args, args.docs)
        result = await compare(engine, index, args.question, args.retriever, args.top_k)
        want_answer = args.answer or engine.mode == "mock"  # mock 은 답이 즉시 나오므로 늘 보여 준다
        answers = {}
        if not args.json:
            c = result["chunking"]
            prefix = t(args.lang, "cli.prefix_on" if c["title_prefix"] else "cli.prefix_off")
            print(t(args.lang, "cli.question", question=result["question"]))
            print(t(args.lang, "cli.settings", retriever=args.retriever, top_k=args.top_k, strategy=c["strategy"],
                    prefix=prefix, chunks=len(index.chunks)))
            # 척도 이름은 JSON 값(한국어)을 그대로 쓰지 않고 화면 언어 표에서 찾는다
            _print_table(args.lang, t(args.lang, "cli.before"), t(args.lang, f"scale.{args.retriever}"),
                         result["before"], after=False)
            if result["after"] is not None:
                _print_table(args.lang, t(args.lang, "cli.after"), t(args.lang, "scale.after"), result["after"], after=True)
            ratio = result["top2_ratio"]
            if ratio["before"] is not None and ratio["after"] is not None:
                # 2위 점수가 1위의 몇 %인지: 작을수록 1위가 나머지보다 확실히 앞선다(화면 '점수 분포'와 같은 값)
                print("\n" + t(args.lang, "cli.ratio", before=f"{ratio['before']:.0%}", after=f"{ratio['after']:.0%}"))
            if result["no_evidence"]:
                print("\n" + t(args.lang, "cli.no_evidence"))
            tm = result["timings_ms"]
            print("\n" + t(args.lang, "cli.timings", embed=tm["embed"], retrieve=tm["retrieve"], rerank=tm["rerank"]))
        if want_answer:
            for side, label in (("A", t(args.lang, "cli.answer_a")), ("B", t(args.lang, "cli.answer_b"))):
                text = ""
                async with aclosing(answer_stream(engine, index, result, side)) as stream:
                    async for ev in stream:
                        if ev["type"] == "thinking" and engine.mode == "real" and not args.json:
                            print("\r" + t(args.lang, "cli.thinking", tokens=ev["tokens"]), end="", file=sys.stderr, flush=True)
                        elif ev["type"] == "notice" and not args.json:
                            print(f"\n  ※ {render_notice(ev, args.lang)['text']}", file=sys.stderr)
                        elif ev["type"] == "end":
                            text = ev["text"]
                            answers[side] = {"text": text, "citations": ev["citations"]}
                if not args.json:
                    if engine.mode == "real":
                        print("", file=sys.stderr)
                    shown = t(args.lang, "no_evidence_answer") if text == NO_EVIDENCE else text  # 근거 없음 문구만 화면 언어로
                    print(f"\n{label}: {shown}")
                    cites = ", ".join(f"[{c['n']}]={c['chunk_id']}" for c in answers[side]["citations"])
                    print(t(args.lang, "cli.cites", cites=cites or t(args.lang, "cli.none")))
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
        _prepare_reranker(engine, args.lang)
        index = await _build(engine, args, None)
        result = await evaluate(engine, index, load_questions(settings.samples_dir / "questions.json"), args.top_k)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=1))
            return 0
        n = result["n_questions"]
        print(t(args.lang, "cli.eval_title", n=n, top_k=args.top_k, reranker=result["reranker"]))
        print(t(args.lang, "cli.eval_head"))
        for name, r in result["table"].items():
            print(f"  {name:<8}  {r['hit1_before']:>2}/{n} → {r['hit1_after']:>2}/{n}   "
                  f"{r['mrr_before']:.2f} → {r['mrr_after']:.2f}      {r['orders_changed']}")
        print("\n" + t(args.lang, "cli.eval_changed", count=result["orders_changed"]))
        return 0
    finally:
        await engine.aclose()


# ---------- index ----------
async def _index(args) -> int:
    settings, engine = _engine(args)
    try:
        index = await _build(engine, args, args.folder)
        print(t(args.lang, "cli.index_summary", docs=len({c.doc for c in index.chunks}), chunks=len(index.chunks),
                strategy=index.options.strategy))
        for c in index.chunks[: args.show]:
            print(f"  {c.chunk_id:<24} {_short(c.content, 60)}")
        return 0
    finally:
        await engine.aclose()


# ---------- doctor ----------
def _doctor(args) -> int:
    lang = args.lang
    settings = Settings.from_env()
    checks: list[tuple[str, bool, str, str]] = []  # (항목, 통과?, 설명, 해결 방법)
    checks.append(("Python 3.14+", sys.version_info >= (3, 14), sys.version.split()[0], "uv python install 3.14"))
    sqlite_name = t(lang, "doctor.sqlite")
    try:
        with closing(connect(":memory:")) as db:
            ver = db.execute("select vec_version()").fetchone()[0]
        checks.append((sqlite_name, True, ver, ""))
    except (AttributeError, sqlite3.Error) as e:
        checks.append((sqlite_name, False, str(e), t(lang, "doctor.sqlite_fix")))
    ok, names, err = probe_ollama(settings)
    checks.append((t(lang, "doctor.ollama"), ok, settings.ollama_base_url if ok else err, t(lang, "doctor.ollama_fix")))
    for m in (settings.embed_model, settings.chat_model):
        present = ok and has_model(names, m)
        checks.append((t(lang, "doctor.model", name=m), present,
                       t(lang, "doctor.present" if present else "doctor.absent"), f"ollama pull {m}"))
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        checks.append(("onnxruntime·tokenizers", True, f"onnxruntime {onnxruntime.__version__}", ""))
    except ImportError as e:
        checks.append(("onnxruntime·tokenizers", False, str(e), "uv sync"))
    from .reranker import RerankerHub

    hub = RerankerHub(settings)
    if args.download and hub.local_paths() is None:
        print(t(lang, "cli.model_download_doctor", repo=RERANKER_REPO), file=sys.stderr)
        hub.prepare_blocking(show_progress=True)
    have = hub.local_paths() is not None
    checks.append((t(lang, "doctor.reranker_file"), have,
                   t(lang, "doctor.present") if have else t(lang, "doctor.reranker_absent"),
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
    mode_info = resolve_mode(settings, force_mock=getattr(args, "mock", False), lang=args.lang)
    reason, hints = mode_info.render(args.lang)
    port = args.port or settings.port
    url = browser_url(args.host, port)
    probe_host = "127.0.0.1" if args.host in WILDCARD_HOSTS else args.host
    with closing(socket.socket()) as probe:  # 포트가 이미 쓰이고 있으면 알아듣기 쉬운 말로 알린다
        if probe.connect_ex((probe_host, port)) == 0:
            print(t(args.lang, "serve.port_busy", port=port, next_port=port + 1), file=sys.stderr)
            return 1
    # flush=True: 터미널이 아닌 곳(도커 로그, 파이프)으로 내보낼 때도 이 줄들이 바로 보이게 한다
    print(f"[{mode_info.mode.upper()}] {reason}", flush=True)
    for hint in hints:
        print(f"  · {hint}", flush=True)
    print(t(args.lang, "serve.url", url=url), flush=True)
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
            print(t(args.lang, "serve.lan", port=port), flush=True)
        else:
            extra = [args.host]
        uvicorn.run(create_app(settings, mode_info, extra_hosts=extra, lang=args.lang), host=args.host, port=port, log_level="warning")
    finally:
        if timer is not None:
            timer.cancel()
    return 0


# ---------- 인자 ----------
def _pre_parse_lang(argv: list[str] | None) -> str:
    """도움말 문구도 언어를 따라야 하므로, 본 파서를 만들기 전에 --lang 만 먼저 읽는다(잘못된 값은 본 파서가 거부)."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--lang", default=None)
    known, _ = pre.parse_known_args(argv)
    return resolve_lang(known.lang)


def build_parser(lang: str = "en") -> argparse.ArgumentParser:
    def h(message_id: str) -> str:
        return t(lang, message_id)

    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: 'rag-rerank --mock ask …' 와 'rag-rerank ask … --mock' 둘 다 되게 한다
    common.add_argument("--mock", action="store_true", default=argparse.SUPPRESS, help=h("help.mock"))
    # --lang 도 하위 명령 앞뒤 어디에 써도 된다. 값이 없으면 main() 이 RAG_LANG·LANG 으로 정한다
    common.add_argument("--lang", choices=LANGS, default=argparse.SUPPRESS, metavar="{en,ko}", help=h("help.lang"))

    def chunk_args(p):
        p.add_argument("--chunking", choices=STRATEGIES, default="paragraph", help=h("help.chunking"))
        p.add_argument("--no-title-prefix", action="store_true", help=h("help.no_title_prefix"))
        p.add_argument("--top-k", type=int, default=8, help=h("help.top_k"))

    parser = argparse.ArgumentParser(prog="rag-rerank", parents=[common], description=h("help.description"))
    # 하위 명령 없이 'rag-rerank --no-browser' 로도 쓸 수 있게(화면 없는 서버·컨테이너·SSH)
    parser.add_argument("--no-browser", action="store_true", default=argparse.SUPPRESS, help=h("help.no_browser"))
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("serve", parents=[common], help=h("help.serve"))
    p.add_argument("--host", default="127.0.0.1", help=h("help.host"))
    p.add_argument("--port", type=int, default=None, help=h("help.port"))
    p.add_argument("--no-browser", action="store_true", default=argparse.SUPPRESS, help=h("help.no_browser"))

    p = sub.add_parser("ask", parents=[common], help=h("help.ask"))
    p.add_argument("question", help=h("help.question"))
    p.add_argument("--retriever", choices=RETRIEVERS, default="hybrid", help=h("help.retriever"))
    p.add_argument("--docs", help=h("help.docs"))
    p.add_argument("--answer", action="store_true", help=h("help.answer"))
    p.add_argument("--json", action="store_true", help=h("help.json"))
    chunk_args(p)

    p = sub.add_parser("eval", parents=[common], help=h("help.eval"))
    p.add_argument("--json", action="store_true", help=h("help.json"))
    chunk_args(p)

    p = sub.add_parser("index", parents=[common], help=h("help.index"))
    p.add_argument("folder", nargs="?", help=h("help.folder"))
    p.add_argument("--show", type=int, default=5, help=h("help.show"))
    chunk_args(p)

    p = sub.add_parser("doctor", parents=[common], help=h("help.doctor"))
    p.add_argument("--download", action="store_true", help=h("help.download"))
    return parser


def main(argv: list[str] | None = None) -> int:
    lang = _pre_parse_lang(argv)
    args = build_parser(lang).parse_args(argv)
    args.lang = getattr(args, "lang", None) or lang  # --lang 이 있으면 그 값, 없으면 환경에서 정한 값
    cmd = args.cmd or "serve"
    if cmd == "serve" and args.cmd is None:  # 하위 명령 없이 실행 → serve 기본값 채우기
        args.host, args.port = "127.0.0.1", None
    try:
        if cmd == "ask":
            return asyncio.run(_ask(args))
        if cmd == "eval":
            return asyncio.run(_eval(args))
        if cmd == "index":
            return asyncio.run(_index(args))
        if cmd == "doctor":
            return _doctor(args)
        return _serve(args)
    except RagError as e:  # 입력·준비 오류: 트레이스백 없이 한 줄, 종료 코드 1
        print(e.render(args.lang), file=sys.stderr)
        return 1
    except OSError as e:  # 읽기·쓰기 권한 없음 등(PermissionError 포함)
        print(os_error_to_rag_error(e).render(args.lang), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
