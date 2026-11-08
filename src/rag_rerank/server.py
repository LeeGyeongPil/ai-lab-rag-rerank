"""웹 서버(FastAPI): 화면(static/)과 JSON API.

  GET  /                  화면
  GET  /api/status        모드·리랭커 내려받기 진행률·예시 질문
  POST /api/compare       리랭킹 전/후 비교(JSON)
  POST /api/answer        답변 A/B 스트리밍(NDJSON: 한 줄에 이벤트 하나)
  POST /api/eval          성적표(hit@1·MRR 전/후)
  POST /api/index         내 문서 색인(폴더 경로 또는 파일 내용)
  POST /api/reranker/retry  리랭커 내려받기 다시 시도(최대 3회)

자원 관리: Ollama HTTP 클라이언트는 서버 수명(lifespan) 동안 하나만 쓰고 종료 때 닫는다.
색인은 최근 6개만 메모리에 둔다(디스크 파일 정리 때 이 6개의 파일은 지우지 않는다).
답변 스트림은 브라우저가 끊으면 Ollama 스트림까지 닫는다.

보안: 127.0.0.1 에만 묶고, Host 헤더가 127.0.0.1/localhost 인 요청만 받는다
(DNS 리바인딩 — 다른 사이트가 이름만 바꿔 이 서버를 부르는 공격 — 을 막는다).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .chunking import ChunkOptions, Document
from .config import ModeInfo, Settings, resolve_mode
from .evaluate import evaluate, load_questions
from .pipeline import Engine, answer_stream, compare, create_engine
from .store import Index, build_index, load_folder

STATIC = Path(__file__).parent / "static"
MAX_INDEXES = 6  # (컬렉션 × 청킹 설정) 조합을 최근 6개까지만 메모리에 둔다(store.MAX_CACHED_INDEXES 이하)
LOOPBACK_HOSTS = ["127.0.0.1", "localhost"]  # 포트는 상관없이 이 이름으로 온 요청만 받는다
FEATURED = "중학생도 공유자전거를 탈 수 있나요?"  # 첫 번째 예시 질문 칩
log = logging.getLogger("rag_rerank")


# ---------- 요청 모양 ----------
class Options(BaseModel):
    collection: Literal["samples", "mine"] = "samples"
    chunking: Literal["paragraph", "recursive", "heading"] = "paragraph"
    title_prefix: bool = True
    top_k: int = Field(8, ge=4, le=20)


class CompareRequest(Options):
    question: str = Field(min_length=1, max_length=500)
    retriever: Literal["vector", "keyword", "hybrid"] = "hybrid"


class AnswerRequest(CompareRequest):
    sides: Literal["AB", "A", "B"] = "AB"


class FileIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(max_length=1_000_000)


class IndexRequest(BaseModel):
    folder: str | None = None
    files: list[FileIn] | None = Field(None, max_length=200)
    chunking: Literal["paragraph", "recursive", "heading"] = "paragraph"
    title_prefix: bool = True


# ---------- 서버 상태 ----------
class AppState:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.settings = engine.settings
        self.samples = load_folder(self.settings.samples_dir)
        self.questions = load_questions(self.settings.samples_dir / "questions.json")
        self.user_docs: list[Document] = []  # 마지막으로 올린 내 문서만 기억(새로 올리면 교체)
        self.indexes: OrderedDict[tuple, Index] = OrderedDict()
        self.lock = asyncio.Lock()  # 같은 색인을 동시에 두 번 만들지 않게

    async def index(self, collection: str, chunking: str, title_prefix: bool) -> Index:
        docs = self.samples if collection == "samples" else self.user_docs
        if not docs:
            raise HTTPException(400, "먼저 '내 문서' 탭에서 문서를 색인하세요.")
        key = (collection, chunking, title_prefix)
        async with self.lock:
            cached = self.indexes.get(key)
            # 파일이 밖에서 지워졌으면(사용자가 .cache 삭제 등) 메모리 색인을 버리고 다시 만든다
            if cached is not None and (cached.db_path.exists() or not cached.chunks):
                self.indexes.move_to_end(key)
                return cached
            in_use = [i.db_path for k, i in self.indexes.items() if k != key]  # 디스크 정리 때 지우면 안 되는 파일
            idx = await build_index(collection, docs, ChunkOptions(chunking, title_prefix),
                                    self.engine.embedder, self.settings.cache_dir, in_use=in_use)
            self.indexes[key] = idx
            while len(self.indexes) > MAX_INDEXES:
                self.indexes.popitem(last=False)
            return idx


def create_app(settings: Settings | None = None, mode_info: ModeInfo | None = None,
               extra_hosts: list[str] | None = None) -> FastAPI:
    """extra_hosts: serve --host 로 127.0.0.1 이 아닌 주소를 직접 고른 경우 그 이름도 허용한다."""
    settings = settings or Settings.from_env()
    mode_info = mode_info or resolve_mode(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = create_engine(settings, mode_info)  # 여기서 httpx 클라이언트를 하나 만든다
        app.state.rag = AppState(engine)
        if engine.hub is not None:
            engine.hub.start_background()  # 리랭커 모델 내려받기를 바로 시작(그동안 '전' 결과는 쓸 수 있음)
        try:
            await app.state.rag.index("samples", "paragraph", True)  # 샘플 색인을 미리 만들어 둔다
        except Exception as e:  # Ollama 가 잠깐 바쁘더라도 서버는 뜨게 한다(첫 요청 때 다시 시도)
            log.warning("샘플 색인 준비 실패: %s", e)
        try:
            yield
        finally:
            await engine.aclose()  # httpx 클라이언트 닫기 + ONNX 세션 놓기

    app = FastAPI(title="RAG rerank workbench", lifespan=lifespan)
    # Host 헤더 검사: 127.0.0.1·localhost 가 아니면 400. (포트 번호는 떼고 비교한다)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOOPBACK_HOSTS + list(extra_hosts or []))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def rag(request: Request) -> AppState:
        return request.app.state.rag

    @app.get("/")
    async def page():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status(request: Request):
        st = rag(request)
        eng = st.engine
        chips = sorted(st.questions, key=lambda q: q["q"] != FEATURED)
        return {
            "mode": eng.mode,
            "mode_reason": eng.mode_info.reason,
            "hints": eng.mode_info.hints,
            "models": {"embed": eng.embedder.name, "chat": eng.generator.name},
            "reranker": eng.hub.status() if eng.hub else {"state": "ready", "model": "mock-overlap", "percent": 100},
            "questions": [{"q": q["q"], "gold": q["gold"], "note": q.get("note")} for q in chips],
            "samples": [d.filename for d in st.samples],
            "mine": {"docs": len(st.user_docs), "files": [d.filename for d in st.user_docs[:50]]},
            "defaults": {"retriever": "hybrid", "top_k": settings.top_k, "chunking": "paragraph", "title_prefix": True},
        }

    @app.post("/api/reranker/retry")
    async def retry(request: Request):
        hub = rag(request).engine.hub
        if hub is None:
            return {"started": False, "reason": "mock 모드에는 내려받을 모델이 없습니다."}
        return {"started": hub.start_background(), "reranker": hub.status()}

    @app.post("/api/index")
    async def index_docs(req: IndexRequest, request: Request):
        st = rag(request)
        if req.folder:
            try:
                docs = load_folder(Path(req.folder))
            except ValueError as e:
                raise HTTPException(400, str(e))
        elif req.files:
            docs = [Document(f.name.rsplit(".", 1)[0], f.name, f.text)
                    for f in req.files if f.name.lower().endswith((".md", ".txt"))]
        else:
            raise HTTPException(400, "folder 또는 files 중 하나를 보내세요.")
        if not docs:
            raise HTTPException(400, ".md 또는 .txt 문서를 찾지 못했습니다.")
        st.user_docs = docs
        st.indexes = OrderedDict((k, v) for k, v in st.indexes.items() if k[0] != "mine")  # 예전 내 문서 색인 버리기
        idx = await st.index("mine", req.chunking, req.title_prefix)
        return {"collection": "mine", "docs": len(docs), "chunks": len(idx.chunks),
                "files": [d.filename for d in docs[:50]]}

    @app.post("/api/compare")
    async def compare_route(req: CompareRequest, request: Request):
        st = rag(request)
        idx = await st.index(req.collection, req.chunking, req.title_prefix)
        return await compare(st.engine, idx, req.question, req.retriever, req.top_k)

    @app.post("/api/eval")
    async def eval_route(req: Options, request: Request):
        st = rag(request)
        idx = await st.index("samples", req.chunking, req.title_prefix)
        try:
            return await evaluate(st.engine, idx, st.questions, req.top_k)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/answer")
    async def answer_route(req: AnswerRequest, request: Request):
        st = rag(request)
        idx = await st.index(req.collection, req.chunking, req.title_prefix)
        result = await compare(st.engine, idx, req.question, req.retriever, req.top_k)

        async def events():
            # 한 줄에 JSON 하나(NDJSON). 답변 A가 끝나면 B를 만든다(Ollama 대기열을 두 답이 다투지 않게).
            yield json.dumps({"type": "start", "answer_evidence": result["answer_evidence"],
                              "no_evidence": result["no_evidence"]}, ensure_ascii=False) + "\n"
            for side in req.sides:
                try:
                    async with aclosing(answer_stream(st.engine, idx, result, side)) as stream:
                        async for event in stream:
                            if await request.is_disconnected():
                                return  # 브라우저가 끊음/중지 → aclosing 이 Ollama 스트림까지 닫는다
                            yield json.dumps({"side": side, **event}, ensure_ascii=False) + "\n"
                except Exception as e:  # Ollama 오류 등은 화면에 보여 주고 다음 답변으로 넘어간다
                    yield json.dumps({"side": side, "type": "error", "text": f"{type(e).__name__}: {e}"},
                                     ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return app
