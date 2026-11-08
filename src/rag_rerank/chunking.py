"""청킹: 문서를 검색 단위 조각(청크)으로 자른다.

세 가지 방식을 고를 수 있다.
- paragraph : 빈 줄로 나뉜 문단 하나 = 청크 하나 (기본값)
- recursive : 절 → 문단 → 문장 → 글자 순서로 잘라 최대 글자 수를 넘지 않게, 앞 조각 끝을 조금 겹쳐서
- heading   : 마크다운 제목(##) 아래 절 전체 = 청크 하나

옵션 title_prefix 를 켜면 청크 앞에 "[문서 > 절 제목]"을 붙여서 검색·리랭킹에 쓴다.
('분실과 같은 기준으로…'처럼 맥락이 끊긴 문단이 무슨 이야기인지 알려 주는 효과)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")
STRATEGIES = ("paragraph", "recursive", "heading")


@dataclass(frozen=True)
class Document:
    name: str  # 확장자 뺀 이름(청크 id·접두에 씀). 예: library-guide
    filename: str  # 화면에 보여 줄 파일 이름. 예: library-guide.md
    text: str


@dataclass(frozen=True)
class ChunkOptions:
    strategy: str = "paragraph"
    title_prefix: bool = True
    max_chars: int = 200  # recursive 전용
    overlap: int = 40  # recursive 전용

    def key(self) -> str:
        return f"{self.strategy}-{int(self.title_prefix)}-{self.max_chars}-{self.overlap}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "library-guide#3" — 문서 이름 + 문서 안 순번
    doc: str
    filename: str
    section: str | None  # 속한 절 제목(없으면 None)
    text: str  # 화면에 보여 줄 원문 조각
    content: str  # 검색·리랭킹에 넣는 글(제목 접두 포함)

    @property
    def source(self) -> str:
        return f"{self.filename} > {self.section}" if self.section else self.filename


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    """문서를 (절 제목, 문단 목록) 묶음으로 나눈다.

    - '# 제목'(1단계)은 문서 제목이라 절로 치지 않는다.
    - '> 인용문'으로 시작하는 문단은 머리말·주석으로 보고 건너뛴다(샘플의 "가상 기관" 안내문).
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    para: list[str] = []

    def flush() -> None:
        block = "\n".join(para).strip()
        para.clear()
        if block and not block.startswith(">"):
            sections[-1][1].append(block)

    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            flush()
            if len(m.group(1)) >= 2:  # '##' 이하만 절 제목으로 쓴다
                sections.append((m.group(2).strip(), []))
        elif not line.strip():
            flush()
        else:
            para.append(line.rstrip())
    flush()
    return [(title, paras) for title, paras in sections if paras]


def _recursive_split(text: str, max_chars: int) -> list[str]:
    """max_chars 이하가 될 때까지 문단 → 문장 → 글자 순서로 더 잘게 자른다."""
    if len(text) <= max_chars:
        return [text]
    for pattern in ("\n\n", SENTENCE_END, " "):
        parts = text.split(pattern) if isinstance(pattern, str) else pattern.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            joiner = "\n\n" if pattern == "\n\n" else " "
            out, buf = [], ""
            for p in parts:  # 작은 조각은 상한 안에서 다시 붙여 너무 잘게 쪼개지지 않게 한다
                cand = f"{buf}{joiner}{p}" if buf else p
                if len(cand) <= max_chars:
                    buf = cand
                else:
                    if buf:
                        out.append(buf)
                    buf = p
            if buf:
                out.append(buf)
            return [piece for part in out for piece in _recursive_split(part, max_chars)]
    # 더 나눌 곳이 없으면 글자 수로 자른다
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    """앞 조각의 끝 overlap 글자를 다음 조각 앞에 붙인다(경계에 걸친 문장을 살리려는 겹침)."""
    if overlap <= 0:
        return pieces
    return [pieces[0]] + [f"…{prev[-overlap:]} {cur}" for prev, cur in zip(pieces, pieces[1:])]


def chunk_document(doc: Document, opts: ChunkOptions) -> list[Chunk]:
    if opts.strategy not in STRATEGIES:
        raise ValueError(f"알 수 없는 청킹 방식: {opts.strategy}")
    texts: list[tuple[str | None, str]] = []
    for title, paras in _sections(doc.text):
        if opts.strategy == "paragraph":
            texts += [(title, p) for p in paras]
        elif opts.strategy == "heading":
            texts.append((title, "\n".join(paras)))
        else:  # recursive
            pieces = _recursive_split("\n\n".join(paras), opts.max_chars)
            texts += [(title, p) for p in _with_overlap(pieces, opts.overlap)]

    chunks = []
    for i, (title, text) in enumerate(texts):
        label = f"{doc.name} > {title}" if title else doc.name
        content = f"[{label}] {text}" if opts.title_prefix else text
        chunks.append(Chunk(f"{doc.name}#{i}", doc.name, doc.filename, title, text, content))
    return chunks


def chunk_documents(docs: list[Document], opts: ChunkOptions) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, opts)]
