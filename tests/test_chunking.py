from rag_rerank.chunking import ChunkOptions, Document, chunk_document, chunk_documents
from rag_rerank.store import load_folder


def test_sample_chunk_counts(engine):
    """AC-5: 샘플 2개를 문단 단위로 자르면 33개, 제목 접두는 '[library-guide > 연체] …'."""
    docs = load_folder(engine.settings.samples_dir)
    chunks = chunk_documents(docs, ChunkOptions("paragraph", True))
    assert len(chunks) == 33
    overdue = [c for c in chunks if c.text.startswith("우리 도서관은 연체료를 받지 않습니다")]
    assert overdue[0].content.startswith("[library-guide > 연체] 우리 도서관은")
    assert overdue[0].source == "library-guide.md > 연체"
    # 접두를 끄면 검색용 글 = 원문
    plain = chunk_documents(docs, ChunkOptions("paragraph", False))
    assert all(c.content == c.text for c in plain)
    # '> 인용문' 머리말은 청크가 되지 않는다
    assert not any("가상의" in c.text for c in chunks)


def test_heading_and_recursive_strategies(engine):
    docs = load_folder(engine.settings.samples_dir)
    assert len(chunk_documents(docs, ChunkOptions("heading", True))) == 12  # 절 12개
    rec = chunk_documents(docs, ChunkOptions("recursive", False, max_chars=120, overlap=20))
    assert len(rec) > 12  # 절보다 잘게, 하지만 작은 문단끼리는 상한 안에서 다시 붙는다
    for c in rec:  # 겹침 접두('…' + 20자 + 공백)를 빼면 상한 이하
        body = c.text.split(" ", 1)[1] if c.text.startswith("…") else c.text
        assert len(body) <= 120 + 21


def test_txt_without_headings():
    doc = Document("memo", "memo.txt", "첫 문단입니다.\n\n둘째 문단입니다.")
    chunks = chunk_document(doc, ChunkOptions())
    assert [c.content for c in chunks] == ["[memo] 첫 문단입니다.", "[memo] 둘째 문단입니다."]
    assert [c.chunk_id for c in chunks] == ["memo#0", "memo#1"]
