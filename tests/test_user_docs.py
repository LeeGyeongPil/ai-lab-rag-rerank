from fastapi.testclient import TestClient
from schema_check import assert_compare_schema

from rag_rerank.generator import NO_EVIDENCE
from rag_rerank.server import create_app

GARDEN = """# 햇살 주말농장 안내

## 분양

텃밭은 한 가구에 한 구획씩 배정합니다. 분양 신청은 매년 3월 첫째 주에 받습니다.

## 물과 도구

물은 오전 7시부터 오후 7시까지 쓸 수 있습니다. 호미와 삽은 관리동에서 무료로 빌려 줍니다.
"""
PARKING = "주차장은 농장 입구 왼쪽에 있으며 최대 20대까지 세울 수 있습니다.\n\n주차 요금은 받지 않습니다."


def test_index_folder_and_no_evidence(tmp_path):
    """AC-9: 폴더의 .md·.txt 를 색인해 질문하고, 무관한 질문엔 근거 부족을 알린다."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "garden.md").write_text(GARDEN, encoding="utf-8")
    (folder / "parking.txt").write_text(PARKING, encoding="utf-8")
    (folder / "ignore.csv").write_text("a,b", encoding="utf-8")

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r = client.post("/api/index", json={"folder": str(folder)})
        assert r.status_code == 200, r.text
        assert r.json()["docs"] == 2 and r.json()["chunks"] == 4

        hit = client.post("/api/compare", json={"question": "텃밭 분양 신청은 언제 받나요?", "collection": "mine"}).json()
        assert_compare_schema(hit)
        assert hit["after"][0]["chunk_id"].startswith("garden#")
        assert not hit["no_evidence"]
        hit2 = client.post("/api/compare", json={"question": "주차 요금", "collection": "mine", "retriever": "keyword"}).json()
        assert hit2["before"][0]["chunk_id"].startswith("parking#")

        body = {"question": "화성 탐사선은 언제 발사되나요?", "collection": "mine", "sides": "B"}
        miss = client.post("/api/compare", json=body).json()
        assert miss["no_evidence"] is True
        with client.stream("POST", "/api/answer", json=body) as resp:
            lines = [line for line in resp.iter_lines() if line.strip()]
        import json

        end = [json.loads(x) for x in lines if '"side": "B"' in x and '"end"' in x][0]
        assert end["text"] == NO_EVIDENCE == "문서에서 근거를 찾지 못했습니다."  # 답변 데이터라 언어와 무관하게 그대로


def test_index_uploaded_files():
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        files = [{"name": "garden.md", "text": GARDEN}, {"name": "photo.png", "text": "x"}]
        r = client.post("/api/index", json={"files": files, "chunking": "heading"})
        assert r.status_code == 200 and r.json()["docs"] == 1 and r.json()["chunks"] == 2
        st = client.get("/api/status").json()
        assert st["mine"]["docs"] == 1
