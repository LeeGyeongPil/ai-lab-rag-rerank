"""비교 결과(dict) 모양 검사 — test_pipeline·test_api·test_user_docs 가 함께 쓴다."""


def assert_compare_schema(r: dict) -> None:
    assert r["scales"]["before"] and r["scales"]["after"]  # 두 열의 척도 이름
    before, after = r["before"], r["after"]
    assert after is not None
    # after 는 before 와 같은 청크 집합의 순열
    assert sorted(c["chunk_id"] for c in after) == sorted(c["chunk_id"] for c in before)
    assert [c["rank"] for c in before] == list(range(1, len(before) + 1))
    assert [c["rank"] for c in after] == list(range(1, len(after) + 1))
    by_id = {c["chunk_id"]: c for c in before}
    for c in after:
        assert c["move"] == c["rank_before"] - c["rank"]  # 이동 = 전 순위 - 후 순위
        assert by_id[c["chunk_id"]]["rank"] == c["rank_before"]
        assert by_id[c["chunk_id"]]["move"] == c["move"]
        assert 0.0 <= c["score"] <= 1.0  # 리랭커 점수는 0~1
    # 리랭커 점수 내림차순
    assert all(a["score"] >= b["score"] for a, b in zip(after, after[1:]))
    assert r["answer_evidence"]["A"] == [c["chunk_id"] for c in before[:3]]
    assert r["answer_evidence"]["B"] == [c["chunk_id"] for c in after[:3]]
