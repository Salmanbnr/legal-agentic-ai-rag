"""Tests for Reciprocal Rank Fusion over dense and lexical results."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import RRF_K, reciprocal_rank_fusion


def test_chunk_found_by_both_methods_ranks_first() -> None:
    fused = reciprocal_rank_fusion(
        {
            "dense": ["chunk-a", "chunk-b", "chunk-c"],
            "lexical": ["chunk-a", "chunk-d", "chunk-b"],
        }
    )

    assert fused[0].chunk_id == "chunk-a"
    assert fused[0].methods == ["dense", "lexical"]
    assert fused[0].ranks == {"dense": 1, "lexical": 1}
    assert [item.chunk_id for item in fused] == [
        "chunk-a",
        "chunk-b",
        "chunk-d",
        "chunk-c",
    ]


def test_fusion_uses_ranks_and_not_raw_scores() -> None:
    fused = reciprocal_rank_fusion(
        {
            "dense": ["chunk-a", "chunk-b"],
            "lexical": ["chunk-b", "chunk-c"],
        },
        k=RRF_K,
    )

    expected_a = 1 / (RRF_K + 1)
    expected_b = 1 / (RRF_K + 2) + 1 / (RRF_K + 1)
    expected_c = 1 / (RRF_K + 2)

    assert fused[0].chunk_id == "chunk-b"
    assert fused[0].score == pytest.approx(expected_b)
    assert fused[1].score == pytest.approx(expected_a)
    assert fused[2].score == pytest.approx(expected_c)


def test_a_chunk_missing_from_one_list_still_contributes() -> None:
    fused = reciprocal_rank_fusion({"dense": ["chunk-a"], "lexical": []})

    assert [item.chunk_id for item in fused] == ["chunk-a"]
    assert fused[0].methods == ["dense"]


def test_ties_are_broken_deterministically() -> None:
    first = reciprocal_rank_fusion({"dense": ["chunk-b", "chunk-a"]})
    second = reciprocal_rank_fusion({"dense": ["chunk-b", "chunk-a"]})

    assert [item.chunk_id for item in first] == ["chunk-b", "chunk-a"]
    assert [item.chunk_id for item in first] == [
        item.chunk_id for item in second
    ]


def test_empty_rankings_produce_no_results() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == []


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["chunk-a"]}, k=0)
