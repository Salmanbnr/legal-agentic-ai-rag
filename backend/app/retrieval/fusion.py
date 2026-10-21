"""Reciprocal Rank Fusion over dense and lexical candidate lists.

Fusion uses only the rank position of a chunk inside each list. Raw similarity
scores from the two methods are never compared because they are not on the
same scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedChunk:
    """One fused candidate with its per-method ranks."""

    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    methods: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[FusedChunk]:
    """Fuse ranked id lists with Reciprocal Rank Fusion.

    Args:
        rankings: Mapping of method name to the ordered chunk ids it returned.
        k: Rank offset constant; the default of 60 is the standard value.

    Returns:
        Fused candidates ordered by descending fused score. Ties are broken by
        chunk id so the order is stable across runs.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method, ranking in rankings.items():
        for rank, raw_id in enumerate(ranking, start=1):
            chunk_id = str(raw_id)
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, {})[method] = rank

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        FusedChunk(
            chunk_id=chunk_id,
            score=scores[chunk_id],
            ranks=dict(ranks[chunk_id]),
            methods=sorted(ranks[chunk_id]),
        )
        for chunk_id in ordered
    ]
