from collections import defaultdict


def reciprocal_rank_fusion_with_ranks(rankings: dict[str, list[tuple[str, float]]], k: int = 60):
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    fused: dict[str, float] = defaultdict(float)
    branches: dict[str, dict[str, float]] = defaultdict(dict)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, results in rankings.items():
        seen: set[str] = set()
        for rank, (image_id, original_score) in enumerate(results, start=1):
            if image_id in seen:
                continue
            seen.add(image_id)
            fused[image_id] += 1.0 / (k + rank)
            branches[image_id][name] = original_score
            ranks[image_id][name] = rank
    return [(image_id, fused[image_id], branches[image_id], ranks[image_id])
            for image_id in sorted(fused, key=lambda key: (-fused[key], key))]


def reciprocal_rank_fusion(rankings: dict[str, list[tuple[str, float]]], k: int = 60):
    return [(image_id, score, branches)
            for image_id, score, branches, _ in reciprocal_rank_fusion_with_ranks(rankings, k)]
