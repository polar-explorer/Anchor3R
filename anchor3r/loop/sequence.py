from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .types import AugmentedSequence, LoopCandidate


def build_augmented_sequence(
    image_paths: Sequence[str],
    candidates: Sequence[LoopCandidate],
    *,
    insert_position: str = "before",
) -> AugmentedSequence:
    paths = [str(path) for path in image_paths]
    if not paths:
        raise ValueError("image_paths must not be empty.")
    position = str(insert_position).strip().lower()
    if position not in {"before", "after"}:
        raise ValueError("insert_position must be 'before' or 'after'.")

    grouped: dict[int, list[tuple[int, LoopCandidate]]] = defaultdict(list)
    for rank, candidate in enumerate(candidates, start=1):
        query_frame = int(candidate.query_frame)
        match_frame = int(candidate.match_frame)
        if query_frame < 0 or query_frame >= len(paths):
            raise IndexError(f"candidate query frame {query_frame} is outside the sequence.")
        if match_frame < 0 or match_frame >= len(paths):
            raise IndexError(f"candidate match frame {match_frame} is outside the sequence.")
        grouped[query_frame].append((rank, candidate))

    for frame, items in grouped.items():
        grouped[frame] = sorted(items, key=lambda item: item[1].score, reverse=True)

    aug_paths: list[str] = []
    aug_to_orig: list[int] = []
    aug_is_inserted: list[bool] = []
    orig_to_main_aug = [-1] * len(paths)
    insertions: list[dict] = []

    def append_inserted(rank: int, candidate: LoopCandidate) -> None:
        aug_index = len(aug_paths)
        aug_paths.append(paths[int(candidate.match_frame)])
        aug_to_orig.append(int(candidate.match_frame))
        aug_is_inserted.append(True)
        insertions.append(
            {
                "aug_index": aug_index,
                "orig_index": int(candidate.match_frame),
                "query_frame": int(candidate.query_frame),
                "match_frame": int(candidate.match_frame),
                "candidate_rank": int(rank),
                "candidate_score": float(candidate.score),
                "insert_position": position,
            }
        )

    for orig_index, path in enumerate(paths):
        items = grouped.get(orig_index, [])
        if position == "before":
            for rank, candidate in items:
                append_inserted(rank, candidate)

        orig_to_main_aug[orig_index] = len(aug_paths)
        aug_paths.append(path)
        aug_to_orig.append(orig_index)
        aug_is_inserted.append(False)

        if position == "after":
            for rank, candidate in items:
                append_inserted(rank, candidate)

    if any(index < 0 for index in orig_to_main_aug):
        raise RuntimeError("failed to build orig_to_main_aug mapping.")

    return AugmentedSequence(
        paths=aug_paths,
        aug_to_orig=aug_to_orig,
        aug_is_inserted=aug_is_inserted,
        orig_to_main_aug=orig_to_main_aug,
        insertions=insertions,
    )
