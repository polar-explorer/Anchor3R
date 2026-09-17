from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .types import AugmentedSequence, LoopCandidate, LoopConfig


def save_loop_artifacts(
    out_dir: str | Path,
    *,
    config: LoopConfig,
    candidates: Sequence[LoopCandidate],
    retrieval_stats: dict[str, Any],
    augmented: AugmentedSequence,
    edge_summary: dict[str, Any] | None = None,
    motion_edges: Sequence[dict[str, Any]] | None = None,
) -> None:
    loop_dir = Path(out_dir) / "loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    _save_json(loop_dir / "config.json", asdict(config))
    _save_json(loop_dir / "retrieval_stats.json", retrieval_stats)
    _save_json(loop_dir / "candidates.json", [candidate.to_json() for candidate in candidates])
    _save_json(loop_dir / "augmented_sequence.json", augmented.to_json())
    if edge_summary is not None:
        _save_json(loop_dir / "edge_summary.json", edge_summary)
    if motion_edges is not None:
        _save_json(loop_dir / "motion_edges.json", list(motion_edges))


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
