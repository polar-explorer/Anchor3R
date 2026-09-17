from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoopCandidate:
    query_pos: int
    match_pos: int
    query_frame: int
    match_frame: int
    score: float
    method: str = "salad"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AugmentedSequence:
    paths: list[str]
    aug_to_orig: list[int]
    aug_is_inserted: list[bool]
    orig_to_main_aug: list[int]
    insertions: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "num_augmented_frames": len(self.paths),
            "num_original_frames": len(self.orig_to_main_aug),
            "num_inserted_frames": sum(1 for flag in self.aug_is_inserted if flag),
            "aug_to_orig": self.aug_to_orig,
            "aug_is_inserted": self.aug_is_inserted,
            "orig_to_main_aug": self.orig_to_main_aug,
            "insertions": self.insertions,
        }


@dataclass(frozen=True)
class LoopConfig:
    enabled: bool = False
    salad_checkpoint: str = "checkpoints/loop/dino_salad.ckpt"
    dino_checkpoint: str = "checkpoints/loop/dinov2_vitb14_pretrain.pth"
    salad_backbone: str = "dinov2_vitb14"
    salad_image_size: tuple[int, int] = (336, 336)
    salad_batch_size: int = 128
    descriptor_num_workers: int = 8
    keyframe_stride: int = 2
    retrieval_top_k: int = 3
    temporal_threshold: int = 30
    salad_score_threshold: float = 0.85
    insert_position: str = "before"
    motion_edge_weight: float = 1.0
    loop_edge_weight: float = 1.0
    verbose: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "LoopConfig":
        if values is None:
            return cls()
        data = dict(values)
        aliases = {
            "salad_ckpt_path": "salad_checkpoint",
            "salad_dino_weights_path": "dino_checkpoint",
            "salad_score_thresh": "salad_score_threshold",
            "min_frame_separation": "temporal_threshold",
            "temporal_exclusion": "temporal_threshold",
            "top_k": "retrieval_top_k",
            "loop_insert_position": "insert_position",
        }
        for old_name, new_name in aliases.items():
            if old_name in data and new_name not in data:
                data[new_name] = data.pop(old_name)
        if "salad_image_size" in data:
            data["salad_image_size"] = tuple(int(x) for x in data["salad_image_size"])
        if "enabled" in data and not isinstance(data["enabled"], bool):
            raise ValueError("loop.enabled must be a YAML boolean (true or false).")
        allowed = cls.__dataclass_fields__
        return cls(**{name: data[name] for name in data if name in allowed})

    def enabled_for_mode(self, infer_mode: str) -> "LoopConfig":
        if self.enabled and str(infer_mode).lower() != "offline":
            raise ValueError("Loop closure requires --infer-mode offline. Use --no-loop for online mode.")
        return self
