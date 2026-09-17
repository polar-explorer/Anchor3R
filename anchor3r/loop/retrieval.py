from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from anchor3r.model.layers.vision_transformer import DinoVisionTransformer

from .types import LoopCandidate, LoopConfig


def _log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[Anchor3R-loop] {message}", flush=True)


def select_keyframes(num_frames: int, stride: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    stride = max(1, int(stride))
    keyframes = list(range(0, num_frames, stride))
    if keyframes[-1] != num_frames - 1:
        keyframes.append(num_frames - 1)
    return keyframes


def retrieve_loop_candidates(
    image_paths: Sequence[str],
    config: LoopConfig,
    device: torch.device,
) -> tuple[list[LoopCandidate], dict[str, Any]]:
    started = time.perf_counter()
    if not config.enabled:
        return [], {"enabled": False}
    if len(image_paths) < 2:
        return [], {"enabled": True, "num_candidates": 0, "reason": "too_few_frames"}

    keyframe_indices = select_keyframes(len(image_paths), config.keyframe_stride)
    if len(keyframe_indices) < 2:
        return [], {"enabled": True, "num_candidates": 0, "reason": "too_few_keyframes"}

    salad_checkpoint = _resolve_existing_path(config.salad_checkpoint, "SALAD checkpoint")
    dino_checkpoint = _resolve_existing_path(config.dino_checkpoint, "DINO checkpoint")
    keyframe_paths = [image_paths[index] for index in keyframe_indices]

    desc_started = time.perf_counter()
    descriptors = compute_descriptors(
        keyframe_paths,
        salad_checkpoint=salad_checkpoint,
        dino_checkpoint=dino_checkpoint,
        backbone=config.salad_backbone,
        image_size=config.salad_image_size,
        batch_size=config.salad_batch_size,
        device=device,
        num_workers=config.descriptor_num_workers,
    )
    desc_sec = time.perf_counter() - desc_started

    search_started = time.perf_counter()
    candidates, search_stats = retrieve_candidates_from_descriptors(
        descriptors,
        keyframe_indices,
        config,
        device,
    )
    search_sec = time.perf_counter() - search_started
    candidates.sort(key=lambda item: item.score, reverse=True)

    stats = {
        "enabled": True,
        "num_frames": int(len(image_paths)),
        "keyframe_stride": int(max(1, config.keyframe_stride)),
        "num_keyframes": int(len(keyframe_indices)),
        "retrieval_top_k": int(config.retrieval_top_k),
        "temporal_threshold": int(config.temporal_threshold),
        "salad_score_threshold": float(config.salad_score_threshold),
        "num_candidates": int(len(candidates)),
        "descriptor_sec": float(desc_sec),
        "search_sec": float(search_sec),
        "total_sec": float(time.perf_counter() - started),
        **search_stats,
    }
    _log(
        f"keyframes={len(keyframe_indices)} candidates={len(candidates)} total={stats['total_sec']:.2f}s",
        config.verbose,
    )
    return candidates, stats


def compute_descriptors(
    image_paths: Sequence[str],
    *,
    salad_checkpoint: Path,
    dino_checkpoint: Path,
    backbone: str,
    image_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    num_workers: int = 8,
) -> torch.Tensor:
    model = SaladDescriptor(backbone, dino_checkpoint)
    _load_salad_checkpoint(model, salad_checkpoint)
    model = model.to(device).eval()
    chunks: list[torch.Tensor] = []
    batch_size = max(1, int(batch_size))
    autocast_dtype = torch.bfloat16 if _cuda_bf16_supported(device) else torch.float16
    storage_dtype = _descriptor_storage_dtype(device)
    with torch.no_grad(), ThreadPoolExecutor(max_workers=1) as prefetcher:
        batches = range(0, len(image_paths), batch_size)
        first_start = next(iter(batches), None)
        if first_start is None:
            return torch.empty((0, 0), dtype=storage_dtype)
        image_future = prefetcher.submit(
            _prepare_image_paths,
            image_paths[first_start : first_start + batch_size],
            image_size,
            torch.device("cpu"),
            num_workers,
        )
        for start in batches:
            images = image_future.result().to(device, non_blocking=True)
            next_start = start + batch_size
            if next_start < len(image_paths):
                image_future = prefetcher.submit(
                    _prepare_image_paths,
                    image_paths[next_start : next_start + batch_size],
                    image_size,
                    torch.device("cpu"),
                    num_workers,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                descriptor = model(images)
            descriptor = torch.nn.functional.normalize(descriptor.float(), p=2, dim=1)
            chunks.append(descriptor.to(dtype=storage_dtype).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0).contiguous()


def retrieve_candidates_from_descriptors(
    descriptors: np.ndarray | torch.Tensor,
    keyframe_indices: Sequence[int],
    config: LoopConfig,
    device: torch.device,
) -> tuple[list[LoopCandidate], dict[str, Any]]:
    descriptor_tensor = _prepare_descriptor_tensor(descriptors, device)
    if descriptor_tensor.shape[0] != len(keyframe_indices):
        raise ValueError("descriptor count must match keyframe_indices.")
    if descriptor_tensor.shape[0] < 2:
        return [], {"search_backend": "none", "descriptor_dtype": str(descriptor_tensor.dtype)}

    similarities = descriptor_tensor @ descriptor_tensor.T
    candidates, stats = _retrieve_candidates_from_similarity(similarities, keyframe_indices, config)
    stats.update(
        {
            "search_backend": "torch_full_pairwise",
            "descriptor_dtype": str(descriptor_tensor.dtype).replace("torch.", ""),
            "similarity_dtype": str(similarities.dtype).replace("torch.", ""),
        }
    )
    del descriptor_tensor, similarities
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates, stats


def _retrieve_candidates_from_similarity(
    similarities: torch.Tensor,
    keyframe_indices: Sequence[int],
    config: LoopConfig,
) -> tuple[list[LoopCandidate], dict[str, Any]]:
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarities must be a square matrix.")
    num_keyframes = int(similarities.shape[0])
    if num_keyframes != len(keyframe_indices):
        raise ValueError("similarity count must match keyframe_indices.")
    if num_keyframes < 2:
        return [], {"search_backend": "none"}

    top_k = min(max(1, int(config.retrieval_top_k)), num_keyframes)
    frame_indices = torch.as_tensor(keyframe_indices, device=similarities.device, dtype=torch.long)
    temporal_gap = torch.abs(frame_indices[:, None] - frame_indices[None, :])
    valid = temporal_gap > int(config.temporal_threshold)
    valid.fill_diagonal_(False)

    masked = similarities.masked_fill(~valid, -torch.inf)
    top_scores, top_indices = torch.topk(masked, k=top_k, dim=1, largest=True, sorted=True)
    candidates: list[LoopCandidate] = []
    for query_pos in range(num_keyframes):
        for rank in range(top_k):
            match_pos = int(top_indices[query_pos, rank].item())
            if match_pos < 0 or match_pos == query_pos:
                continue
            score = float(top_scores[query_pos, rank].item())
            if not math.isfinite(score) or score <= float(config.salad_score_threshold):
                break
            candidates.append(
                LoopCandidate(
                    query_pos=int(query_pos),
                    match_pos=int(match_pos),
                    query_frame=int(keyframe_indices[query_pos]),
                    match_frame=int(keyframe_indices[match_pos]),
                    score=score,
                    method="salad",
                )
            )

    return candidates, {
        "candidate_mode": "directed_query_to_match",
        "temporal_valid_directed_pairs": int(valid.sum().item()),
        "accepted_directed_candidates": int(len(candidates)),
    }


def _resolve_existing_path(raw: str, label: str) -> Path:
    path = Path(os.path.expanduser(str(raw))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        if device.index is None:
            return bool(torch.cuda.is_bf16_supported())
        with torch.cuda.device(device):
            return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _descriptor_storage_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if _cuda_bf16_supported(device) else torch.float32


def _prepare_descriptor_tensor(
    descriptors: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(descriptors, torch.Tensor):
        values = descriptors.to(device=device, dtype=torch.float32)
    else:
        values = torch.from_numpy(np.asarray(descriptors)).to(device=device, dtype=torch.float32)
    values = torch.nn.functional.normalize(values, p=2, dim=1)
    return values.to(dtype=_descriptor_storage_dtype(device)).contiguous()


def _load_dinov2(model_name: str, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    return _make_local_dinov2(model_name, state)


def _make_local_dinov2(model_name: str, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    register_tokens = int(state["register_tokens"].shape[1]) if "register_tokens" in state else 0
    specs = {
        "dinov2_vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "dinov2_vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "dinov2_vitl14": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "dinov2_vitg14": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
    }
    if model_name not in specs:
        raise ValueError(f"unsupported DINOv2 backbone: {model_name}")

    return DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=1.0,
        num_register_tokens=register_tokens,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **specs[model_name],
    )


def _sinkhorn_log_transport(
    log_rows: torch.Tensor,
    log_columns: torch.Tensor,
    scores: torch.Tensor,
    *,
    iterations: int = 3,
) -> torch.Tensor:
    row_update = torch.zeros_like(log_rows)
    column_update = torch.zeros_like(log_columns)
    for _ in range(iterations):
        row_update = log_rows - torch.logsumexp(scores + column_update.unsqueeze(1), dim=2)
        column_update = log_columns - torch.logsumexp(scores + row_update.unsqueeze(2), dim=1)
    return scores + row_update.unsqueeze(2) + column_update.unsqueeze(1)


def _cluster_probabilities(scores: torch.Tensor, dustbin: torch.Tensor) -> torch.Tensor:
    batch, clusters, patches = scores.shape
    augmented = torch.empty(batch, clusters + 1, patches, dtype=scores.dtype, device=scores.device)
    augmented[:, :clusters] = scores
    augmented[:, clusters] = dustbin
    normalizer = -torch.tensor(math.log(patches + clusters), device=scores.device)
    log_rows = normalizer.expand(clusters + 1).contiguous()
    log_columns = normalizer.expand(patches).contiguous()
    log_rows[-1] += math.log(max(1, patches - clusters))
    transport = _sinkhorn_log_transport(
        log_rows.expand(batch, -1),
        log_columns.expand(batch, -1),
        augmented,
    )
    return torch.exp(transport - normalizer)[:, :-1]


class SaladAggregator(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.num_channels = int(channels)
        self.num_clusters = 64
        self.cluster_dim = 128
        self.token_dim = 256
        self.token_features = torch.nn.Sequential(
            torch.nn.Linear(channels, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.token_dim),
        )
        self.cluster_features = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 512, 1),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 512, 1),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature_map, class_token = inputs
        local_features = self.cluster_features(feature_map).flatten(2)
        assignment = _cluster_probabilities(self.score(feature_map).flatten(2), self.dust_bin)
        assignment = assignment.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        local_features = local_features.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        pooled = torch.nn.functional.normalize(
            (local_features * assignment).sum(dim=-1),
            p=2,
            dim=1,
        ).flatten(1)
        token = torch.nn.functional.normalize(self.token_features(class_token), p=2, dim=-1)
        return torch.nn.functional.normalize(torch.cat((token, pooled), dim=-1), p=2, dim=-1)


class DinoBackbone(torch.nn.Module):
    CHANNELS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str, checkpoint: Path) -> None:
        super().__init__()
        if model_name not in self.CHANNELS:
            raise ValueError(f"unsupported DINOv2 backbone: {model_name}")
        self.num_channels = self.CHANNELS[model_name]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = _load_dinov2(model_name, state)
        self.model.load_state_dict(state, strict=True)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        class_token = tokens[:, 0]
        patch_start = 1 + int(getattr(self.model, "num_register_tokens", 0))
        feature_map = (
            tokens[:, patch_start:]
            .reshape(batch, height // 14, width // 14, self.num_channels)
            .permute(0, 3, 1, 2)
        )
        return feature_map, class_token


class SaladDescriptor(torch.nn.Module):
    def __init__(self, backbone_name: str, dino_checkpoint: Path) -> None:
        super().__init__()
        self.backbone = DinoBackbone(backbone_name, dino_checkpoint)
        self.aggregator = SaladAggregator(self.backbone.num_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(images))


def _load_salad_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("model.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    missing, _ = model.load_state_dict(cleaned, strict=False)
    missing_aggregator = [name for name in missing if name.startswith("aggregator.")]
    if missing_aggregator:
        raise RuntimeError(
            "SALAD checkpoint is missing aggregator parameters: "
            + ", ".join(missing_aggregator[:5])
        )


def _prepare_image_paths(
    image_paths: Sequence[str],
    image_size: tuple[int, int],
    device: torch.device,
    num_workers: int = 8,
) -> torch.Tensor:
    height, width = int(image_size[0]), int(image_size[1])
    def read_and_resize(path: str) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    workers = min(max(1, int(num_workers)), max(1, len(image_paths)))
    with ThreadPoolExecutor(max_workers=workers) as loader:
        frames = list(loader.map(read_and_resize, image_paths))
    tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2).to(device) / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std
