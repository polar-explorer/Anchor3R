from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from anchor3r.geometry.camera import build_camera_trajectories, decode_cam_map
from anchor3r.io.images import (
    PREPROCESS_MODES,
    TRAIN_PATCH_SIZE,
    TRAIN_PREPROCESS_SIZE,
    extract_video,
    list_sequences,
    load_images,
    save_dpt_outputs,
    save_geometry_outputs,
    save_results,
)
from anchor3r.loop import LoopConfig, build_augmented_sequence, retrieve_loop_candidates
from anchor3r.loop.io import save_loop_artifacts
from anchor3r.runtime.model import Anchor3RModel


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning,
)


def _log(message: str) -> None:
    print(f"[Anchor3R] {message}", flush=True)


def run_inference(cfg: dict[str, Any]) -> None:
    run_tic = time.perf_counter()
    device = _device(cfg.get("device", "cuda"))
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    infer_cfg = cfg.get("inference", {}) or {}
    out_cfg = cfg.get("output", {}) or {}
    infer_mode = str(infer_cfg.get("mode", "online")).strip().lower()
    if infer_mode not in {"online", "offline"}:
        raise ValueError("inference.mode must be 'online' or 'offline'.")
    loop_cfg = LoopConfig.from_mapping(cfg.get("loop", {}) or {})
    loop_cfg = loop_cfg.enabled_for_mode(infer_mode)

    checkpoint = _none(model_cfg.get("checkpoint"))
    if not checkpoint or not os.path.isfile(os.path.expanduser(checkpoint)):
        raise FileNotFoundError("Provide a released checkpoint with --checkpoint; see README.md#checkpoints.")
    strict_load = bool(model_cfg.get("strict_load", True))
    window_size = int(infer_cfg.get("window_size", 10))
    chunk_size = int(infer_cfg.get("chunk_size", 64))
    if window_size < 2 or chunk_size < 1:
        raise ValueError("window_size must be at least 2 and chunk_size must be positive.")
    preprocess = str(data_cfg.get("preprocess", "center")).strip().lower()
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"data.preprocess must be one of {PREPROCESS_MODES}, got: {preprocess}")
    if "camera_preprocess" in data_cfg:
        selected = "intrinsics" if _bool(data_cfg["camera_preprocess"]) else "center"
        if "preprocess" in data_cfg and preprocess != selected:
            raise ValueError("Conflicting data.camera_preprocess and legacy data.preprocess settings.")
        preprocess = selected
    use_amp = bool(infer_cfg.get("use_amp", device.type == "cuda")) and device.type == "cuda"
    amp_dtype = _dtype(infer_cfg.get("amp_dtype", "float16"))
    offload = bool(infer_cfg.get("offload_outputs_to_cpu", True))
    out_root = out_cfg.get("root", "outputs_anchor3r")
    save_geometry = _bool(out_cfg.get("save_geometry", False))
    save_geometry_world_points = _bool(out_cfg.get("save_geometry_world_points", True))
    save_geometry_confidence = _bool(out_cfg.get("save_geometry_confidence", True))
    save_dpt_map = _bool(out_cfg.get("save_dpt_map", False))
    save_dpt_conf = _bool(out_cfg.get("save_dpt_conf", False))
    image_num_workers = max(0, int(data_cfg.get("image_num_workers", 0)))
    prefetch_chunks = _bool(data_cfg.get("prefetch_chunks", False))
    pin_memory = _bool(data_cfg.get("pin_memory", device.type == "cuda"))
    if _bool(data_cfg.get("transfer_uint8", False)):
        raise ValueError("Remove data.transfer_uint8: inference requires float32 image transfer to preserve legacy evaluation preprocessing.")

    _log("starting inference")
    _log(f"device={device} cuda_available={torch.cuda.is_available()}")
    _log(f"checkpoint={checkpoint or '<none>'} strict_load={strict_load}")
    _log(
        "settings: "
        f"mode={infer_mode} window_size={window_size} chunk_size={chunk_size} "
        f"camera_preprocess={preprocess == 'intrinsics'} preprocess={preprocess} "
        f"size={TRAIN_PREPROCESS_SIZE} patch_size={TRAIN_PATCH_SIZE} "
        f"use_amp={use_amp} amp_dtype={amp_dtype} offload_outputs_to_cpu={offload}"
    )
    _log(
        f"output_root={out_root} save_geometry={save_geometry} "
        f"save_geometry_world_points={save_geometry_world_points} "
        f"save_geometry_confidence={save_geometry_confidence} "
        f"save_dpt_map={save_dpt_map} save_dpt_conf={save_dpt_conf}"
    )
    _log(
        "image loading: "
        f"num_workers={image_num_workers} prefetch_chunks={prefetch_chunks} "
        f"pin_memory={pin_memory} transfer_dtype=float32"
    )

    model_tic = time.perf_counter()
    _log("initializing model")
    model = Anchor3RModel(
        model_cfg.get("sampler_cfg", {}),
        checkpoint=checkpoint,
        strict_load=strict_load,
    )
    _log(f"model initialized on CPU in {time.perf_counter() - model_tic:.2f}s")

    move_tic = time.perf_counter()
    _log(f"moving model to {device}")
    model = model.to(device)
    _sync(device)
    _log(f"model moved to {device} in {time.perf_counter() - move_tic:.2f}s")
    model.eval()

    seq_tic = time.perf_counter()
    _log("discovering input sequences")
    sequences = _sequences_from_config(data_cfg)
    _log(
        f"discovered {len(sequences)} sequence(s) in {time.perf_counter() - seq_tic:.2f}s: "
        + ", ".join(f"{seq.name}({len(seq.paths)})" for seq in sequences)
    )
    os.makedirs(out_root, exist_ok=True)
    _log(f"ensured output root exists: {out_root}")

    for seq in sequences:
        seq_tic = time.perf_counter()
        _log(f"{seq.name}: start, frames={len(seq.paths)}")
        stream_paths = seq.paths
        augmented = None
        loop_candidates = []
        loop_retrieval_stats = {"enabled": False}
        if loop_cfg.enabled:
            _log(
                f"{seq.name}: offline loop retrieval start "
                f"keyframe_stride={loop_cfg.keyframe_stride} top_k={loop_cfg.retrieval_top_k}"
            )
            loop_candidates, loop_retrieval_stats = retrieve_loop_candidates(seq.paths, loop_cfg, device)
            augmented = build_augmented_sequence(
                seq.paths,
                loop_candidates,
                insert_position=loop_cfg.insert_position,
            )
            stream_paths = augmented.paths
            _log(
                f"{seq.name}: augmented sequence built "
                f"original={len(seq.paths)} augmented={len(stream_paths)} "
                f"inserted={sum(augmented.aug_is_inserted)} candidates={len(loop_candidates)}"
            )

        state = model.new_state()
        _log(f"{seq.name}: initialized streaming state")
        target_hw = None
        chunk_cam_maps = []
        fov_chunks = []
        dpt_map_chunks = [] if save_dpt_map or save_geometry else None
        dpt_conf_chunks = [] if save_dpt_conf or (save_geometry and save_geometry_confidence) else None
        elapsed = 0.0

        chunks = _chunk_ranges(len(stream_paths), window_size, chunk_size)
        _log(f"{seq.name}: planned {len(chunks)} chunk(s)")
        prefetch_executor = ThreadPoolExecutor(max_workers=1) if prefetch_chunks and len(chunks) > 1 else None
        prefetch_future = None
        with torch.inference_mode():
            for chunk_idx, (start, end) in enumerate(chunks):
                _log(f"{seq.name}: chunk {chunk_idx + 1}/{len(chunks)}")
                if prefetch_future is None:
                    images_cpu, target_hw = _load_chunk_images(
                        stream_paths[start:end],
                        target_hw=target_hw,
                        image_num_workers=image_num_workers,
                        preprocess=preprocess,
                    )
                else:
                    images_cpu, target_hw = prefetch_future.result()
                    prefetch_future = None

                if prefetch_executor is not None and chunk_idx + 1 < len(chunks):
                    next_start, next_end = chunks[chunk_idx + 1]
                    prefetch_future = prefetch_executor.submit(
                        _load_chunk_images,
                        stream_paths[next_start:next_end],
                        target_hw=target_hw,
                        image_num_workers=image_num_workers,
                        preprocess=preprocess,
                    )

                if pin_memory and device.type == "cuda" and not images_cpu.is_pinned():
                    images_cpu = images_cpu.pin_memory()

                images = images_cpu.to(device, non_blocking=True)
                _sync(device)
                current_window = end - start if chunk_idx == 0 else window_size

                forward_tic = time.perf_counter()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    out = model.forward_chunk(
                        images,
                        window_size=current_window,
                        chunk_idx=chunk_idx,
                        state=state,
                    )
                _sync(device)
                forward_time = time.perf_counter() - forward_tic
                elapsed += forward_time

                cam = out["chunk_cam_map"].detach().float()
                chunk_cam_maps.append(cam.cpu() if offload else cam)
                if "fov" in out:
                    fov = out["fov"].detach().float()
                    fov_chunks.append(fov.cpu() if offload else fov)
                if dpt_map_chunks is not None:
                    if "dpt_map" not in out:
                        raise RuntimeError("output.save_dpt_map is true, but the model did not return dpt_map.")
                    dpt_map_chunks.append(out["dpt_map"].detach().float().cpu())
                if dpt_conf_chunks is not None:
                    if "dpt_cnf" not in out:
                        raise RuntimeError("output.save_dpt_conf is true, but the model did not return dpt_cnf.")
                    dpt_conf_chunks.append(out["dpt_cnf"].detach().float().cpu())
                model.advance_state(state, is_last=chunk_idx == len(chunks) - 1)
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)

        assert target_hw is not None
        _log(f"{seq.name}: building camera trajectories")
        traj_tic = time.perf_counter()
        trajectory_maps = build_camera_trajectories(
            chunk_cam_maps,
            fov_chunks,
            num_frames=len(stream_paths),
            window_size=min(window_size, len(stream_paths)),
            aug_to_orig=None if augmented is None else augmented.aug_to_orig,
            orig_to_main_aug=None if augmented is None else augmented.orig_to_main_aug,
            aug_is_inserted=None if augmented is None else augmented.aug_is_inserted,
            motion_edge_weight=loop_cfg.motion_edge_weight,
            loop_edge_weight=loop_cfg.loop_edge_weight,
            loop_min_frame_separation=loop_cfg.temporal_threshold,
        )
        _log(f"{seq.name}: camera trajectories built in {time.perf_counter() - traj_tic:.2f}s")

        decode_tic = time.perf_counter()
        _log(f"{seq.name}: decoding camera maps")
        online_w2c, intri = decode_cam_map(trajectory_maps["online"], target_hw)
        offline_w2c, _ = decode_cam_map(trajectory_maps["offline"], target_hw)
        _log(f"{seq.name}: camera maps decoded in {time.perf_counter() - decode_tic:.2f}s")

        seq_out = os.path.join(out_root, seq.name)
        pose_dir_name = "poses_offline" if infer_mode == "offline" else "poses"
        if augmented is not None:
            save_loop_artifacts(
                seq_out,
                config=loop_cfg,
                candidates=loop_candidates,
                retrieval_stats=loop_retrieval_stats,
                augmented=augmented,
                edge_summary=trajectory_maps.get("edge_summary"),
                motion_edges=trajectory_maps.get("motion_edges"),
            )
        save_tic = time.perf_counter()
        _log(f"{seq.name}: saving pose results to {os.path.join(seq_out, pose_dir_name)}")
        save_results(
            seq_out,
            intri=intri,
            online_w2c=None if loop_cfg.enabled else online_w2c,
            offline_w2c=offline_w2c,
            pose_dir_name=pose_dir_name,
        )
        _log(f"{seq.name}: pose results saved in {time.perf_counter() - save_tic:.2f}s")
        if dpt_map_chunks is not None or dpt_conf_chunks is not None:
            dpt_map = torch.cat(dpt_map_chunks, dim=1) if dpt_map_chunks is not None else None
            dpt_conf = torch.cat(dpt_conf_chunks, dim=1) if dpt_conf_chunks is not None else None
            if augmented is not None:
                main_indices = torch.as_tensor(augmented.orig_to_main_aug, dtype=torch.long)
                if dpt_map is not None:
                    dpt_map = dpt_map.index_select(1, main_indices)
                if dpt_conf is not None:
                    dpt_conf = dpt_conf.index_select(1, main_indices)
            dpt_tic = time.perf_counter()
            if save_dpt_map or save_dpt_conf:
                _log(f"{seq.name}: saving DPT outputs")
                save_dpt_outputs(
                    seq_out,
                    image_hw=target_hw,
                    dpt_map=dpt_map if save_dpt_map else None,
                    dpt_conf=dpt_conf if save_dpt_conf else None,
                )
                _log(f"{seq.name}: DPT outputs saved in {time.perf_counter() - dpt_tic:.2f}s")
            if save_geometry:
                geom_tic = time.perf_counter()
                _log(f"{seq.name}: saving geometry outputs")
                if dpt_map is None:
                    raise RuntimeError("output.save_geometry is true, but no dpt_map was collected.")
                save_geometry_outputs(
                    seq_out,
                    image_paths=seq.paths,
                    image_hw=target_hw,
                    intri=intri,
                    dpt_map=dpt_map,
                    dpt_conf=dpt_conf,
                    online_w2c=None if loop_cfg.enabled else online_w2c,
                    offline_w2c=offline_w2c,
                    pose_dir_name=pose_dir_name,
                    save_world_points=save_geometry_world_points,
                    save_confidence=save_geometry_confidence,
                )
                _log(f"{seq.name}: geometry outputs saved in {time.perf_counter() - geom_tic:.2f}s")
        fps = len(stream_paths) / max(elapsed, 1e-12)
        _log(
            f"{seq.name}: done, saved to {seq_out}, forward={elapsed:.2f}s, "
            f"fps={fps:.2f}, streamed_frames={len(stream_paths)}, total={time.perf_counter() - seq_tic:.2f}s"
        )

    _log(f"all inference done in {time.perf_counter() - run_tic:.2f}s")


def _sequences_from_config(data_cfg: dict[str, Any]):
    max_frames = _int_or_none(data_cfg.get("max_frames"))
    sequence_name = _none(data_cfg.get("sequence_name"))
    video = _none(data_cfg.get("video"))
    if video:
        _log(
            f"extracting video={video} cache={data_cfg.get('video_cache', 'video_cache')} "
            f"stride={int(data_cfg.get('video_stride', 1))} max_frames={max_frames}"
        )
        sequences = [
            extract_video(
                video,
                data_cfg.get("video_cache", "video_cache"),
                stride=int(data_cfg.get("video_stride", 1)),
                max_frames=max_frames,
            )
        ]
    else:
        images = _none(data_cfg.get("images"))
        if not images:
            raise ValueError("Set data.images or data.video.")
        _log(f"listing images={images} max_frames={max_frames}")
        sequences = list_sequences(images, max_frames=max_frames)

    if sequence_name:
        if len(sequences) != 1:
            raise ValueError("data.sequence_name requires exactly one input sequence.")
        _log(f"renaming sequence {sequences[0].name} -> {sequence_name}")
        sequences[0].name = sequence_name
    return sequences


def _chunk_ranges(num_frames: int, window_size: int, chunk_size: int) -> list[tuple[int, int]]:
    if num_frames <= window_size:
        return [(0, num_frames)]
    ranges = [(0, window_size)]
    start = window_size
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        ranges.append((start, end))
        start = end
    return ranges


def _load_chunk_images(
    paths,
    *,
    target_hw: tuple[int, int] | None,
    image_num_workers: int,
    preprocess: str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    return load_images(
        paths,
        target_hw=target_hw,
        num_workers=image_num_workers,
        preprocess=preprocess,
    )


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def _dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown amp dtype: {name}")


def _none(value):
    return None if value in (None, "", "null") else value


def _int_or_none(value) -> int | None:
    value = _none(value)
    return None if value is None else int(value)


def _bool(value) -> bool:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off", "", "null", "none"):
            return False
    return bool(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
