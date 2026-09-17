from __future__ import annotations

import os
import time
from typing import Any

import torch

from anchor3r.utils.core import dotdict
from anchor3r.model.sampler import MultiViewStreamSampler
from anchor3r.model.stream_aggregator import KVCache


def _to_dotdict(value: Any, dotdict_cls):
    if isinstance(value, dict):
        return dotdict_cls({k: _to_dotdict(v, dotdict_cls) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_dotdict(v, dotdict_cls) for v in value]
    return value


def _log(message: str) -> None:
    print(f"[Anchor3RModel] {message}", flush=True)


class Anchor3RModel(torch.nn.Module):
    def __init__(
        self,
        sampler_cfg: dict[str, Any],
        *,
        checkpoint: str | None = None,
        strict_load: bool = True,
    ):
        super().__init__()
        self._dotdict = dotdict
        self._kv_cache_cls = KVCache

        tic = time.perf_counter()
        _log("building sampler")
        self.sampler = MultiViewStreamSampler(
            **_to_dotdict(sampler_cfg or {}, dotdict),
        )
        self.sampler.eval()
        _log(f"sampler built in {time.perf_counter() - tic:.2f}s")
        if checkpoint:
            self.load_checkpoint(checkpoint, strict=strict_load)
        else:
            _log("no checkpoint provided")

    def load_checkpoint(self, checkpoint: str, *, strict: bool = True) -> Any:
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)

        total_tic = time.perf_counter()
        _log(f"loading checkpoint: {checkpoint}")
        load_tic = time.perf_counter()
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            _log(f"torch.load failed after {time.perf_counter() - load_tic:.2f}s: {type(exc).__name__}: {exc}")
            raise
        _log(f"checkpoint read in {time.perf_counter() - load_tic:.2f}s")
        if isinstance(state, dict):
            for key in ("model", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(state)!r}")

        load_state_tic = time.perf_counter()
        try:
            result = self.sampler.load_state_dict(state, strict=strict)
        except Exception as exc:
            _log(
                "load_state_dict failed after "
                f"{time.perf_counter() - load_state_tic:.2f}s: {type(exc).__name__}: {exc}"
            )
            raise
        missing = getattr(result, "missing_keys", ())
        unexpected = getattr(result, "unexpected_keys", ())
        _log(
            "checkpoint loaded in "
            f"{time.perf_counter() - total_tic:.2f}s "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        return result

    def new_state(self) -> dict[str, Any]:
        depth = len(self.sampler.stream_aggregator.global_blocks)
        return {
            "global_kv_caches": [self._kv_cache_cls() for _ in range(depth)],
            "frame_kv_caches": [self._kv_cache_cls() for _ in range(depth)],
            "pose_window_kv_caches": [self._kv_cache_cls() for _ in range(depth)],
        }

    def advance_state(self, state: dict[str, Any], *, is_last: bool) -> None:
        for global_kv, frame_kv, pose_window_kv in zip(
            state["global_kv_caches"],
            state["frame_kv_caches"],
            state["pose_window_kv_caches"],
        ):
            global_kv.kv_ori = None if is_last else global_kv.kv_new
            frame_kv.kv_ori = None if is_last else frame_kv.kv_new
            pose_window_kv.kv_ori = None if is_last else pose_window_kv.kv_new
            global_kv.kv_new = None
            frame_kv.kv_new = None
            pose_window_kv.kv_new = None

    def forward_chunk(
        self,
        images: torch.Tensor,
        *,
        window_size: int,
        chunk_idx: int,
        state: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B,S,3,H,W], got {tuple(images.shape)}")
        if images.shape[0] != 1:
            raise ValueError("Anchor3R currently supports batch size 1.")

        batch = self._make_batch(images, window_size, chunk_idx, state)
        out_batch = self.sampler(batch)
        out = out_batch.output if hasattr(out_batch, "output") else out_batch
        result = {"chunk_cam_map": out.chunk_cam_map}
        if "fov" in out:
            result["fov"] = out.fov
        if "dpt_map" in out:
            result["dpt_map"] = out.dpt_map
        if "dpt_cnf" in out:
            result["dpt_cnf"] = out.dpt_cnf
        return result

    def _make_batch(self, images: torch.Tensor, window_size: int, chunk_idx: int, state: dict[str, Any]):
        dotdict = self._dotdict
        b, s, c, h, w = images.shape
        batch = dotdict()
        batch.output = dotdict()
        batch.meta = dotdict()
        batch.rgb = images.permute(0, 1, 3, 4, 2).reshape(b, s, h * w, c)
        batch.global_kv_caches = state["global_kv_caches"]
        batch.frame_kv_caches = state["frame_kv_caches"]
        batch.pose_window_kv_caches = state["pose_window_kv_caches"]
        batch.window_size = int(window_size)
        batch.chunk_idx = int(chunk_idx)
        batch.first_chunk_full = True
        batch.meta.H = torch.as_tensor([h], device=images.device, dtype=torch.long)
        batch.meta.W = torch.as_tensor([w], device=images.device, dtype=torch.long)
        return batch
