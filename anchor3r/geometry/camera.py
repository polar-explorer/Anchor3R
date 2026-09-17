from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.spatial.transform import Rotation as R
from sksparse.cholmod import analyze, cholesky


@dataclass
class CameraGraph:
    online_abs: torch.Tensor
    valid_relative: torch.Tensor
    fov: torch.Tensor
    online_cam_map: torch.Tensor | None = None


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1).clamp_min(1e-12)
    m = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    shape = m.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m.reshape(shape + (9,)), dim=-1)
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    quat = candidates[torch.nn.functional.one_hot(q_abs.argmax(-1), 4) > 0.5].reshape(shape + (4,))
    quat = quat[..., [1, 2, 3, 0]]
    return torch.where(quat[..., 3:4] < 0, -quat, quat)


def build_camera_trajectories(
    chunk_cam_maps: list[torch.Tensor],
    fov_chunks: list[torch.Tensor],
    *,
    num_frames: int,
    window_size: int,
    aug_to_orig: Sequence[int] | None = None,
    orig_to_main_aug: Sequence[int] | None = None,
    aug_is_inserted: Sequence[bool] | None = None,
    motion_edge_weight: float = 1.0,
    loop_edge_weight: float = 1.0,
    loop_min_frame_separation: int = 1,
    explicit_graph_max_iters: int = 100,
) -> dict[str, Any]:
    graph = stitch_camera_graph(
        chunk_cam_maps,
        fov_chunks,
        num_frames=num_frames,
        window_size=window_size,
    )
    if aug_to_orig is not None or orig_to_main_aug is not None or aug_is_inserted is not None:
        return _build_augmented_camera_trajectories(
            graph,
            aug_to_orig=aug_to_orig,
            orig_to_main_aug=orig_to_main_aug,
            aug_is_inserted=aug_is_inserted,
            motion_edge_weight=motion_edge_weight,
            loop_edge_weight=loop_edge_weight,
            loop_min_frame_separation=loop_min_frame_separation,
            max_iters=explicit_graph_max_iters,
        )
    return _build_standard_camera_trajectories(graph, max_iters=explicit_graph_max_iters)


def stitch_camera_graph(
    chunk_cam_maps: list[torch.Tensor],
    fov_chunks: list[torch.Tensor],
    *,
    num_frames: int,
    window_size: int,
) -> CameraGraph:
    if not chunk_cam_maps:
        raise ValueError("No camera chunks to stitch.")

    first = chunk_cam_maps[0].float()
    device = first.device
    dtype = torch.float32
    fov = _collect_fov(fov_chunks, chunk_cam_maps, num_frames, window_size, device)

    if len(chunk_cam_maps) == 1:
        cam = first.reshape(1, -1, window_size, first.shape[-1])
        w2c = _cam_to_w2c(cam[..., :7])
        online_cam_map = torch.cat([cam[:, -1, :, :7].to(dtype=dtype), fov], dim=-1)
        return CameraGraph(
            online_abs=w2c[:, -1].to(dtype=dtype),
            valid_relative=w2c[:, -1:].to(dtype=dtype),
            fov=fov,
            online_cam_map=online_cam_map,
        )

    online_abs = torch.empty(1, num_frames, 3, 4, device=device, dtype=dtype)
    valid_relative = torch.empty(1, num_frames - window_size + 1, window_size, 3, 4, device=device, dtype=dtype)
    ptr = valid_ptr = 0
    for chunk_idx, chunk in enumerate(chunk_cam_maps):
        cam = chunk.to(device=device, dtype=dtype)
        cam = cam.reshape(1, -1, window_size, cam.shape[-1])
        w2c = _cam_to_w2c(cam[..., :7])
        if chunk_idx == 0:
            valid_relative[:, valid_ptr] = w2c[:, -1]
            valid_ptr += 1
            online_abs[:, :window_size] = w2c[:, -1]
            ptr = window_size
            continue

        for row in range(w2c.shape[1]):
            rel = w2c[:, row]
            valid_relative[:, valid_ptr] = rel
            valid_ptr += 1
            start = ptr - window_size + 1

            cached = online_abs[:, start:ptr]
            cached_r = cached[..., :3, :3]
            cached_t = cached[..., :3, 3:]
            rotation_candidates = rel[:, :-1, :3, :3].transpose(-1, -2) @ cached_r
            rotation, _ = get_median_rotations(rotation_candidates)
            cached_c = -cached_r.transpose(-1, -2) @ cached_t
            relative_world_t = cached_r.transpose(-1, -2) @ rel[:, :-1, :3, 3:]
            center = torch.median((cached_c + relative_world_t).reshape(1, window_size - 1, 3), dim=1).values
            translation = -rotation @ center[:, None, :, None]
            online_abs[:, ptr] = torch.cat([rotation, translation], dim=-1)[:, 0]

            ptr += 1

    if ptr != num_frames or valid_ptr != num_frames - window_size + 1:
        raise RuntimeError(f"Stitched {ptr} poses/{valid_ptr} windows, expected {num_frames}/{num_frames - window_size + 1}.")

    return CameraGraph(online_abs=online_abs, valid_relative=valid_relative, fov=fov)


def _build_standard_camera_trajectories(graph: CameraGraph, *, max_iters: int = 100) -> dict[str, Any]:
    if graph.online_cam_map is not None and graph.valid_relative.shape[1] == 1:
        return {"online": graph.online_cam_map, "offline": graph.online_cam_map}

    trajectories = {"online": _relative_cam_map(graph.online_abs, graph.fov)}
    if graph.valid_relative.shape[1] == 1:
        trajectories["offline"] = trajectories["online"]
        return trajectories

    rel = graph.valid_relative[0]
    offline_r = rotation_averaging(rel[..., :3, :3], graph.online_abs[0, :, :3, :3], max_iters=max_iters)
    offline_r = offline_r @ offline_r[0].transpose(-1, -2)
    offline_c, _ = position_averaging(rel[..., :3, 3:], offline_r)
    offline_t = (-offline_r @ (offline_c[:, :, None] - offline_c[:1, :, None])).squeeze(-1)
    offline_abs = torch.cat([offline_r, offline_t[..., None]], dim=-1).unsqueeze(0)
    trajectories["offline"] = _relative_cam_map(offline_abs, graph.fov)
    return trajectories


def _build_augmented_camera_trajectories(
    graph: CameraGraph,
    *,
    aug_to_orig: Sequence[int] | None,
    orig_to_main_aug: Sequence[int] | None,
    aug_is_inserted: Sequence[bool] | None,
    motion_edge_weight: float,
    loop_edge_weight: float,
    loop_min_frame_separation: int,
    max_iters: int,
) -> dict[str, Any]:
    if aug_to_orig is None or orig_to_main_aug is None or aug_is_inserted is None:
        raise ValueError("aug_to_orig, orig_to_main_aug, and aug_is_inserted must be provided together.")
    aug_to_orig = [int(x) for x in aug_to_orig]
    orig_to_main_aug = [int(x) for x in orig_to_main_aug]
    aug_is_inserted = [bool(x) for x in aug_is_inserted]
    if len(aug_to_orig) != graph.online_abs.shape[1]:
        raise ValueError(
            f"aug_to_orig length {len(aug_to_orig)} does not match augmented frames {graph.online_abs.shape[1]}."
        )
    if len(aug_is_inserted) != len(aug_to_orig):
        raise ValueError("aug_is_inserted length must match aug_to_orig.")
    if not orig_to_main_aug:
        raise ValueError("orig_to_main_aug must not be empty.")
    if min(orig_to_main_aug) < 0 or max(orig_to_main_aug) >= len(aug_to_orig):
        raise IndexError("orig_to_main_aug contains an augmented index outside the sequence.")

    device = graph.online_abs.device
    main_aug = torch.as_tensor(orig_to_main_aug, device=device, dtype=torch.long)
    original_abs = graph.online_abs.index_select(1, main_aug)
    original_fov = graph.fov.index_select(1, main_aug)
    trajectories: dict[str, Any] = {
        "online": _relative_cam_map(original_abs, original_fov),
    }

    edge_data = _collect_augmented_motion_edges(
        graph.valid_relative[0],
        aug_to_orig=aug_to_orig,
        aug_is_inserted=aug_is_inserted,
        num_orig_frames=len(orig_to_main_aug),
        motion_edge_weight=motion_edge_weight,
        loop_edge_weight=loop_edge_weight,
        loop_min_frame_separation=loop_min_frame_separation,
    )
    trajectories["edge_summary"] = edge_data["summary"]
    trajectories["motion_edges"] = edge_data["records"]

    if edge_data["src"].numel() == 0:
        trajectories["offline"] = trajectories["online"]
        return trajectories

    fix_index = min(graph.valid_relative.shape[2] - 1, len(orig_to_main_aug) - 1)
    offline_r = rotation_averaging_edges(
        edge_data["src"],
        edge_data["dst"],
        edge_data["rel_r"],
        original_abs[0, :, :3, :3],
        edge_weights=edge_data["weights"],
        fix_index=fix_index,
        max_iters=max_iters,
    )
    offline_r = offline_r @ offline_r[0].transpose(-1, -2)
    offline_c, _ = position_averaging_edges(
        edge_data["src"],
        edge_data["dst"],
        edge_data["rel_t"],
        offline_r,
        edge_weights=edge_data["weights"],
        fix_index=fix_index,
    )
    offline_t = (-offline_r @ (offline_c[:, :, None] - offline_c[:1, :, None])).squeeze(-1)
    offline_abs = torch.cat([offline_r, offline_t[..., None]], dim=-1).unsqueeze(0)
    trajectories["offline"] = _relative_cam_map(offline_abs, original_fov)
    return trajectories


def _collect_augmented_motion_edges(
    valid_relative: torch.Tensor,
    *,
    aug_to_orig: Sequence[int],
    aug_is_inserted: Sequence[bool],
    num_orig_frames: int,
    motion_edge_weight: float,
    loop_edge_weight: float,
    loop_min_frame_separation: int,
) -> dict[str, Any]:
    num_windows, window_size = valid_relative.shape[:2]
    src_indices: list[int] = []
    dst_indices: list[int] = []
    rel_rotations: list[torch.Tensor] = []
    rel_translations: list[torch.Tensor] = []
    weights: list[float] = []
    records: list[dict[str, Any]] = []
    skipped_same_orig = 0
    skipped_inserted_near = 0
    motion_edges = 0
    loop_edges = 0

    for window_idx in range(num_windows):
        dst_aug = window_idx + window_size - 1
        if dst_aug >= len(aug_to_orig):
            raise IndexError("destination augmented index is outside aug_to_orig.")
        dst_orig = int(aug_to_orig[dst_aug])
        dst_inserted = bool(aug_is_inserted[dst_aug])
        for offset in range(window_size - 1):
            src_aug = window_idx + offset
            src_orig = int(aug_to_orig[src_aug])
            src_inserted = bool(aug_is_inserted[src_aug])
            if src_orig == dst_orig:
                skipped_same_orig += 1
                continue

            has_inserted_endpoint = src_inserted or dst_inserted
            if has_inserted_endpoint:
                if abs(src_orig - dst_orig) < int(loop_min_frame_separation):
                    skipped_inserted_near += 1
                    continue
                kind = "loop"
                weight = float(loop_edge_weight)
                loop_edges += 1
            else:
                kind = "motion"
                weight = float(motion_edge_weight)
                motion_edges += 1
            if weight <= 0:
                continue

            rel = valid_relative[window_idx, offset]
            src_indices.append(src_orig)
            dst_indices.append(dst_orig)
            rel_rotations.append(rel[:3, :3])
            rel_translations.append(rel[:3, 3:])
            weights.append(weight)
            records.append(
                {
                    "src_orig": src_orig,
                    "dst_orig": dst_orig,
                    "src_aug": int(src_aug),
                    "dst_aug": int(dst_aug),
                    "src_inserted": src_inserted,
                    "dst_inserted": dst_inserted,
                    "kind": kind,
                    "direction": "src<-dst",
                    "weight": weight,
                }
            )

    device = valid_relative.device
    if src_indices:
        src = torch.as_tensor(src_indices, device=device, dtype=torch.long)
        dst = torch.as_tensor(dst_indices, device=device, dtype=torch.long)
        rel_r = torch.stack(rel_rotations).to(device=device, dtype=torch.float32)
        rel_t = torch.stack(rel_translations).to(device=device, dtype=torch.float32)
        weight_tensor = torch.as_tensor(weights, device=device, dtype=torch.float32)
    else:
        src = torch.empty(0, device=device, dtype=torch.long)
        dst = torch.empty(0, device=device, dtype=torch.long)
        rel_r = valid_relative.new_empty((0, 3, 3), dtype=torch.float32)
        rel_t = valid_relative.new_empty((0, 3, 1), dtype=torch.float32)
        weight_tensor = valid_relative.new_empty((0,), dtype=torch.float32)

    if src.numel() and (int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= num_orig_frames or int(dst.max()) >= num_orig_frames):
        raise IndexError("Mapped motion graph edge is outside the original sequence.")

    return {
        "src": src,
        "dst": dst,
        "rel_r": rel_r,
        "rel_t": rel_t,
        "weights": weight_tensor,
        "records": records,
        "summary": {
            "num_edges": int(len(records)),
            "num_motion_edges": int(motion_edges),
            "num_loop_edges": int(loop_edges),
            "num_skipped_same_orig": int(skipped_same_orig),
            "num_skipped_inserted_near": int(skipped_inserted_near),
        },
    }


def decode_cam_map(cam_map: torch.Tensor, image_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_hw
    rot = quat_to_mat(cam_map[..., 3:7])
    trans = cam_map[..., :3]
    w2c = torch.cat([rot, trans[..., None]], dim=-1)
    fov = cam_map[..., 7:9].clamp_min(1e-5)
    intri = cam_map.new_zeros(cam_map.shape[:-1] + (3, 3))
    intri[..., 0, 0] = width / 2.0 / torch.tan(fov[..., 1] / 2.0).clamp_min(1e-5)
    intri[..., 1, 1] = height / 2.0 / torch.tan(fov[..., 0] / 2.0).clamp_min(1e-5)
    intri[..., 0, 2] = width / 2.0
    intri[..., 1, 2] = height / 2.0
    intri[..., 2, 2] = 1.0
    return w2c, intri


def _cam_to_w2c(cam: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat_to_mat(cam[..., 3:7]), cam[..., :3, None]], dim=-1)


def _relative_cam_map(abs_w2c: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    rot = abs_w2c[..., :3, :3]
    trans = abs_w2c[..., :3, 3:]
    center = -rot.transpose(-1, -2) @ trans
    rel_t = (-rot @ (center - center[:, :1])).squeeze(-1)
    rel_q = mat_to_quat(rot @ rot[:, :1].transpose(-1, -2))
    return torch.cat([rel_t, rel_q, fov], dim=-1)


def _collect_fov(
    fov_chunks: list[torch.Tensor],
    chunk_cam_maps: list[torch.Tensor],
    num_frames: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    if fov_chunks:
        fov = torch.cat([x.to(device=device, dtype=torch.float32) for x in fov_chunks], dim=1)
        if fov.shape[1] != num_frames:
            raise RuntimeError(f"Collected {fov.shape[1]} FoV values, expected {num_frames}.")
        return fov

    first = chunk_cam_maps[0]
    if first.shape[-1] >= 9 and len(chunk_cam_maps) == 1:
        return first.reshape(1, -1, window_size, first.shape[-1])[:, -1, :, 7:9].to(device=device, dtype=torch.float32)
    return torch.ones(1, num_frames, 2, device=device, dtype=torch.float32)

def _median_midpoint(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    中位数（偶数长度时取中间两项平均），与你原实现一致。
    """
    xs, _ = torch.sort(x, dim=dim)
    n = xs.size(dim)
    mid = n // 2
    if n % 2 == 1:
        return xs.select(dim, mid)
    else:
        return 0.5 * (xs.select(dim, mid - 1) + xs.select(dim, mid))


def _hat(w: torch.Tensor) -> torch.Tensor:
    """
    w(...,3) -> W(...,3,3)
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    return torch.stack(
        (
            torch.stack((O, -wz,  wy), dim=-1),
            torch.stack((wz,  O, -wx), dim=-1),
            torch.stack((-wy, wx,  O), dim=-1),
        ),
        dim=-2,
    )


def _so3_exp(w: torch.Tensor, eye3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    exp: so(3) 向量 w(...,3) -> SO(3) 矩阵 R(...,3,3)
    Rodrigues 公式 + 小角度稳定
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (...,1)
    W = _hat(w)  # (...,3,3)

    # broadcast I to (...,3,3)
    I = eye3.view((1,) * (W.ndim - 2) + (3, 3)).expand_as(W)

    theta2 = theta * theta
    small = theta < eps

    # a = sinθ/θ, b = (1-cosθ)/θ^2 with stable series for small θ
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2,
    )

    return I + a[..., None] * W + b[..., None] * (W @ W)


def _so3_log(R: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    log: SO(3) 矩阵 R(...,3,3) -> (w(...,3), theta(...))
    w = angle * axis
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...)

    vee = torch.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        dim=-1,
    )  # (...,3)

    sin_theta = torch.sin(theta)
    k = theta / (2.0 * sin_theta + eps)
    w_general = k[..., None] * vee

    # small-angle: w ≈ 0.5 * vee
    small = theta < 1e-4
    w_small = 0.5 * vee

    # near pi: recover axis from diagonal (avoid division by tiny sinθ)
    near_pi = (math.pi - theta) < 1e-4
    x = torch.sqrt(torch.clamp((R[..., 0, 0] + 1.0) * 0.5, min=0.0))
    y = torch.sqrt(torch.clamp((R[..., 1, 1] + 1.0) * 0.5, min=0.0))
    z = torch.sqrt(torch.clamp((R[..., 2, 2] + 1.0) * 0.5, min=0.0))
    axis = torch.stack((x, y, z), dim=-1)

    sx = torch.sign(R[..., 2, 1] - R[..., 1, 2])
    sy = torch.sign(R[..., 0, 2] - R[..., 2, 0])
    sz = torch.sign(R[..., 1, 0] - R[..., 0, 1])
    axis = axis * torch.stack((sx, sy, sz), dim=-1)

    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    w_pi = theta[..., None] * axis

    w = torch.where(small[..., None], w_small, w_general)
    w = torch.where(near_pi[..., None], w_pi, w)
    return w, theta


@torch.no_grad()
def get_median_rotations(
    rotations: torch.Tensor,                 # (B,W,3,3)
    times: int = 3,
    max_iters: int = 100,
    eps: float = 1e-5,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched median rotation for input rotations (B,W,3,3), taking median over dim=1 (W).

    Returns:
      best_R: (B,1,3,3)
      final_angle_errors: (B,W)
    """
    R = torch.as_tensor(rotations).to(dtype=dtype)
    assert R.ndim == 4 and R.shape[-2:] == (3, 3), "rotations must be (B,W,3,3)"
    B, W = R.shape[0], R.shape[1]
    assert W > 0

    device = R.device
    eye3 = torch.eye(3, device=device, dtype=R.dtype)

    # ids: (B,times)  每个batch独立随机初值（允许重复，和C++一致）
    ids = torch.randint(0, W, (B, times), device=device, generator=generator)

    # gather initial R_med: (B,times,3,3)
    R_med = R.gather(
        dim=1,
        index=ids[..., None, None].expand(B, times, 3, 3),
    ).clone()

    # record median error for each candidate when it converges
    median_err = torch.full((B, times), float("inf"), device=device, dtype=R.dtype)
    converged = torch.zeros((B, times), device=device, dtype=torch.bool)

    # main loop
    for _ in range(max_iters):
        # residual_R: (B,times,W,3,3) = R[b,w] @ R_med[b,t]^T
        R_med_T = R_med.transpose(-1, -2)  # (B,times,3,3)
        residual_R = torch.einsum("bwij,btjk->btwik", R, R_med_T)

        residual_vec, angle_errors = _so3_log(residual_R)  # (B,times,W,3), (B,times,W)

        # median residual vector over W (dim=2): -> (B,times,3)
        med0 = _median_midpoint(residual_vec[..., 0], dim=2)
        med1 = _median_midpoint(residual_vec[..., 1], dim=2)
        med2 = _median_midpoint(residual_vec[..., 2], dim=2)
        med_vec = torch.stack((med0, med1, med2), dim=-1)  # (B,times,3)

        norm = torch.linalg.norm(med_vec, dim=-1)  # (B,times)
        newly_converged = (norm < eps) & (~converged)

        # 对 newly_converged 记录 score：median(angle_errors)
        if newly_converged.any():
            med_ang = _median_midpoint(angle_errors, dim=2)  # (B,times)
            median_err = torch.where(newly_converged, med_ang, median_err)
            converged = converged | newly_converged

        # 全部收敛则提前退出
        if converged.all():
            break

        # update only for not-yet-converged
        active = ~converged
        if active.any():
            dR = _so3_exp(med_vec, eye3=eye3)  # (B,times,3,3)
            R_med_new = torch.einsum("btij,btjk->btik", dR, R_med)
            R_med = torch.where(active[..., None, None], R_med_new, R_med)

    # choose best per batch among converged candidates
    min_err, best_idx = torch.min(median_err, dim=1)  # (B,), (B,)
    has_solution = torch.isfinite(min_err)

    best_R_cand = R_med[torch.arange(B, device=device), best_idx]  # (B,3,3)
    fallback_R = R[:, 0]  # (B,3,3)
    best_R = torch.where(has_solution[:, None, None], best_R_cand, fallback_R)

    # final angle errors for chosen best_R: (B,W)
    best_R_T = best_R.transpose(-1, -2)  # (B,3,3)
    residual_final = torch.einsum("bwij,bjk->bwik", R, best_R_T)  # (B,W,3,3)
    _, final_angle_errors = _so3_log(residual_final)  # (B,W)

    return best_R[:, None], final_angle_errors

def _solve_l1_lad_admm(
    A: sparse.csc_array,
    b: np.ndarray,
    *,
    rho: float = 1.0,
    max_iter: int = 1000,
) -> np.ndarray:
    at = A.T
    factor = cholesky((at @ A).tocsc(), mode="simplicial")

    m, n = A.shape
    x = np.zeros(n, dtype=np.float64)
    z = np.zeros(m, dtype=np.float64)
    u = np.zeros(m, dtype=np.float64)

    for _ in range(max_iter):
        residual_target = b + z - u
        x = factor(at @ residual_target)
        ax_minus_b = A @ x - b
        z = np.sign(ax_minus_b + u) * np.maximum(np.abs(ax_minus_b + u) - 1.0 / rho, 0.0)
        u = u + ax_minus_b - z

    return x


def _setup_position_system(
    num_frames: int,
    num_windows: int,
    window_size: int,
    global_rel_t: torch.Tensor,
    fix_index: int,
) -> tuple[sparse.csc_array, np.ndarray]:
    pair_count = num_windows * (window_size - 1)
    row_count = 3 + 3 * pair_count
    col_count = 3 * num_frames

    rows = np.zeros(3 + 6 * pair_count, dtype=np.int32)
    cols = np.zeros(3 + 6 * pair_count, dtype=np.int32)
    data = np.zeros(3 + 6 * pair_count, dtype=np.float64)
    b = np.zeros(row_count, dtype=np.float64)

    rows[:3] = [0, 1, 2]
    cols[:3] = [3 * fix_index, 3 * fix_index + 1, 3 * fix_index + 2]
    data[:3] = 1.0

    cursor = 3
    for window_idx in range(num_windows):
        for offset in range(window_size - 1):
            row_start = 3 + 3 * (window_idx * (window_size - 1) + offset)
            src_col = 3 * (window_idx + offset)
            dst_col = 3 * (window_idx + window_size - 1)
            translation = global_rel_t[window_idx, offset].detach().cpu().numpy()
            for axis in range(3):
                rows[cursor] = row_start + axis
                cols[cursor] = src_col + axis
                data[cursor] = 1.0
                cursor += 1

                rows[cursor] = row_start + axis
                cols[cursor] = dst_col + axis
                data[cursor] = -1.0
                cursor += 1

                b[row_start + axis] = -translation[axis]

    A = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(row_count, col_count),
        dtype=np.float64,
    ).tocsc()
    return A, b


def _setup_position_edge_system(
    num_frames: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    global_rel_t: np.ndarray,
    edge_weights: np.ndarray,
    fix_index: int,
) -> tuple[sparse.csc_array, np.ndarray]:
    edge_count = int(len(edge_src))
    row_count = 3 + 3 * edge_count
    col_count = 3 * num_frames
    rows = np.zeros(3 + 6 * edge_count, dtype=np.int32)
    cols = np.zeros(3 + 6 * edge_count, dtype=np.int32)
    data = np.zeros(3 + 6 * edge_count, dtype=np.float64)
    b = np.zeros(row_count, dtype=np.float64)

    rows[:3] = [0, 1, 2]
    cols[:3] = [3 * fix_index, 3 * fix_index + 1, 3 * fix_index + 2]
    data[:3] = 1.0

    cursor = 3
    for edge_idx in range(edge_count):
        row_start = 3 + 3 * edge_idx
        src_col = 3 * int(edge_src[edge_idx])
        dst_col = 3 * int(edge_dst[edge_idx])
        weight = float(edge_weights[edge_idx])
        translation = np.asarray(global_rel_t[edge_idx], dtype=np.float64)
        for axis in range(3):
            rows[cursor] = row_start + axis
            cols[cursor] = src_col + axis
            data[cursor] = weight
            cursor += 1

            rows[cursor] = row_start + axis
            cols[cursor] = dst_col + axis
            data[cursor] = -weight
            cursor += 1

            b[row_start + axis] = -weight * translation[axis]

    A = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(row_count, col_count),
        dtype=np.float64,
    ).tocsc()
    return A, b


@torch.no_grad()
def position_averaging(rel_t: torch.Tensor, abs_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_windows, window_size = rel_t.shape[:2]
    num_frames = abs_r.shape[0]
    if num_windows != num_frames - window_size + 1:
        raise ValueError(
            f"Expected {num_frames - window_size + 1} windows, got {num_windows}."
        )

    fix_index = window_size - 1
    window_abs_r = abs_r.unfold(dimension=0, size=window_size, step=1)
    window_abs_r = window_abs_r.permute(0, 3, 1, 2)
    global_rel_t = (window_abs_r.mT @ rel_t)[:, : window_size - 1, :, 0]

    positions = _solve_l1_lad_admm(
        *_setup_position_system(
            num_frames,
            num_windows,
            window_size,
            global_rel_t,
            fix_index,
        )
    )
    positions = torch.from_numpy(positions).to(device=rel_t.device, dtype=torch.float32)
    positions = positions.reshape(num_frames, 3)
    scales = torch.ones(num_windows, device=rel_t.device, dtype=positions.dtype)
    return positions, scales


@torch.no_grad()
def position_averaging_edges(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    rel_t: torch.Tensor,
    abs_r: torch.Tensor,
    *,
    edge_weights: torch.Tensor | None = None,
    fix_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = abs_r.shape[0]
    if edge_src.shape != edge_dst.shape:
        raise ValueError("edge_src and edge_dst must have the same shape.")
    if rel_t.shape[:1] != edge_src.shape or rel_t.shape[-2:] != (3, 1):
        raise ValueError("rel_t must have shape [E,3,1].")
    if num_frames <= 0:
        raise ValueError("abs_r must contain at least one frame.")
    if fix_index < 0 or fix_index >= num_frames:
        raise IndexError("fix_index is outside the trajectory.")

    device = rel_t.device
    edge_src = edge_src.to(device=device, dtype=torch.long)
    edge_dst = edge_dst.to(device=device, dtype=torch.long)
    if edge_weights is None:
        edge_weights = torch.ones(edge_src.shape[0], device=device, dtype=torch.float32)
    else:
        edge_weights = edge_weights.to(device=device, dtype=torch.float32)
    if edge_src.numel() == 0:
        positions = torch.zeros(num_frames, 3, device=device, dtype=torch.float32)
        return positions, edge_weights

    global_rel_t = (abs_r[edge_src].mT @ rel_t.to(device=device, dtype=torch.float32))[..., 0]
    positions = _solve_l1_lad_admm(
        *_setup_position_edge_system(
            num_frames,
            edge_src.detach().cpu().numpy(),
            edge_dst.detach().cpu().numpy(),
            global_rel_t.detach().cpu().numpy(),
            edge_weights.detach().cpu().numpy(),
            fix_index,
        )
    )
    positions = torch.from_numpy(positions).to(device=device, dtype=torch.float32)
    return positions.reshape(num_frames, 3), edge_weights

def _rotation_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    device = matrix.device
    dtype = matrix.dtype
    orig_shape = matrix.shape[:-2]
    matrix_np = matrix.reshape(-1, 3, 3).detach().cpu().numpy()
    axis_np = R.from_matrix(matrix_np).as_rotvec()
    return torch.from_numpy(axis_np).to(device=device, dtype=dtype).reshape(*orig_shape, 3)


def _axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    device = axis_angle.device
    dtype = axis_angle.dtype
    orig_shape = axis_angle.shape[:-1]
    axis_np = axis_angle.reshape(-1, 3).detach().cpu().numpy()
    matrix_np = R.from_rotvec(axis_np).as_matrix()
    return torch.from_numpy(matrix_np).to(device=device, dtype=dtype).reshape(*orig_shape, 3, 3)


def _setup_rotation_system(num_windows: int, window_size: int, num_frames: int, fix_index: int) -> sparse.csc_array:
    if num_windows != num_frames - window_size + 1:
        raise ValueError("num_windows must equal num_frames - window_size + 1.")

    num_edges = num_windows * (window_size - 1)
    rows_num = 3 + 3 * num_edges
    cols_num = 3 * num_frames
    nnz = 3 + 6 * num_edges

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)

    xyz = np.array([0, 1, 2], dtype=np.int32)
    rows[:3] = xyz
    cols[:3] = 3 * fix_index + xyz
    data[:3] = 1.0

    window_ids = np.arange(num_windows, dtype=np.int32)
    local_ids = np.arange(window_size - 1, dtype=np.int32)
    src_frames = (window_ids[:, None] + local_ids[None, :]).reshape(-1)
    dst_frames = np.repeat(window_ids + window_size - 1, window_size - 1)

    edge_rows = np.arange(3, 3 + 3 * num_edges, dtype=np.int32)

    head_start = 3
    head_end = head_start + 3 * num_edges
    rows[head_start:head_end] = edge_rows
    cols[head_start:head_end] = (3 * src_frames[:, None] + xyz[None, :]).reshape(-1)
    data[head_start:head_end] = 1.0

    tail_start = head_end
    tail_end = tail_start + 3 * num_edges
    rows[tail_start:tail_end] = edge_rows
    cols[tail_start:tail_end] = (3 * dst_frames[:, None] + xyz[None, :]).reshape(-1)
    data[tail_start:tail_end] = -1.0

    return sparse.coo_matrix((data, (rows, cols)), shape=(rows_num, cols_num), dtype=np.float64).tocsc()


def _setup_rotation_edge_system(
    num_frames: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    fix_index: int,
) -> sparse.csc_array:
    num_edges = int(len(edge_src))
    rows_num = 3 + 3 * num_edges
    cols_num = 3 * num_frames
    nnz = 3 + 6 * num_edges

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)

    xyz = np.array([0, 1, 2], dtype=np.int32)
    rows[:3] = xyz
    cols[:3] = 3 * fix_index + xyz
    data[:3] = 1.0

    edge_rows = np.arange(3, 3 + 3 * num_edges, dtype=np.int32)
    head_start = 3
    head_end = head_start + 3 * num_edges
    rows[head_start:head_end] = edge_rows
    cols[head_start:head_end] = (3 * edge_src[:, None] + xyz[None, :]).reshape(-1)
    data[head_start:head_end] = 1.0

    tail_start = head_end
    tail_end = tail_start + 3 * num_edges
    rows[tail_start:tail_end] = edge_rows
    cols[tail_start:tail_end] = (3 * edge_dst[:, None] + xyz[None, :]).reshape(-1)
    data[tail_start:tail_end] = -1.0

    return sparse.coo_matrix((data, (rows, cols)), shape=(rows_num, cols_num), dtype=np.float64).tocsc()


def _compute_residual_rotations(
    abs_rotations: torch.Tensor,
    rel_rotations: torch.Tensor,
    window_size: int,
    fix_index: int,
) -> torch.Tensor:
    window_abs = abs_rotations.unfold(dimension=0, size=window_size, step=1).permute(0, 3, 1, 2)
    src_abs = window_abs[:, : window_size - 1].mT
    dst_abs = window_abs[:, -1:]
    residual = src_abs @ rel_rotations @ dst_abs
    residual_axis = _rotation_matrix_to_axis_angle(residual)
    fixed_axis = _rotation_matrix_to_axis_angle(abs_rotations[fix_index].mT)
    return torch.cat([fixed_axis, residual_axis.reshape(-1)], dim=0)


def _compute_residual_rotation_edges(
    abs_rotations: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    rel_rotations: torch.Tensor,
    fix_index: int,
) -> torch.Tensor:
    src_abs = abs_rotations[edge_src].mT
    dst_abs = abs_rotations[edge_dst]
    residual = src_abs @ rel_rotations @ dst_abs
    residual_axis = _rotation_matrix_to_axis_angle(residual)
    fixed_axis = _rotation_matrix_to_axis_angle(abs_rotations[fix_index].mT)
    return torch.cat([fixed_axis, residual_axis.reshape(-1)], dim=0)


def _solve_delta_axis_angles(
    system: sparse.csc_array,
    system_t: sparse.csc_array,
    residual_axis: torch.Tensor,
    factor,
    residual_weights: np.ndarray | None = None,
) -> torch.Tensor:
    device = residual_axis.device
    residual_np = residual_axis.detach().cpu().numpy()

    sigma_sq = math.radians(5.0) ** 2
    err_sq = (residual_np.reshape(-1, 3) ** 2).sum(axis=-1, keepdims=True)
    weights = sigma_sq / (sigma_sq + err_sq) ** 2
    if residual_weights is not None:
        residual_weights = np.asarray(residual_weights, dtype=np.float64).reshape(-1, 1)
        if residual_weights.shape[0] != weights.shape[0]:
            raise ValueError("residual_weights length does not match rotation residual groups.")
        weights = weights * residual_weights
    weights = np.repeat(weights, 3, axis=1).ravel()

    weighted_system_t = system_t.multiply(weights)
    hessian = (weighted_system_t @ system).tocsc()
    factor.cholesky_inplace(hessian)
    delta_np = factor(weighted_system_t @ residual_np).reshape(-1, 3)
    return torch.from_numpy(delta_np).to(device=device, dtype=torch.float32)


@torch.no_grad()
def rotation_averaging(
    valid_relative_rotations: torch.Tensor,
    online_absolute_rotations: torch.Tensor,
    max_iters: int = 100,
) -> torch.Tensor:
    num_windows, window_size = valid_relative_rotations.shape[:2]
    num_frames = online_absolute_rotations.shape[0]
    if num_windows != num_frames - window_size + 1:
        raise ValueError("valid_relative_rotations shape does not match online_absolute_rotations.")

    fix_index = window_size - 1
    system = _setup_rotation_system(num_windows, window_size, num_frames, fix_index)
    system_t = system.T.tocsc()
    factor = analyze((system_t @ system).tocsc(), ordering_method="amd")

    rel_rotations = valid_relative_rotations[:, : window_size - 1]
    abs_rotations = online_absolute_rotations @ online_absolute_rotations[fix_index].mT

    for _ in range(max_iters):
        residual_axis = _compute_residual_rotations(abs_rotations, rel_rotations, window_size, fix_index)
        delta_axis = _solve_delta_axis_angles(system, system_t, residual_axis, factor)
        delta_rotations = _axis_angle_to_rotation_matrix(delta_axis)
        abs_rotations = abs_rotations @ delta_rotations

        if delta_axis.norm(dim=-1).mean() < 1e-6:
            break

    return abs_rotations


@torch.no_grad()
def rotation_averaging_edges(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    rel_rotations: torch.Tensor,
    online_absolute_rotations: torch.Tensor,
    *,
    edge_weights: torch.Tensor | None = None,
    fix_index: int = 0,
    max_iters: int = 100,
) -> torch.Tensor:
    num_frames = online_absolute_rotations.shape[0]
    if edge_src.shape != edge_dst.shape:
        raise ValueError("edge_src and edge_dst must have the same shape.")
    if rel_rotations.shape[:1] != edge_src.shape or rel_rotations.shape[-2:] != (3, 3):
        raise ValueError("rel_rotations must have shape [E,3,3].")
    if num_frames <= 0:
        raise ValueError("online_absolute_rotations must contain at least one frame.")
    if fix_index < 0 or fix_index >= num_frames:
        raise IndexError("fix_index is outside the trajectory.")

    device = online_absolute_rotations.device
    edge_src = edge_src.to(device=device, dtype=torch.long)
    edge_dst = edge_dst.to(device=device, dtype=torch.long)
    rel_rotations = rel_rotations.to(device=device, dtype=torch.float32)
    if edge_weights is None:
        edge_weights = torch.ones(edge_src.shape[0], device=device, dtype=torch.float32)
    else:
        edge_weights = edge_weights.to(device=device, dtype=torch.float32)
    if edge_src.numel() == 0:
        return online_absolute_rotations @ online_absolute_rotations[fix_index].mT

    if int(edge_src.min()) < 0 or int(edge_dst.min()) < 0 or int(edge_src.max()) >= num_frames or int(edge_dst.max()) >= num_frames:
        raise IndexError("rotation edge index is outside the trajectory.")
    edge_src_np = edge_src.detach().cpu().numpy().astype(np.int32, copy=False)
    edge_dst_np = edge_dst.detach().cpu().numpy().astype(np.int32, copy=False)
    system = _setup_rotation_edge_system(num_frames, edge_src_np, edge_dst_np, fix_index)
    system_t = system.T.tocsc()
    factor = analyze((system_t @ system).tocsc(), ordering_method="amd")
    residual_weights = np.concatenate(
        [
            np.ones(1, dtype=np.float64),
            edge_weights.detach().cpu().numpy().astype(np.float64, copy=False),
        ]
    )

    abs_rotations = online_absolute_rotations.to(device=device, dtype=torch.float32)
    abs_rotations = abs_rotations @ abs_rotations[fix_index].mT

    for iter_idx in range(max_iters):
        residual_axis = _compute_residual_rotation_edges(
            abs_rotations,
            edge_src,
            edge_dst,
            rel_rotations,
            fix_index,
        )
        delta_axis = _solve_delta_axis_angles(
            system,
            system_t,
            residual_axis,
            factor,
            residual_weights=residual_weights,
        )
        delta_rotations = _axis_angle_to_rotation_matrix(delta_axis)
        abs_rotations = abs_rotations @ delta_rotations

        mean_update = delta_axis.norm(dim=-1).mean()
        if mean_update < 1e-8:
            break

    return abs_rotations
