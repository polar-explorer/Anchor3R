from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def evaluate_saved_predictions(
    *,
    pose_pred: str,
    images: str,
    gt_poses: str,
    gt_format: str = "auto",
    alignment: str = "sim3",
    output_dir: str | None = None,
    sequence_name: str | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if alignment not in {"sim3", "se3"}:
        raise ValueError(f"Unknown alignment: {alignment}")

    image_paths = _load_image_paths(images, max_frames)
    pose_names, pose_w2c = _load_matrix_txt(pose_pred)
    if not image_paths:
        image_paths = [f"{name}.png" for name in pose_names]

    gt_by_name, gt_order = _load_gt_c2w(gt_poses, gt_format, image_paths)
    gt, pose, frame_idx = _match_poses(
        gt_by_name,
        gt_order,
        image_paths,
        _w2c_to_c2w(pose_w2c),
    )
    gt = _orthogonalize(gt)
    pose = _orthogonalize(pose)
    pose_aligned, transform = _align(gt, pose, alignment)
    metrics = _metrics(gt, pose_aligned, frame_idx)

    prediction_root = Path(pose_pred).resolve().parents[1]
    prediction_dir = Path(pose_pred).resolve().parent.name
    default_eval_dir = "pose_eval_offline" if prediction_dir == "poses_offline" else "pose_eval"
    eval_dir = Path(output_dir or prediction_root / default_eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_path = eval_dir / f"trajectory_{alignment}.png"
    metrics_path = eval_dir / "metrics.json"
    _plot_trajectories(
        gt,
        pose_aligned,
        sequence_name or _default_sequence_name(images),
        alignment,
        transform,
        metrics,
        frame_idx,
        plot_path,
    )
    report = {
        "sequence": sequence_name or _default_sequence_name(images),
        "alignment": alignment,
        "valid_frames": int(len(gt)),
        "pose": _scalar_metrics(metrics, transform),
    }
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"[PoseEval] {sequence_name or _default_sequence_name(images)}: "
        f"pose ATE={metrics['ate_rmse']:.3f}m RTE={metrics['rte_rmse']:.3f}m "
        f"RRE={metrics['rre_rmse']:.3f}deg scale={transform['scale']:.3f}; "
        f"plot={plot_path}",
        flush=True,
    )
    return report


def _scalar_metrics(metrics: dict[str, Any], transform: dict[str, Any]) -> dict[str, float]:
    return {
        "ate_rmse_m": float(metrics["ate_rmse"]),
        "rte_rmse_m": float(metrics["rte_rmse"]),
        "rre_rmse_deg": float(metrics["rre_rmse"]),
        "sim3_scale": float(transform["scale"]),
    }


def _load_gt_c2w(gt_poses: str, gt_format: str, image_paths: Sequence[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    path = _resolve_gt_path(gt_poses, gt_format, image_paths)
    fmt = _infer_gt_format(path, gt_format)
    if fmt == "vbr-extri":
        return _load_vbr_extri_c2w(path)
    if fmt in {"w2c-txt", "c2w-txt"}:
        names, poses = _load_matrix_txt(path)
    elif fmt in {"w2c-npy", "c2w-npy"}:
        poses = _as_pose_stack(np.load(path))
        names = [f"{idx:06d}" for idx in range(len(poses))]
    else:
        raise ValueError(f"Unsupported gt_format: {gt_format}")
    if fmt.startswith("w2c"):
        poses = _w2c_to_c2w(poses)
    return dict(zip(names, poses)), names


def _resolve_gt_path(gt_poses: str, gt_format: str, image_paths: Sequence[str]) -> str:
    path = Path(os.path.expanduser(gt_poses))
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    camera_id = Path(image_paths[0]).parent.name if image_paths else ""
    for candidate in (path / "extri.yml", path / "cameras" / camera_id / "extri.yml", path / camera_id / "extri.yml"):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not resolve ground-truth pose file from {path}")


def _infer_gt_format(path: str, gt_format: str) -> str:
    if gt_format != "auto":
        return gt_format
    suffix = Path(path).suffix.lower()
    if suffix in {".yml", ".yaml"}:
        return "vbr-extri"
    if suffix in {".npy", ".npz"}:
        return "w2c-npy"
    return "w2c-txt"


def _load_vbr_extri_c2w(path: str) -> tuple[dict[str, np.ndarray], list[str]]:
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    try:
        node = storage.getNode("names")
        names = [node.at(idx).string() or str(int(node.at(idx).real())) for idx in range(node.size())]
        poses = {}
        for name in names:
            rotation = storage.getNode(f"Rot_{name}").mat()
            if rotation is None:
                rotation_vector = storage.getNode(f"R_{name}").mat()
                if rotation_vector is None:
                    raise KeyError(f"Missing rotation for {name}")
                rotation = cv2.Rodrigues(rotation_vector.reshape(3, 1))[0]
            translation = storage.getNode(f"T_{name}").mat()
            if translation is None:
                raise KeyError(f"Missing translation for {name}")
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = np.asarray(rotation).reshape(3, 3)
            w2c[:3, 3] = np.asarray(translation).reshape(3)
            poses[name] = np.linalg.inv(w2c)
    finally:
        storage.release()
    return poses, names


def _load_matrix_txt(path: str) -> tuple[list[str], np.ndarray]:
    names, poses = [], []
    with open(path, "r") as file:
        for line_idx, line in enumerate(file):
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) == 13:
                name, values = Path(parts[0]).stem, parts[1:]
            elif len(parts) == 12:
                name, values = f"{line_idx:06d}", parts
            elif len(parts) == 17:
                name, values = Path(parts[0]).stem, parts[1:]
            elif len(parts) == 16:
                name, values = f"{line_idx:06d}", parts
            else:
                raise ValueError(f"Unsupported pose row with {len(parts)} columns in {path}")
            matrix = np.asarray([float(value) for value in values], dtype=np.float64)
            pose = np.eye(4, dtype=np.float64)
            if matrix.size == 12:
                pose[:3, :3] = matrix[:9].reshape(3, 3)
                pose[:3, 3] = matrix[9:]
            else:
                pose = matrix.reshape(4, 4)
            names.append(name)
            poses.append(pose)
    if not poses:
        raise RuntimeError(f"No poses found in {path}")
    return names, np.stack(poses)


def _match_poses(
    gt_by_name: dict[str, np.ndarray],
    gt_order: list[str],
    image_paths: Sequence[str],
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = min(len(image_paths), len(pose))
    gt_poses, pose_poses, frame_indices = [], [], []
    for idx, image_path in enumerate(image_paths[:length]):
        keys = (Path(image_path).stem, Path(image_path).name, str(idx), f"{idx:06d}")
        key = next((candidate for candidate in keys if candidate in gt_by_name), None)
        if key is None and idx < len(gt_order):
            key = gt_order[idx]
        if key not in gt_by_name:
            continue
        values = (gt_by_name[key], pose[idx])
        if not all(np.isfinite(value).all() for value in values):
            continue
        gt_poses.append(values[0])
        pose_poses.append(values[1])
        frame_indices.append(idx)
    if len(gt_poses) < 2:
        raise RuntimeError(f"Need at least 2 valid pose pairs, got {len(gt_poses)}")
    return np.stack(gt_poses), np.stack(pose_poses), np.asarray(frame_indices)


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_as_pose_stack(w2c))


def _as_pose_stack(value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim == 2 and poses.shape[-1] in {12, 16}:
        poses = poses.reshape(len(poses), 3 if poses.shape[-1] == 12 else 4, 4)
    if poses.shape[-2:] == (3, 4):
        out = np.tile(np.eye(4), (len(poses), 1, 1))
        out[:, :3] = poses
        poses = out
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Invalid pose shape: {poses.shape}")
    return poses


def _orthogonalize(poses: np.ndarray) -> np.ndarray:
    poses = poses.copy()
    for idx in range(len(poses)):
        u, _, vt = np.linalg.svd(poses[idx, :3, :3])
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        poses[idx, :3, :3] = rotation
    return poses


def _align(gt: np.ndarray, pred: np.ndarray, alignment: str) -> tuple[np.ndarray, dict[str, Any]]:
    src, dst = pred[:, :3, 3], gt[:, :3, 3]
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    variance = np.mean(np.sum(src_centered**2, axis=1))
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    scale = 1.0 if alignment == "se3" else float(np.trace(np.diag(singular_values) @ sign) / max(variance, 1e-12))
    translation = dst_mean - scale * rotation @ src_mean
    aligned = pred.copy()
    aligned[:, :3, :3] = rotation[None] @ pred[:, :3, :3]
    aligned[:, :3, 3] = (scale * (rotation @ src.T)).T + translation
    return aligned, {"scale": scale, "R": rotation, "t": translation}


def _metrics(gt: np.ndarray, pred: np.ndarray, frame_idx: np.ndarray) -> dict[str, Any]:
    ate = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    rte = np.full(len(gt), np.nan, dtype=np.float64)
    rre = np.full(len(gt), np.nan, dtype=np.float64)
    for idx in range(1, len(gt)):
        if frame_idx[idx] != frame_idx[idx - 1] + 1:
            continue
        relative_error = np.linalg.inv(np.linalg.inv(gt[idx - 1]) @ gt[idx]) @ (np.linalg.inv(pred[idx - 1]) @ pred[idx])
        rte[idx] = np.linalg.norm(relative_error[:3, 3])
        cosine = np.clip((np.trace(relative_error[:3, :3]) - 1) * 0.5, -1.0, 1.0)
        rre[idx] = np.degrees(np.arccos(cosine))
    return {
        "ate": ate,
        "rte": rte,
        "rre": rre,
        "ate_rmse": float(np.sqrt(np.mean(ate**2))),
        "rte_rmse": _nan_rmse(rte),
        "rre_rmse": _nan_rmse(rre),
    }


def _nan_rmse(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(valid**2))) if valid.size else float("nan")


def _plot_trajectories(
    gt: np.ndarray,
    pose: np.ndarray,
    sequence_name: str,
    alignment: str,
    transform: dict[str, Any],
    metrics: dict[str, float],
    frame_idx: np.ndarray,
    output_path: Path,
) -> None:
    gt_xyz, pose_xyz = gt[:, :3, 3], pose[:, :3, 3]
    points = np.concatenate([gt_xyz, pose_xyz], axis=0)
    mins, maxs = points.min(0), points.max(0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 1e-6)
    limits = [(center[idx] - radius, center[idx] + radius) for idx in range(3)]

    pose_label = (
        f"Pose {alignment.upper()} | ATE {metrics['ate_rmse']:.3f}m | "
        f"RTE {metrics['rte_rmse']:.3f}m | RRE {metrics['rre_rmse']:.3f}deg | "
        f"{alignment.upper()} scale={transform['scale']:.3f}"
    )
    figure = plt.figure(figsize=(20, 12))
    ax3d = figure.add_subplot(2, 3, 1, projection="3d")
    ax_xy = figure.add_subplot(2, 3, 2)
    ax_xz = figure.add_subplot(2, 3, 3)
    ax_ate = figure.add_subplot(2, 3, 4)
    ax_rte = figure.add_subplot(2, 3, 5)
    ax_rre = figure.add_subplot(2, 3, 6)

    ax3d.plot(*gt_xyz.T, color="red", linewidth=2.3, label="GT")
    ax3d.plot(*pose_xyz.T, color="#1f77b4", linewidth=2.0, label=pose_label)
    ax3d.set_xlim(*limits[0]); ax3d.set_ylim(*limits[1]); ax3d.set_zlim(*limits[2])
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"{sequence_name} - 3D Trajectory")
    ax3d.legend(loc="best", fontsize=7)

    _plot_projection(ax_xy, gt_xyz[:, [0, 1]], pose_xyz[:, [0, 1]], limits[0], limits[1], "XY Projection", "Y (m)", alignment)
    _plot_projection(ax_xz, gt_xyz[:, [0, 2]], pose_xyz[:, [0, 2]], limits[0], limits[2], "XZ Projection", "Z (m)", alignment)
    _plot_error_curve(ax_ate, frame_idx, metrics["ate"], "ATE Curve", "ATE (m)")
    _plot_error_curve(ax_rte, frame_idx, metrics["rte"], "RTE Curve (true per-frame)", "RTE (m)")
    _plot_error_curve(ax_rre, frame_idx, metrics["rre"], "RRE Curve (true per-frame)", "RRE (deg)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_projection(ax, gt, pose, xlim, ylim, title, ylabel, alignment) -> None:
    ax.plot(gt[:, 0], gt[:, 1], color="red", linewidth=2.3, label="GT")
    ax.plot(pose[:, 0], pose[:, 1], color="#1f77b4", linewidth=2.0, label=f"Pose {alignment.upper()}")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("X (m)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.4); ax.legend(loc="best")


def _plot_error_curve(ax, frame_idx, values, title, ylabel) -> None:
    stats = _stats(values)
    ax.plot(
            frame_idx,
            values,
            color="#1f77b4",
            linewidth=1.8,
            label=(
                f"Pose: rmse={stats['rmse']:.3f}, mean={stats['mean']:.3f}, "
                f"med={stats['median']:.3f}, p90={stats['p90']:.3f}"
            ),
    )
    finite = np.isfinite(values)
    ax.fill_between(frame_idx, 0.0, np.where(finite, values, 0.0), where=finite, color="#1f77b4", alpha=0.10)
    _annotate_peaks(ax, frame_idx, values, "#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")


def _stats(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if not valid.size:
        return {key: float("nan") for key in ("rmse", "mean", "median", "p90")}
    return {
        "rmse": float(np.sqrt(np.mean(valid**2))),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p90": float(np.percentile(valid, 90)),
    }


def _annotate_peaks(ax, frame_idx: np.ndarray, values: np.ndarray, color: str, topk: int = 8) -> None:
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not valid_indices.size:
        return
    peak_indices = valid_indices[np.argsort(values[valid_indices])[-min(topk, valid_indices.size) :]]
    for idx in peak_indices:
        ax.plot(frame_idx[idx], values[idx], "o", color=color, markersize=3)
        ax.text(frame_idx[idx], values[idx], f" {int(frame_idx[idx])}", fontsize=6, color=color)


def _load_image_paths(images: str, max_frames: int | None) -> list[str]:
    path = Path(os.path.expanduser(images))
    if path.is_file():
        paths = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        paths = sorted(str(item) for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTS)
    return paths if max_frames is None else paths[:max_frames]


def _default_sequence_name(images: str) -> str:
    path = Path(os.path.expanduser(images))
    if path.is_file():
        return path.stem
    if path.parent.name in {"images", "rgb"}:
        return path.parent.parent.name
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser("Anchor3R pose evaluation")
    parser.add_argument("--pose-pred", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--gt-poses", required=True)
    parser.add_argument("--gt-format", default="auto")
    parser.add_argument("--alignment", default="sim3", choices=("sim3", "se3"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sequence-name", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    evaluate_saved_predictions(
        pose_pred=args.pose_pred,
        images=args.images,
        gt_poses=args.gt_poses,
        gt_format=args.gt_format,
        alignment=args.alignment,
        output_dir=args.output_dir,
        sequence_name=args.sequence_name,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
