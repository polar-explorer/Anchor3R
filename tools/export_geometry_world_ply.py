from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    manifest = json.loads((scene_dir / "geometry" / "manifest.json").read_text())
    frames = manifest["frames"]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    image_dir = Path(manifest["image_dir"]) if not args.no_rgb else None
    sky_mask_dir = resolve_sky_mask_dir(args.sky_mask_dir, manifest) if args.mask_sky else None
    pose_dir = (scene_dir / "geometry" / manifest.get("pose_dir", "../poses")).resolve()
    poses = load_pose_txt(pose_dir / args.pose, args.pose_convention)

    chunks: list[np.ndarray] = []
    total = 0
    for ordinal, frame in enumerate(frames[:: args.frame_stride]):
        pose = poses.get(frame)
        if pose is None:
            pose = poses.get(str(ordinal * args.frame_stride))
        if pose is None:
            continue
        local_path = scene_dir / "geometry" / "local_points" / f"{frame}.npy"
        conf_path = scene_dir / "geometry" / "confidence" / f"{frame}.npy"
        if not local_path.is_file():
            continue
        local = np.load(local_path, mmap_mode="r")
        conf = np.load(conf_path, mmap_mode="r") if conf_path.is_file() else None
        rgb = load_rgb(image_dir, frame) if image_dir is not None else None
        sky_mask = load_sky_mask(sky_mask_dir, frame) if sky_mask_dir is not None else None
        points = select_points(
            local,
            rgb,
            conf,
            sky_mask,
            pose,
            pixel_stride=args.pixel_stride,
            min_conf=args.min_conf,
            max_depth=args.max_depth,
        )
        if len(points):
            chunks.append(points)
            total += len(points)
        if args.max_points > 0 and total >= args.max_points * 1.2:
            break

    if not chunks:
        raise RuntimeError("No points selected; lower --min-conf or check inputs.")
    points = np.concatenate(chunks)
    if args.max_points > 0 and len(points) > args.max_points:
        keep = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points = points[keep]
    write_binary_ply(Path(args.output), points)
    print(f"Wrote {args.output}: {len(points):,} points")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export geometry/local_points to a world-space RGB PLY")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pose", default="online_pose.txt")
    parser.add_argument("--pose-convention", choices=("w2c", "c2w"), default="w2c")
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--mask-sky", action="store_true")
    parser.add_argument("--sky-mask-dir", default="")
    parser.add_argument("--min-conf", type=float, default=5.0)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--max-points", type=int, default=5_000_000)
    return parser.parse_args()


def resolve_sky_mask_dir(path: str, manifest: dict) -> Path:
    if path:
        return Path(path)
    image_dir = Path(manifest["image_dir"])
    return image_dir.parent.parent / "sky_masks" / image_dir.name


def load_pose_txt(path: Path, convention: str = "w2c") -> dict[str, np.ndarray]:
    poses = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name = Path(parts[0]).stem
            vals = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            if len(vals) == 12:
                w2c[:3, :3] = vals[:9].reshape(3, 3)
                w2c[:3, 3] = vals[9:]
            elif len(vals) == 16:
                w2c = vals.reshape(4, 4)
            else:
                raise ValueError(f"Unsupported pose row in {path}: {line}")
            poses[name] = np.linalg.inv(w2c) if convention == "w2c" else w2c
    return poses


def load_rgb(image_dir: Path, frame: str) -> np.ndarray:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = image_dir / f"{frame}{suffix}"
        if path.is_file():
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
    raise FileNotFoundError(f"Missing RGB image for frame {frame} under {image_dir}")


def load_sky_mask(mask_dir: Path, frame: str) -> np.ndarray:
    path = mask_dir / f"{frame}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sky mask for frame {frame} under {mask_dir}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def select_points(
    local: np.ndarray,
    rgb: np.ndarray | None,
    conf: np.ndarray | None,
    sky_mask: np.ndarray | None,
    c2w: np.ndarray,
    *,
    pixel_stride: int,
    min_conf: float,
    max_depth: float,
) -> np.ndarray:
    pts = np.asarray(local[::pixel_stride, ::pixel_stride], dtype=np.float32)
    if rgb is None:
        colors = np.full((*pts.shape[:2], 3), 180, dtype=np.uint8)
    else:
        colors = resize_nearest(rgb, pts.shape[:2]).astype(np.uint8)
    confidence = None
    if conf is not None:
        confidence = np.asarray(conf[::pixel_stride, ::pixel_stride], dtype=np.float32)
    keep_mask = None
    if sky_mask is not None:
        keep_mask = resize_nearest(sky_mask, pts.shape[:2]) > 0

    flat_pts = pts.reshape(-1, 3)
    flat_rgb = colors.reshape(-1, 3)
    valid = np.isfinite(flat_pts).all(axis=1)
    valid &= flat_pts[:, 2] > 0
    if max_depth > 0:
        valid &= flat_pts[:, 2] <= max_depth
    if confidence is not None:
        flat_conf = confidence.reshape(-1)
        valid &= np.isfinite(flat_conf)
        valid &= flat_conf >= min_conf
    else:
        flat_conf = np.ones(len(flat_pts), dtype=np.float32)
    if keep_mask is not None:
        valid &= keep_mask.reshape(-1)

    flat_pts = flat_pts[valid]
    if len(flat_pts) == 0:
        return np.empty(0, dtype=ply_dtype())
    world = flat_pts @ c2w[:3, :3].T + c2w[:3, 3]
    flat_rgb = flat_rgb[valid]
    flat_conf = flat_conf[valid]
    out = np.empty(len(world), dtype=ply_dtype())
    out["x"] = world[:, 0].astype(np.float32)
    out["y"] = world[:, 1].astype(np.float32)
    out["z"] = world[:, 2].astype(np.float32)
    out["red"] = flat_rgb[:, 0]
    out["green"] = flat_rgb[:, 1]
    out["blue"] = flat_rgb[:, 2]
    out["confidence"] = flat_conf.astype(np.float32)
    return out


def resize_nearest(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    src_h, src_w = image.shape[:2]
    ys = np.linspace(0, src_h - 1, h).round().astype(np.int64)
    xs = np.linspace(0, src_w - 1, w).round().astype(np.int64)
    return image[ys[:, None], xs[None, :]]


def ply_dtype() -> np.dtype:
    return np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<f4"),
        ]
    )


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float confidence\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        points.tofile(handle)


if __name__ == "__main__":
    main()
