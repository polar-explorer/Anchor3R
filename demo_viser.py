from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_pose_file(path: Path) -> list[np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    has_frame_id = rows.shape[1] == 13
    if has_frame_id:
        rows = rows[:, 1:]
    if rows.shape[1] != 12:
        raise ValueError(f"Expected 12 pose values, optionally preceded by frame id: {path}")
    poses = []
    for row in rows:
        w2c = np.eye(4, dtype=np.float64)
        if has_frame_id:
            w2c[:3, :3] = row[:9].reshape(3, 3)
            w2c[:3, 3] = row[9:]
        else:
            w2c[:3, :4] = row.reshape(3, 4)
        poses.append(np.linalg.inv(w2c))
    return poses


def load_intrinsics(path: Path) -> list[np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.shape[1] != 5:
        raise ValueError(f"Expected frame_id fx fy cx cy rows: {path}")
    matrices = []
    for _, fx, fy, cx, cy in rows:
        matrices.append(np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))
    return matrices


def find_image(image_dir: Path, frame: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{frame}{suffix}"
        if path.is_file():
            return path
    return None


def resize_nearest(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    ys = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    xs = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[ys[:, None], xs[None, :]]


def load_scene(
    sequence_dir: Path,
    source: str,
    frame_stride: int,
    pixel_stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
    geometry_dir = sequence_dir / "geometry"
    manifest = json.loads((geometry_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = list(manifest["frames"])[::frame_stride]
    image_dir = Path(manifest["image_dir"])
    pose_dir = (geometry_dir / manifest.get("pose_dir", "../poses")).resolve()
    poses = load_pose_file(pose_dir / f"{source}_pose.txt")
    intrinsics = load_intrinsics(pose_dir / "intri.txt")

    world_dir_name = manifest.get("world_points_dirs", {}).get(source)
    world_dir = geometry_dir / world_dir_name if world_dir_name else None
    local_dir = geometry_dir / manifest.get("local_points_dir", "local_points")
    confidence_name = manifest.get("confidence_dir")
    confidence_dir = geometry_dir / confidence_name if confidence_name else None

    point_chunks = []
    color_chunks = []
    confidence_chunks = []
    selected_poses = []
    selected_intrinsics = []
    selected_images = []

    for frame_index in range(0, min(len(manifest["frames"]), len(poses)), frame_stride):
        frame = manifest["frames"][frame_index]
        local = np.load(local_dir / f"{frame}.npy", mmap_mode="r")
        if world_dir is not None and (world_dir / f"{frame}.npy").is_file():
            world = np.load(world_dir / f"{frame}.npy", mmap_mode="r")
        else:
            c2w = poses[frame_index]
            world = local @ c2w[:3, :3].T + c2w[:3, 3]

        image_path = find_image(image_dir, frame)
        image = None if image_path is None else np.asarray(Image.open(image_path).convert("RGB"))
        colors = (
            np.full((*local.shape[:2], 3), 180, dtype=np.uint8)
            if image is None
            else resize_nearest(image, local.shape[:2]).astype(np.uint8)
        )
        confidence = (
            np.ones(local.shape[:2], dtype=np.float32)
            if confidence_dir is None or not (confidence_dir / f"{frame}.npy").is_file()
            else np.asarray(np.load(confidence_dir / f"{frame}.npy", mmap_mode="r"), dtype=np.float32)
        )
        if confidence.ndim == 3 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]

        sampled_local = np.asarray(local[::pixel_stride, ::pixel_stride], dtype=np.float32)
        sampled_world = np.asarray(world[::pixel_stride, ::pixel_stride], dtype=np.float32)
        sampled_colors = colors[::pixel_stride, ::pixel_stride]
        sampled_confidence = confidence[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(sampled_world).all(axis=-1)
        valid &= np.isfinite(sampled_confidence)
        valid &= sampled_local[..., 2] > 0
        point_chunks.append(sampled_world[valid])
        color_chunks.append(sampled_colors[valid])
        confidence_chunks.append(sampled_confidence[valid])
        selected_poses.append(poses[frame_index])
        selected_intrinsics.append(intrinsics[min(frame_index, len(intrinsics) - 1)])
        selected_images.append(image)

    if not point_chunks:
        raise RuntimeError(f"No geometry frames found in {geometry_dir}")
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    confidence = np.concatenate(confidence_chunks, axis=0)
    if max_points > 0 and len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors, confidence = points[keep], colors[keep], confidence[keep]
    return points, colors, confidence, selected_poses, selected_intrinsics, selected_images


def run_viewer(args: argparse.Namespace) -> None:
    import viser
    from viser import transforms as viser_transforms

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    points, colors, confidence, poses, intrinsics, images = load_scene(
        sequence_dir,
        source=args.source,
        frame_stride=max(1, args.frame_stride),
        pixel_stride=max(1, args.pixel_stride),
        max_points=max(0, args.max_points),
    )
    print(
        f"Loaded {len(points):,} points and {len(poses)} cameras from {sequence_dir}",
        flush=True,
    )
    if args.dry_run:
        return

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction(args.up_direction)
    threshold_min = float(np.nanmin(confidence)) if len(confidence) else 0.0
    threshold_max = float(np.nanmax(confidence)) if len(confidence) else 1.0
    initial_threshold = float(np.clip(args.min_confidence, threshold_min, threshold_max))

    with server.gui.add_folder("Point Cloud", expand_by_default=True):
        point_size = server.gui.add_slider("Point size", 0.0001, 0.1, 0.0001, args.point_size)
        min_confidence = server.gui.add_slider(
            "Minimum confidence",
            threshold_min,
            max(threshold_max, threshold_min + 1e-6),
            max((threshold_max - threshold_min) / 500.0, 1e-6),
            initial_threshold,
        )
        show_points = server.gui.add_checkbox("Show points", True)
    with server.gui.add_folder("Cameras", expand_by_default=True):
        show_cameras = server.gui.add_checkbox("Show cameras", True)
        camera_scale = server.gui.add_slider("Camera scale", 0.01, 2.0, 0.01, args.camera_scale)

    initial_mask = confidence >= initial_threshold
    cloud = server.scene.add_point_cloud(
        "/reconstruction",
        points=points[initial_mask],
        colors=colors[initial_mask],
        point_size=args.point_size,
        point_shape="rounded",
    )

    frustums = []
    for index, (c2w, intrinsic, image) in enumerate(zip(poses, intrinsics, images)):
        height = int(round(intrinsic[1, 2] * 2.0))
        width = int(round(intrinsic[0, 2] * 2.0))
        fov = 2.0 * math.atan(max(height, 1) / (2.0 * max(float(intrinsic[1, 1]), 1e-6)))
        aspect = float(max(width, 1)) / float(max(height, 1))
        frustums.append(
            server.scene.add_camera_frustum(
                f"/cameras/{index:06d}",
                fov=fov,
                aspect=aspect,
                scale=args.camera_scale,
                line_width=1.0,
                color=(255, 180, 40),
                image=image,
                wxyz=viser_transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )
        )

    @point_size.on_update
    def _(_) -> None:
        cloud.point_size = float(point_size.value)

    @min_confidence.on_update
    def _(_) -> None:
        mask = confidence >= float(min_confidence.value)
        cloud.points = points[mask]
        cloud.colors = colors[mask]

    @show_points.on_update
    def _(_) -> None:
        cloud.visible = bool(show_points.value)

    @show_cameras.on_update
    def _(_) -> None:
        for frustum in frustums:
            frustum.visible = bool(show_cameras.value)

    @camera_scale.on_update
    def _(_) -> None:
        for frustum in frustums:
            frustum.scale = float(camera_scale.value)

    print(f"Open http://{args.host}:{args.port} in a browser", flush=True)
    if args.serve_seconds > 0:
        time.sleep(args.serve_seconds)
        server.stop()
        return
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Interactive Viser viewer for Anchor3R outputs")
    parser.add_argument("sequence_dir", help="Inference output sequence directory")
    parser.add_argument("--source", choices=("online", "offline"), default="offline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--up-direction", default="-y")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--camera-scale", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true", help="Load and validate data without starting a server")
    parser.add_argument("--serve-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    run_viewer(parse_args())


if __name__ == "__main__":
    main()
