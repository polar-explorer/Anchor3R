from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
TRAIN_PREPROCESS_SIZE = 518
TRAIN_PATCH_SIZE = 14
TRAIN_SAFE_BOUND = 4
PREPROCESS_MODES = ("center", "intrinsics")
_EVC_INTRINSICS_CACHE: dict[str, "_SharedEvcIntrinsics"] = {}


@dataclass
class ImageSequence:
    name: str
    paths: list[str]


@dataclass(frozen=True)
class _SharedEvcIntrinsics:
    k: np.ndarray
    height: int | None
    width: int | None


def list_sequences(images: str, max_frames: int | None = None) -> list[ImageSequence]:
    images = os.path.abspath(os.path.expanduser(images))
    if os.path.isfile(images):
        with open(images, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        return [ImageSequence(os.path.splitext(os.path.basename(images))[0], _limit(paths, max_frames))]

    direct = _list_images(images)
    if direct:
        return [ImageSequence(os.path.basename(images.rstrip(os.sep)), _limit(direct, max_frames))]

    sequences = []
    for child in sorted(glob.glob(os.path.join(images, "*"))):
        if not os.path.isdir(child):
            continue
        paths = _list_images(child)
        if paths:
            sequences.append(ImageSequence(os.path.basename(child), _limit(paths, max_frames)))
    if not sequences:
        raise RuntimeError(f"No images found under {images}")
    return sequences


def extract_video(video: str, cache_dir: str, stride: int = 1, max_frames: int | None = None) -> ImageSequence:
    video = os.path.abspath(os.path.expanduser(video))
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    name = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), name)
    os.makedirs(out_dir, exist_ok=True)

    cached = _list_images(out_dir)
    if cached:
        return ImageSequence(name, _limit(cached, max_frames))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")
    stride = max(1, int(stride))
    paths = []
    frame_idx = 0
    kept_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            path = os.path.join(out_dir, f"{kept_idx:06d}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            kept_idx += 1
            if max_frames is not None and kept_idx >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not paths:
        raise RuntimeError(f"No frames extracted from {video}")
    return ImageSequence(name, paths)


def load_images(
    paths: Sequence[str],
    *,
    target_hw: tuple[int, int] | None = None,
    num_workers: int = 0,
    preprocess: str = "center",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load float32 RGB in [0, 1] using center or calibrated evaluation preprocessing."""
    paths = list(paths)
    if not paths:
        raise ValueError("load_images requires at least one image path.")
    preprocess = _preprocess_mode(preprocess)

    workers = max(0, int(num_workers))
    if workers <= 1 or len(paths) == 1:
        images, target_hw = _load_images_serial(paths, target_hw=target_hw, preprocess=preprocess)
    else:
        images, target_hw = _load_images_parallel(
            paths,
            target_hw=target_hw,
            num_workers=workers,
            preprocess=preprocess,
        )

    assert target_hw is not None
    batch = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).contiguous()
    return batch.unsqueeze(0), target_hw


def _load_one_image(
    path: str,
    *,
    target_hw: tuple[int, int] | None,
    preprocess: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    if preprocess == "center":
        return _load_center_image(path, target_hw=target_hw)
    rgb = torch.from_numpy(_read_rgb(path)).float().div_(255.0).numpy()
    intrinsic = _load_evc_intrinsics_for_image(path)
    if intrinsic is None:
        raise RuntimeError(f"Missing EVC intrinsics for image: {path}")
    return _resize_crop(
        rgb,
        target_hw=target_hw,
        intrinsic=intrinsic,
    )


def _load_center_image(
    path: str, *, target_hw: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    long_edge = max(image.size)
    resized_wh = tuple(int(round(size * TRAIN_PREPROCESS_SIZE / long_edge)) for size in image.size)
    if min(resized_wh) < TRAIN_PATCH_SIZE:
        raise ValueError(f"Image is too narrow for patch-aligned center cropping: {path}")
    interpolation = Image.Resampling.LANCZOS if long_edge > TRAIN_PREPROCESS_SIZE else Image.Resampling.BICUBIC
    image = image.resize(resized_wh, interpolation)
    center_x, center_y = image.width // 2, image.height // 2
    half_width = (2 * center_x // TRAIN_PATCH_SIZE) * (TRAIN_PATCH_SIZE // 2)
    half_height = (2 * center_y // TRAIN_PATCH_SIZE) * (TRAIN_PATCH_SIZE // 2)
    if image.width == image.height:
        half_height = int(3 * half_width / 4) // (TRAIN_PATCH_SIZE // 2) * (TRAIN_PATCH_SIZE // 2)
    image = image.crop((center_x - half_width, center_y - half_height,
                        center_x + half_width, center_y + half_height))
    actual_hw = (image.height, image.width)
    if target_hw is not None and actual_hw != target_hw:
        raise ValueError(f"Center-preprocessed image size changed: expected {target_hw}, got {actual_hw}: {path}")
    return np.asarray(image, dtype=np.float32) / 255.0, actual_hw


def _load_images_serial(
    paths: Sequence[str],
    *,
    target_hw: tuple[int, int] | None,
    preprocess: str,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    images = []
    for path in paths:
        rgb, target_hw = _load_one_image(path, target_hw=target_hw, preprocess=preprocess)
        images.append(rgb)
    assert target_hw is not None
    return images, target_hw


def _load_images_parallel(
    paths: Sequence[str],
    *,
    target_hw: tuple[int, int] | None,
    num_workers: int,
    preprocess: str,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    max_workers = min(int(num_workers), len(paths))
    if target_hw is None:
        first, target_hw = _load_one_image(paths[0], target_hw=None, preprocess=preprocess)
        rest_paths = list(paths[1:])
        if not rest_paths:
            return [first], target_hw
        with ThreadPoolExecutor(max_workers=min(max_workers, len(rest_paths))) as executor:
            rest = list(
                executor.map(
                    lambda path: _load_one_image(path, target_hw=target_hw, preprocess=preprocess)[0],
                    rest_paths,
                )
            )
        return [first, *rest], target_hw

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(
            executor.map(
                lambda path: _load_one_image(path, target_hw=target_hw, preprocess=preprocess)[0],
                paths,
            )
        )
    return images, target_hw


def save_results(
    out_dir: str,
    *,
    intri: torch.Tensor,
    online_w2c: torch.Tensor | None,
    offline_w2c: torch.Tensor,
    pose_dir_name: str = "poses",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pose_dir = os.path.join(out_dir, pose_dir_name)
    if online_w2c is not None:
        _save_matrix_txt(os.path.join(pose_dir, "online_pose.txt"), online_w2c[0].cpu().numpy())
    _save_matrix_txt(os.path.join(pose_dir, "offline_pose.txt"), offline_w2c[0].cpu().numpy())
    _save_intri_txt(os.path.join(pose_dir, "intri.txt"), intri[0].cpu().numpy())


def save_dpt_outputs(
    out_dir: str,
    *,
    image_hw: tuple[int, int],
    dpt_map: torch.Tensor | None = None,
    dpt_conf: torch.Tensor | None = None,
) -> None:
    height, width = image_hw
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    if dpt_map is not None:
        np.save(os.path.join(depth_dir, "dpt_map.npy"), _reshape_dpt(dpt_map, height, width))
    if dpt_conf is not None:
        np.save(os.path.join(depth_dir, "dpt_conf.npy"), _reshape_dpt(dpt_conf, height, width))


def save_geometry_outputs(
    out_dir: str,
    *,
    image_paths: list[str],
    image_hw: tuple[int, int],
    intri: torch.Tensor,
    dpt_map: torch.Tensor,
    dpt_conf: torch.Tensor | None = None,
    online_w2c: torch.Tensor | None = None,
    offline_w2c: torch.Tensor | None = None,
    pose_dir_name: str = "poses",
    save_world_points: bool = True,
    save_confidence: bool = True,
) -> None:
    height, width = image_hw
    points = _depth_to_local_points(
        _reshape_dpt(dpt_map, height, width),
        intri[0].detach().cpu().numpy(),
    )
    confidence = None
    if save_confidence and dpt_conf is not None:
        confidence = _reshape_dpt(dpt_conf, height, width)
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]

    geometry_dir = os.path.join(out_dir, "geometry")
    point_dir = os.path.join(geometry_dir, "local_points")
    conf_dir = os.path.join(geometry_dir, "confidence")
    world_pose_arrays = {}
    if save_world_points:
        world_pose_arrays = {
            name: pose_array
            for name, pose_array in (
                ("online", _as_w2c_array(online_w2c)),
                ("offline", _as_w2c_array(offline_w2c)),
            )
            if pose_array is not None
        }
    world_point_dirs = {name: f"{name}_world_points" for name in world_pose_arrays}
    os.makedirs(point_dir, exist_ok=True)
    if confidence is not None:
        os.makedirs(conf_dir, exist_ok=True)
    for dirname in world_point_dirs.values():
        os.makedirs(os.path.join(geometry_dir, dirname), exist_ok=True)

    frame_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths[: points.shape[0]]]
    for name, pose_array in world_pose_arrays.items():
        if pose_array.shape[0] < len(frame_names):
            raise ValueError(
                f"{name} pose count {pose_array.shape[0]} is smaller than geometry frame count {len(frame_names)}."
            )
    for idx, name in enumerate(frame_names):
        np.save(os.path.join(point_dir, f"{name}.npy"), points[idx])
        for pose_name, pose_array in world_pose_arrays.items():
            world_points = _local_points_to_world(points[idx], pose_array[idx])
            np.save(os.path.join(geometry_dir, world_point_dirs[pose_name], f"{name}.npy"), world_points)
        if confidence is not None and idx < confidence.shape[0]:
            np.save(os.path.join(conf_dir, f"{name}.npy"), confidence[idx].astype(np.float32, copy=False))

    manifest = {
        "scene": os.path.basename(os.path.normpath(out_dir)),
        "image_dir": os.path.abspath(os.path.dirname(image_paths[0])) if image_paths else None,
        "pose_dir": f"../{pose_dir_name}",
        "pose_files": {"offline": f"../{pose_dir_name}/offline_pose.txt"},
        "local_points_dir": "local_points",
        "world_points_dirs": world_point_dirs,
        "confidence_dir": "confidence" if confidence is not None else None,
        "point_frame": "camera",
        "confidence_available": confidence is not None,
        "colors_saved": False,
        "world_points_saved": bool(world_point_dirs),
        "format": "npy",
        "source": "dpt_unprojection",
        "frames": frame_names,
    }
    if online_w2c is not None:
        manifest["pose_files"]["online"] = f"../{pose_dir_name}/online_pose.txt"
    with open(os.path.join(geometry_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _list_images(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(path, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    return sorted(set(paths))


def _limit(paths: list[str], max_frames: int | None) -> list[str]:
    return paths if max_frames is None else paths[: int(max_frames)]


def _reshape_dpt(tensor: torch.Tensor, height: int, width: int) -> np.ndarray:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2] != height * width:
        raise ValueError(
            f"Expected DPT output shape [1, frames, {height * width}, channels], got {tuple(tensor.shape)}."
        )
    return tensor[0].reshape(tensor.shape[1], height, width, tensor.shape[-1]).cpu().numpy()


def _depth_to_local_points(depth: np.ndarray, intri: np.ndarray) -> np.ndarray:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    points = []
    for idx, depth_i in enumerate(depth):
        k = intri[min(idx, len(intri) - 1)]
        h, w = depth_i.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        z = depth_i.astype(np.float32, copy=False)
        x = (u - float(k[0, 2])) / float(k[0, 0]) * z
        y = (v - float(k[1, 2])) / float(k[1, 1]) * z
        point = np.stack((x, y, z), axis=-1)
        point[~np.isfinite(point)] = np.nan
        point[z <= 0] = np.nan
        points.append(point.astype(np.float32, copy=False))
    return np.stack(points)


def _as_w2c_array(w2c: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if w2c is None:
        return None
    if isinstance(w2c, torch.Tensor):
        array = w2c.detach().cpu().numpy()
    else:
        array = np.asarray(w2c)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for camera poses, got shape {array.shape}.")
        array = array[0]
    if array.ndim != 3 or array.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected camera poses with shape [frames, 3, 4] or [frames, 4, 4], got {array.shape}.")
    return array.astype(np.float32, copy=False)


def _local_points_to_world(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    rot = w2c[:3, :3].astype(np.float32, copy=False)
    trans = w2c[:3, 3].astype(np.float32, copy=False)
    centered = points.astype(np.float32, copy=False) - trans.reshape(1, 1, 3)
    world = np.einsum("ij,hwj->hwi", rot.T, centered)
    return world.astype(np.float32, copy=False)


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _preprocess_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_MODES}, got: {value}")
    return mode


def _resize_crop(
    rgb: np.ndarray,
    *,
    target_hw: tuple[int, int] | None,
    intrinsic: _SharedEvcIntrinsics,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = rgb.shape[:2]
    if target_hw is None:
        aspect_ratio = float(h) / float(w)
        target_h = int(aspect_ratio * TRAIN_PREPROCESS_SIZE)
        target_w = TRAIN_PREPROCESS_SIZE
        out_h = max(TRAIN_PATCH_SIZE, (target_h // TRAIN_PATCH_SIZE) * TRAIN_PATCH_SIZE)
        out_w = max(TRAIN_PATCH_SIZE, (target_w // TRAIN_PATCH_SIZE) * TRAIN_PATCH_SIZE)
        target_hw = (out_h, out_w)
    else:
        out_h, out_w = target_hw

    resize_scale = max(
        float(out_h + TRAIN_SAFE_BOUND) / float(h),
        float(out_w + TRAIN_SAFE_BOUND) / float(w),
    )
    new_h = max(out_h, int(h * resize_scale))
    new_w = max(out_w, int(w * resize_scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    k = intrinsic.k.astype(np.float32, copy=True)
    source_w = float(intrinsic.width or w)
    source_h = float(intrinsic.height or h)
    k[0:1] *= new_w / source_w
    k[1:2] *= new_h / source_h
    top = int(round(float(k[1, 2]))) - out_h // 2
    left = int(round(float(k[0, 2]))) - out_w // 2
    return _crop_or_pad_at_offset(resized, out_h, out_w, top, left), target_hw


def _crop_or_pad_at_offset(rgb: np.ndarray, out_h: int, out_w: int, top: int, left: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    bottom = top + out_h
    right = left + out_w
    if top < 0 or left < 0 or bottom > h or right > w:
        pad_top = max(0, -top)
        pad_bottom = max(0, bottom - h)
        pad_left = max(0, -left)
        pad_right = max(0, right - w)
        rgb = np.pad(
            rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        top += pad_top
        left += pad_left
    return rgb[top : top + out_h, left : left + out_w]


def _load_evc_intrinsics_for_image(path: str) -> _SharedEvcIntrinsics | None:
    intri_path = _evc_intrinsics_path(path)
    if intri_path is None:
        return None
    if intri_path not in _EVC_INTRINSICS_CACHE:
        _EVC_INTRINSICS_CACHE[intri_path] = _read_shared_evc_intrinsics(intri_path)
    return _EVC_INTRINSICS_CACHE[intri_path]


def _evc_intrinsics_path(path: str) -> str | None:
    image_dir = os.path.dirname(os.path.abspath(path))
    camera_name = os.path.basename(image_dir)
    images_dir = os.path.dirname(image_dir)
    if os.path.basename(images_dir) != "images":
        return None
    scene_dir = os.path.dirname(images_dir)
    intri_path = os.path.join(scene_dir, "cameras", camera_name, "intri.yml")
    return intri_path if os.path.exists(intri_path) else None


def _read_shared_evc_intrinsics(intri_path: str) -> _SharedEvcIntrinsics:
    storage = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Failed to open EVC intrinsics: {intri_path}")
    try:
        names_node = storage.getNode("names")
        if names_node.empty() or not names_node.isSeq() or names_node.size() == 0:
            raise RuntimeError(f"EVC intrinsics must contain a non-empty names sequence: {intri_path}")
        name = names_node.at(0).string()
        return _read_intrinsic_record(storage, name, intri_path)
    finally:
        storage.release()


def _read_intrinsic_record(storage: cv2.FileStorage, name: str, intri_path: str) -> _SharedEvcIntrinsics:
    k_node = storage.getNode(f"K_{name}")
    if k_node.empty():
        raise RuntimeError(f"Missing K_{name} in EVC intrinsics: {intri_path}")
    k = k_node.mat()
    if k is None:
        raise RuntimeError(f"Failed to read K_{name} in EVC intrinsics: {intri_path}")
    d_node = storage.getNode(f"D_{name}")
    if not d_node.empty():
        d = d_node.mat()
        if d is not None and float(np.sum(np.abs(d))) != 0.0:
            raise RuntimeError(
                "Anchor3R public preprocessing expects rectified KITTI/VBR images with zero distortion; "
                f"found non-zero D_{name} in {intri_path}."
            )
    height_node = storage.getNode(f"H_{name}")
    width_node = storage.getNode(f"W_{name}")
    height = None if height_node.empty() else int(round(height_node.real()))
    width = None if width_node.empty() else int(round(width_node.real()))
    return _SharedEvcIntrinsics(k=np.asarray(k), height=height, width=width)


def _save_matrix_txt(path: str, mats: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for idx, mat in enumerate(mats):
            values = [idx] + mat[:3, :3].reshape(-1).tolist() + mat[:3, 3].reshape(-1).tolist()
            f.write(" ".join(str(x) for x in values) + "\n")


def _save_intri_txt(path: str, mats: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for idx, mat in enumerate(mats):
            values = [idx, mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2]]
            f.write(" ".join(str(x) for x in values) + "\n")
