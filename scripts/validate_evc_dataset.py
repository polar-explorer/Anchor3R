from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_camera_dir(image_dir: Path) -> tuple[int, Path]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    if image_dir.parent.name != "images":
        raise RuntimeError(f"Expected .../images/<camera>, got {image_dir}")
    intri_path = image_dir.parent.parent / "cameras" / image_dir.name / "intri.yml"
    if not intri_path.is_file():
        raise FileNotFoundError(intri_path)

    storage = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Cannot open {intri_path}")
    try:
        names = storage.getNode("names")
        if names.empty() or not names.isSeq() or names.size() == 0:
            raise RuntimeError(f"Missing non-empty names sequence in {intri_path}")
        first_name = names.at(0).string()
        matrix = storage.getNode(f"K_{first_name}").mat()
        if matrix is None or matrix.shape != (3, 3):
            raise RuntimeError(f"Missing 3x3 K_{first_name} in {intri_path}")
        distortion = storage.getNode(f"D_{first_name}")
        if not distortion.empty():
            values = distortion.mat()
            if values is not None and abs(values).sum() != 0:
                raise RuntimeError(f"Non-zero distortion is unsupported in {intri_path}")
    finally:
        storage.release()
    return len(images), intri_path


def main() -> None:
    parser = argparse.ArgumentParser("Validate an Anchor3R EVC camera directory")
    parser.add_argument("images", help="Path ending in scene/images/<camera>")
    args = parser.parse_args()
    image_dir = Path(args.images).expanduser().resolve()
    count, intri_path = validate_camera_dir(image_dir)
    print(f"OK: {count} images; intrinsics={intri_path}")


if __name__ == "__main__":
    main()
