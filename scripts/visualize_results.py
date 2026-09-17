from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_w2c(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    has_frame_id = rows.shape[1] == 13
    if has_frame_id:
        rows = rows[:, 1:]
    if rows.shape[1] != 12:
        raise ValueError(f"Expected 12 pose values, optionally preceded by frame id: {path}")
    poses = np.tile(np.eye(4, dtype=np.float64), (len(rows), 1, 1))
    if has_frame_id:
        poses[:, :3, :3] = rows[:, :9].reshape(-1, 3, 3)
        poses[:, :3, 3] = rows[:, 9:]
    else:
        poses[:, :3, :4] = rows.reshape(-1, 3, 4)
    return poses


def camera_centers(w2c: np.ndarray) -> np.ndarray:
    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), translation)


def visualize(sequence_dir: Path, output: Path) -> Path:
    candidates = {
        "Online": sequence_dir / "poses" / "online_pose.txt",
        "Offline": sequence_dir / "poses" / "offline_pose.txt",
        "Loop online": sequence_dir / "poses_offline" / "online_pose.txt",
        "Loop offline": sequence_dir / "poses_offline" / "offline_pose.txt",
    }
    trajectories = {
        label: camera_centers(load_w2c(path))
        for label, path in candidates.items()
        if path.is_file()
    }
    if not trajectories:
        raise FileNotFoundError(f"No pose files found under {sequence_dir}")

    figure = plt.figure(figsize=(12, 9))
    ax3d = figure.add_subplot(2, 2, 1, projection="3d")
    ax_xy = figure.add_subplot(2, 2, 2)
    ax_xz = figure.add_subplot(2, 2, 3)
    ax_yz = figure.add_subplot(2, 2, 4)
    for label, xyz in trajectories.items():
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label)
        ax_xy.plot(xyz[:, 0], xyz[:, 1], label=label)
        ax_xz.plot(xyz[:, 0], xyz[:, 2], label=label)
        ax_yz.plot(xyz[:, 1], xyz[:, 2], label=label)

    ax3d.set_title("3D trajectory")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    for axis, title, xlabel, ylabel in (
        (ax_xy, "XY projection", "X", "Y"),
        (ax_xz, "XZ projection", "X", "Z"),
        (ax_yz, "YZ projection", "Y", "Z"),
    ):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
    ax3d.legend()
    ax_xy.legend()
    figure.suptitle(sequence_dir.name)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Visualize Anchor3R camera trajectories")
    parser.add_argument("sequence_dir", help="Sequence output directory")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else sequence_dir / "trajectory.png"
    print(visualize(sequence_dir, output))


if __name__ == "__main__":
    main()
