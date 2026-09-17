"""Camera geometry and trajectory stitching helpers."""

from anchor3r.geometry.camera import (
    build_camera_trajectories,
    decode_cam_map,
    get_median_rotations,
    mat_to_quat,
    position_averaging,
    quat_to_mat,
    rotation_averaging,
)

__all__ = [
    "build_camera_trajectories",
    "decode_cam_map",
    "get_median_rotations",
    "mat_to_quat",
    "position_averaging",
    "quat_to_mat",
    "rotation_averaging",
]
