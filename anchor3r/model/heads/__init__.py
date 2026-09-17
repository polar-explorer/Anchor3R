"""Prediction heads used by Anchor3R."""

from anchor3r.model.heads.camera import BatchCameraHead, CameraHead
from anchor3r.model.heads.dpt_head import DPTHead

__all__ = ["BatchCameraHead", "CameraHead", "DPTHead"]
