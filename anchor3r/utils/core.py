from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class dotdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = self._wrap(value)
        return self

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, dotdict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value


class InferenceModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


def pad_image(image: torch.Tensor, size, mode: str = "constant", value: float = 0.0):
    batch_shape = image.shape[:-3]
    image = image.reshape(-1, *image.shape[-3:])
    height, width = image.shape[-2:]
    target_h, target_w = int(size[0]), int(size[1])
    if target_h >= height and target_w >= width:
        image = F.pad(image, (0, target_w - width, 0, target_h - height), mode=mode, value=value)
    else:
        if target_h > height:
            image = F.pad(image, (0, 0, 0, target_h - height), mode=mode, value=value)
        if target_w > width:
            image = F.pad(image, (0, target_w - width, 0, 0), mode=mode, value=value)
        image = image[..., :target_h, :target_w]
    return image.reshape(*batch_shape, *image.shape[-3:])
