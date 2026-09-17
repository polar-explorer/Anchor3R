from __future__ import annotations

__all__ = ["Anchor3RModel", "run_inference"]


def __getattr__(name: str):
    if name == "Anchor3RModel":
        from anchor3r.runtime import Anchor3RModel

        return Anchor3RModel
    if name == "run_inference":
        from anchor3r.runtime import run_inference

        return run_inference
    raise AttributeError(name)
