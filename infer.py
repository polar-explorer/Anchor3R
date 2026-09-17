from __future__ import annotations

import argparse
import os
from importlib import resources

import yaml


def _log(message: str) -> None:
    print(f"[Anchor3R-CLI] {message}", flush=True)


def parse_config(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser("Anchor3R inference")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. Defaults to the config bundled with the package.",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--images", default=None)
    inputs.add_argument("--video", default=None)
    parser.add_argument("--sequence-name", default=None)
    preprocessing = parser.add_mutually_exclusive_group()
    preprocessing.add_argument(
        "--camera-preprocess", dest="camera_preprocess", action="store_true", default=None,
        help="Use legacy float32 preprocessing and calibrated principal-point cropping.",
    )
    preprocessing.add_argument(
        "--no-camera-preprocess", dest="camera_preprocess", action="store_false",
        help="Use HorizonStream-style PIL resize and geometric-center crop (default).",
    )
    preprocessing.add_argument(
        "--preprocess", choices=("center", "intrinsics"), default=None,
        help="Compatibility alias: center disables camera preprocessing; intrinsics enables it.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--infer-mode", choices=("online", "offline"), default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    loop = parser.add_mutually_exclusive_group()
    loop.add_argument("--loop", dest="loop_enabled", action="store_true", default=None)
    loop.add_argument("--no-loop", dest="loop_enabled", action="store_false")
    parser.add_argument("--loop-keyframe-stride", type=int, default=None)
    parser.add_argument("--loop-retrieval-top-k", type=int, default=None)
    parser.add_argument("--loop-temporal-threshold", type=int, default=None)
    parser.add_argument("--salad-checkpoint", default=None)
    parser.add_argument("--dino-checkpoint", default=None)
    parser.add_argument("--salad-score-threshold", type=float, default=None)
    parser.add_argument("--loop-insert-position", choices=("before", "after"), default=None)
    parser.add_argument("--loop-edge-weight", type=float, default=None)
    parser.add_argument("--image-num-workers", type=int, default=None)
    parser.add_argument("--prefetch-chunks", default=None)
    parser.add_argument("--pin-memory", default=None)
    parser.add_argument("--save-geometry", action="store_true")
    parser.add_argument("--no-save-geometry", action="store_true")
    parser.add_argument("--save-geometry-world-points", action="store_true")
    parser.add_argument("--no-save-geometry-world-points", action="store_true")
    parser.add_argument("--save-geometry-confidence", action="store_true")
    parser.add_argument("--no-save-geometry-confidence", action="store_true")
    parser.add_argument("--save-dpt-map", action="store_true")
    parser.add_argument("--no-save-dpt-map", action="store_true")
    parser.add_argument("--save-dpt-conf", action="store_true")
    parser.add_argument("--no-save-dpt-conf", action="store_true")
    args = parser.parse_args(argv)

    if args.config is None:
        config_resource = resources.files("anchor3r.configs").joinpath("infer.yaml")
        cfg_path = str(config_resource)
        config_text = config_resource.read_text(encoding="utf-8")
    else:
        cfg_path = os.path.abspath(os.path.expanduser(args.config))
        with open(cfg_path, encoding="utf-8") as handle:
            config_text = handle.read()
    _log(f"loading config: {cfg_path}")
    cfg = yaml.safe_load(config_text) or {}
    if not isinstance(cfg, dict):
        parser.error("The configuration must be a YAML mapping.")
    _log("config loaded")
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("loop", {})
    cfg.setdefault("output", {})

    if args.images is not None:
        cfg["data"]["images"] = args.images
        cfg["data"]["video"] = None
    if args.video is not None:
        cfg["data"]["video"] = args.video
        cfg["data"]["images"] = None
    if args.sequence_name is not None:
        cfg["data"]["sequence_name"] = args.sequence_name
    if args.camera_preprocess is not None or args.preprocess is not None:
        cfg["data"].pop("preprocess", None)
        cfg["data"]["camera_preprocess"] = (
            args.camera_preprocess if args.camera_preprocess is not None else args.preprocess == "intrinsics"
        )
    if args.checkpoint is not None:
        cfg["model"]["checkpoint"] = args.checkpoint
    if args.output is not None:
        cfg["output"]["root"] = args.output
    if args.device is not None:
        cfg["device"] = args.device
    if args.infer_mode is not None:
        cfg["inference"]["mode"] = args.infer_mode
    if args.window_size is not None:
        cfg["inference"]["window_size"] = args.window_size
    if args.chunk_size is not None:
        cfg["inference"]["chunk_size"] = args.chunk_size
    if args.max_frames is not None:
        cfg["data"]["max_frames"] = args.max_frames
    if args.loop_enabled is not None:
        cfg["loop"]["enabled"] = args.loop_enabled
    if args.loop_keyframe_stride is not None:
        cfg["loop"]["keyframe_stride"] = args.loop_keyframe_stride
    if args.loop_retrieval_top_k is not None:
        cfg["loop"]["retrieval_top_k"] = args.loop_retrieval_top_k
    if args.loop_temporal_threshold is not None:
        cfg["loop"]["temporal_threshold"] = args.loop_temporal_threshold
    if args.salad_checkpoint is not None:
        cfg["loop"]["salad_checkpoint"] = args.salad_checkpoint
    if args.dino_checkpoint is not None:
        cfg["loop"]["dino_checkpoint"] = args.dino_checkpoint
    if args.salad_score_threshold is not None:
        cfg["loop"]["salad_score_threshold"] = args.salad_score_threshold
    if args.loop_insert_position is not None:
        cfg["loop"]["insert_position"] = args.loop_insert_position
    if args.loop_edge_weight is not None:
        cfg["loop"]["loop_edge_weight"] = args.loop_edge_weight
    if args.image_num_workers is not None:
        cfg["data"]["image_num_workers"] = args.image_num_workers
    if args.prefetch_chunks is not None:
        cfg["data"]["prefetch_chunks"] = args.prefetch_chunks
    if args.pin_memory is not None:
        cfg["data"]["pin_memory"] = args.pin_memory
    if args.save_geometry:
        cfg["output"]["save_geometry"] = True
    if args.no_save_geometry:
        cfg["output"]["save_geometry"] = False
    if args.save_geometry_world_points:
        cfg["output"]["save_geometry_world_points"] = True
    if args.no_save_geometry_world_points:
        cfg["output"]["save_geometry_world_points"] = False
    if args.save_geometry_confidence:
        cfg["output"]["save_geometry_confidence"] = True
    if args.no_save_geometry_confidence:
        cfg["output"]["save_geometry_confidence"] = False
    if args.save_dpt_map:
        cfg["output"]["save_dpt_map"] = True
    if args.no_save_dpt_map:
        cfg["output"]["save_dpt_map"] = False
    if args.save_dpt_conf:
        cfg["output"]["save_dpt_conf"] = True
    if args.no_save_dpt_conf:
        cfg["output"]["save_dpt_conf"] = False
    return cfg


def main() -> None:
    cfg = parse_config()
    from anchor3r.runtime.inference import run_inference

    _log("dispatching run_inference")
    run_inference(cfg)
    _log("run_inference returned")


if __name__ == "__main__":
    main()
