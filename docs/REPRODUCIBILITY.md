# Inference and Pose Evaluation

This release includes inference, optional loop closure, evaluation and local
visualization. It does not include training code. The maintainer confirms that
the release weights match the project demos and differ slightly in results from
the paper checkpoint; exact reproduction of all paper results is not claimed.

Initialization and training-data sources follow the Anchor3R paper: VGGT and a
mixture of real and synthetic datasets including WildRGB, ScanNet, HyperSim,
Mapillary, Replica, Mapfree, TartanAir, MVS-Synth, Virtual KITTI, Aria Synthetic
Environments, Spring, Waymo Open, BlendedMVS, Co3Dv2, MegaDepth and DL3DV. Exact
dataset revisions and split/overlap details are not separately specified here.

## Inputs

Obtain data from its owner under the applicable terms. The repository does not
distribute benchmark images. KITTI odometry conversion is available:

```bash
python scripts/kitti_to_evc.py --src /path/to/kitti_odometry \
  --out /path/to/kitti_evc --seqs 00,01
python scripts/validate_evc_dataset.py /path/to/kitti_evc/00/images/02
```

For VBR, supply rectified RGB images and matching EVC camera calibration. No raw
VBR conversion pipeline is included. Layout validation alone does not establish
that frame selection or calibration matches a paper benchmark.

## Full-Sequence Inference

For Anchor3R on KITTI, use `--camera-preprocess`.
Run `DATASET_TYPE=kitti bash scripts/run_evc_dataset.sh ...`
to select this mode in batch inference. **VBR uses `--no-camera-preprocess`**
(geometric-center crop), including when calibration files are present; use
`DATASET_TYPE=vbr` for VBR batch inference. The batch script defaults to
`DATASET_TYPE=generic`, which also disables camera preprocessing. Dataset
selection overrides the YAML setting, without guessing from path names.
The general-purpose CLI also remains center-crop by default.

Use `--camera-preprocess` to match legacy calibrated evaluation: convert RGB
to float32 `[0, 1]` before OpenCV `INTER_AREA` resizing, scale intrinsics in
float32, then crop/pad around the rounded principal point. Width 518, patch
alignment 14 and safe bound 4 are unchanged. CPU staging and device transfer
preserve float32 pixels without uint8 quantization or a second normalization.
Image staging/transfer uses four times the bytes of the former uint8 path;
this does not imply four times the total model memory. Remove
`data.transfer_uint8: true` from custom configurations. Earlier runs using
uint8 resizing must be rerun for a numerically consistent comparison.

Without this flag (default `data.camera_preprocess: false`), preprocessing uses
HorizonStream-style PIL long-edge resize to 518 followed by geometric-center
crop, then float32 normalization. Downsampling uses LANCZOS; upsampling uses
BICUBIC. Crops use patch size 14. Square images use the HorizonStream 3:4
height/width preference with extra patch alignment: 378×518 instead of
388×518, which Anchor3R's patch embedding cannot accept. This mode ignores camera calibration and does not
use the calibrated branch's safe bound or AREA interpolation. Use
`--no-camera-preprocess` to override an enabled YAML setting. Legacy
`--preprocess intrinsics|center` remains an alias. Neither mode is selected
automatically by dataset name; record the setting when comparing results.

Omit `--max-frames` to process every image in the selected camera directory:

```bash
python infer.py --images /path/to/kitti_evc/00/images/02 \
  --sequence-name 00 --camera-preprocess \
  --checkpoint checkpoints/Anchor3R.pt --output outputs_anchor3r/kitti \
  --window-size 10 --chunk-size 10 --infer-mode offline --no-loop
```

For loop-enabled inference replace `--no-loop` with `--loop`, provide local
SALAD/DINOv2 paths, and record retrieval settings:

```text
--salad-checkpoint /path/to/dino_salad.ckpt
--dino-checkpoint /path/to/dinov2_vitb14_pretrain.pth
--loop-keyframe-stride 2 --loop-retrieval-top-k 3
--loop-temporal-threshold 30 --salad-score-threshold 0.85
--loop-insert-position before --loop-edge-weight 1
```

Offline mode alone does not enable loops. A no-loop control should use the same
images, checkpoint and inference settings. Loop-enabled runs save only the
loop-corrected `offline_pose.txt` under `poses_offline/`; the intermediate online
trajectory is not exported.

## Pose Convention and Alignment

Saved rows contain `frame_id`, nine row-major rotation entries, then three
translation entries. They are **world-to-camera (w2c)** transforms, not a flattened
3×4 matrix. Invert them to obtain camera-to-world transforms and camera centers.
Depth/translation scale is not guaranteed metric.

```bash
python scripts/evaluate_trajectory.py \
  --pose-pred outputs_anchor3r/kitti/00/poses/offline_pose.txt \
  --images /path/to/kitti_evc/00/images/02 \
  --gt-poses /path/to/kitti_evc/00/cameras/02/extri.yml \
  --gt-format vbr-extri --alignment sim3
```

For online-mode inference, use `poses/` instead of `poses_offline/`. For no-loop
outputs, run the evaluator separately for `online_pose.txt` and
`offline_pose.txt`. Sim(3) fits
one rotation, translation and scale to each full matched trajectory; SE(3) fits
only rotation and translation. ATE is the RMSE of aligned camera-center errors.
The adjacent-frame translation/rotation errors are not KITTI segment-based drift
metrics. Check `valid_frames` against the intended sequence length: the evaluator
can omit unmatched or non-finite samples, which must not go unreported.

The evaluator supports EVC extrinsics, named rotation-then-translation text rows,
and pose arrays in NumPy format. Convert other text layouts explicitly before
evaluation. Images and prediction rows must refer to the same temporal ordering.

## Reporting Results

Record the checkpoint SHA256, source revision, camera, image order/stride/count,
preprocessing, alignment, window/chunk size and loop settings. Retain the loop
candidate count and actual graph-edge count, not just the enabled flag. Report
hardware, precision, synchronization and peak-memory measurement conditions for
performance claims. Model context is bounded; retained outputs, full-pairwise
loop retrieval and global optimization have separate memory/computation costs.
