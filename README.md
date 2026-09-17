<div align="center">

# Anchor3R: Streaming 3D Reconstruction with Transient Anchors for Long-Horizon Visual Mapping

**Accepted at CoRL 2026**

**Peilin Tao · Chong Cheng · Yuansen Du · Caiwei Song · Zhengqing Chen · Xiaoyang Guo**  
**Wei Yin · Weiqiang Ren · Qian Zhang · Hainan Cui · Shuhan Shen**

<p>
<a href="https://arxiv.org/abs/2606.05035"><img src="https://img.shields.io/badge/arXiv-2606.05035-b31b1b.svg?logo=arxiv" alt="arXiv"></a>&nbsp;&nbsp;
<a href="https://polar-explorer.github.io/Anchor3R-project-page/"><img src="https://img.shields.io/badge/Project%20Page-Visit-2ea44f.svg" alt="Project Page"></a>&nbsp;&nbsp;
<a href="https://huggingface.co/polar-explorer/Anchor3R"><img src="https://img.shields.io/badge/Weights-Hugging%20Face-ffcc4d.svg?logo=huggingface&logoColor=black" alt="Weights"></a>&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Python-3.9--3.11-3776AB?logo=python&logoColor=white" alt="Python">&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Original%20Code-Apache%202.0-1f6feb" alt="License">
</p>

</div>

## Features

- Current-centric streaming reconstruction with transient anchors.
- Online and loop-corrected offline camera trajectories.
- 48-frame training to 10K+ frame generalization.
- Approximately constant GPU memory usage of 8.2 GB with `chunk_size=1`.

## Installation

Create a Conda environment:

```bash
conda create -n anchor3r -c conda-forge python=3.11 numpy scipy scikit-sparse suitesparse \
  pyyaml opencv pillow matplotlib
conda activate anchor3r
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install viser
python -m pip install -e . --no-deps
```

For CPU-only installation, replace the PyTorch command with:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

The tested setup is intended for Linux.

## Checkpoints

Download the main checkpoint:

```bash
pip install "huggingface_hub>=0.27"
python scripts/download_weights.py
```

Alternatively, place the checkpoint at:

```text
checkpoints/Anchor3R.pt
```

Loop closure additionally requires:

```text
checkpoints/loop/dino_salad.ckpt
checkpoints/loop/dinov2_vitb14_pretrain.pth
```

See the [Hugging Face model card](https://huggingface.co/polar-explorer/Anchor3R)
for checkpoint details.

## Quick Start

Run on a plain image directory using geometric-center preprocessing:

```bash
python infer.py \
  --images /path/to/images \
  --checkpoint checkpoints/Anchor3R.pt \
  --output outputs_anchor3r/demo
```

Run on a video:

```bash
python infer.py \
  --video /path/to/video.mp4 \
  --checkpoint checkpoints/Anchor3R.pt \
  --output outputs_anchor3r/video_demo
```

After installation, `anchor3r-infer` is equivalent to `python infer.py`.

Use `--max-frames 10` to process only the first ten frames. The first
`window_size` frames are always processed together as a full first chunk to
initialize the streaming state; with the default `window_size: 10`, the first
10 images form this full chunk. Subsequent frames are processed in chunks of
`--chunk-size` frames. The default chunk size is 64, but it can be reduced or
increased according to available GPU memory. The model architecture makes the
inference result independent of `chunk_size`; changing it only changes the
memory and throughput profile. `--window-size` controls the model context
window. Accumulated outputs still grow with sequence length, especially with
geometry enabled.

For strict frame-by-frame streaming inference, set `--chunk-size 1`:

```bash
python infer.py \
  --images /path/to/images \
  --checkpoint checkpoints/Anchor3R.pt \
  --output outputs_anchor3r/strict_stream \
  --infer-mode online \
  --chunk-size 1
```

The first `window_size` frames are still processed together as the required
full initialization chunk; with the default `window_size: 10`, frame-by-frame
processing starts from the 11th image.

## KITTI and VBR

Prepare benchmark data in the EVC layout:

```text
dataset_root/
  00/
    images/
      02/
        000000.png
        000001.png
    cameras/
      02/
        intri.yml
        extri.yml          # optional unless evaluating
```

For KITTI, use calibrated principal-point cropping with
`--camera-preprocess`:

```bash
python infer.py \
  --images /path/to/dataset_root/00/images/02 \
  --sequence-name 00 \
  --camera-preprocess \
  --checkpoint checkpoints/Anchor3R.pt \
  --output outputs_anchor3r/kitti
```

VBR uses geometric-center cropping, so pass `--no-camera-preprocess`:

```bash
python infer.py \
  --images /path/to/vbr_evc/scene/images/02 \
  --sequence-name scene \
  --no-camera-preprocess \
  --checkpoint checkpoints/Anchor3R.pt \
  --output outputs_anchor3r/vbr
```

Convert an official KITTI Odometry tree to EVC with:

```bash
python scripts/kitti_to_evc.py \
  --src /path/to/kitti_odometry \
  --out /path/to/kitti_evc \
  --seqs 00,01,02
```

## Inference Modes

- No loop (`--infer-mode online` or `--infer-mode offline --no-loop`): writes
  both `online_pose.txt` and `offline_pose.txt`.
- Loop (`--infer-mode offline --loop`): retrieves and reinserts loop frames and
  writes only the loop-corrected `offline_pose.txt`. The intermediate online
  trajectory is not exported.

Loop closure is disabled by default. Enable it with `--infer-mode offline --loop`
and provide the SALAD and DINOv2 checkpoint paths. To edit YAML settings, pass
`--config infer.yaml` explicitly; setting `loop.enabled: true` requires
`inference.mode: offline`. Relevant
parameters include `keyframe_stride`, `retrieval_top_k`,
`temporal_threshold`, `salad_score_threshold`, `insert_position`, and
`loop_edge_weight`.

## Outputs

Each sequence is written under `OUTPUT_ROOT/<sequence_name>/`:

```text
poses/online_pose.txt
poses/offline_pose.txt
poses/intri.txt
depth/dpt_map.npy                 # optional
depth/dpt_conf.npy                # optional
geometry/local_points/*.npy      # optional
geometry/online_world_points/*   # optional
geometry/offline_world_points/*  # optional
```

The layout above is for no-loop inference. Loop-enabled runs use `poses_offline/`
and contain only `offline_pose.txt`; their `geometry/manifest.json` records only
the exported offline trajectory. Loop-enabled runs also emit loop artifacts.

Pose files contain one row per frame:

```text
frame_id r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz
```

They store world-to-camera transforms. Intrinsic rows use:

```text
frame_id fx fy cx cy
```

Depth and translation are in model scale, not guaranteed meters. Evaluate camera
centers after inverting w2c poses, with the declared alignment protocol.

## Evaluation

Evaluate one saved trajectory against EVC `extri.yml` ground truth using Sim(3)
alignment:

```bash
python scripts/evaluate_trajectory.py \
  --pose-pred outputs_anchor3r/kitti/00/poses/offline_pose.txt \
  --images /path/to/dataset_root/00/images/02 \
  --gt-poses /path/to/dataset_root/00/cameras/02/extri.yml \
  --gt-format vbr-extri \
  --alignment sim3
```

The evaluator writes `metrics.json` and a trajectory/error plot. Supported
ground-truth formats are OpenCV `extri.yml`, text or NumPy arrays containing
world-to-camera or camera-to-world matrices.

For no-loop outputs, run the command once for each pose file. The dataset
runner does this automatically and writes separate `pose_eval_online/` and
`pose_eval_offline/` results; loop outputs only contain the offline pose.

See [the reproduction protocol](docs/REPRODUCIBILITY.md) for full-sequence
commands and evaluation details.

## Visualization

Create trajectory projections from an inference result:

```bash
python scripts/visualize_results.py outputs_anchor3r/kitti/00
```

This writes `trajectory.png` in the sequence directory. To export saved point
maps as PLY, use `tools/export_geometry_world_ply.py --help`.

For an interactive browser viewer similar to HorizonStream's Viser demo, first
run inference with geometry output enabled:

```bash
python infer.py \
  --images /path/to/dataset_root/00/images/02 \
  --sequence-name 00 \
  --camera-preprocess \
  --checkpoint checkpoints/Anchor3R.pt \
  --output outputs_anchor3r/kitti \
  --save-geometry

python demo_viser.py outputs_anchor3r/kitti/00 --source offline --port 8080
```

Open `http://localhost:8080`. The GUI controls point size, confidence filtering,
point-cloud visibility, camera visibility, and camera-frustum scale. Use
`--pixel-stride` and `--max-points` to control browser memory usage.
After package installation, `anchor3r-viser` is equivalent to
`python demo_viser.py`.

`tools/export_geometry_world_ply.py` exports saved local point maps as a colored
world-space binary PLY.

## License and Acknowledgements

Released under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

Use the following BibTeX entry to cite the paper:

```bibtex
@article{tao2026anchor3r,
  title   = {Anchor3R: Streaming 3D Reconstruction with Transient Anchors for Long-Horizon Visual Mapping},
  author  = {Tao, Peilin and Cheng, Chong and Du, Yuansen and Song, Caiwei and Chen, Zhengqing and Guo, Xiaoyang and Yin, Wei and Ren, Weiqiang and Zhang, Qian and Cui, Hainan and Shen, Shuhan},
  journal = {arXiv preprint arXiv:2606.05035},
  year    = {2026}
}
```
