# Installation

## Conda setup

Use Python 3.11 and let Conda provide SuiteSparse, CHOLMOD and the other
compiled numerical dependencies. This avoids compiling `scikit-sparse` and
does not require system SuiteSparse packages:

```bash
conda create -n anchor3r -c conda-forge python=3.11 numpy scipy scikit-sparse suitesparse \
  pyyaml opencv pillow matplotlib
conda activate anchor3r
```

Install PyTorch 2.6.0 separately so that the CUDA wheel matches the installed
NVIDIA driver, then install the viewer and this package without replacing the
Conda numerical libraries:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install viser
python -m pip install -e . --no-deps
```

For CPU installation, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

The tested Conda setup is intended for Linux. Windows support is not part of
this release. `requirements.txt` remains available for environments that
already provide compatible SuiteSparse libraries, but it is not the
recommended installation path.

## Run the CLI

```bash
anchor3r-infer --help
anchor3r-viser --help
```

For GPU inference, check actual CUDA access before starting the full model:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
```

Skip the CUDA check when deliberately using CPU execution.

Download the release weights:

```bash
python -m pip install "huggingface_hub>=0.27"
python scripts/download_weights.py
anchor3r-infer --images /path/to/authorized/images \
  --checkpoint checkpoints/Anchor3R.pt --output outputs_anchor3r/demo
```

## Common Problems

- **Missing `cholmod.h` or SuiteSparse libraries:** recreate the environment
  with `scikit-sparse suitesparse` from `conda-forge`. Do not mix Conda's
  SuiteSparse libraries with a separately compiled pip installation.
- **OpenCV import fails:** recreate the environment with Conda's `opencv`
  package. Do not install multiple competing OpenCV distributions in the same
  environment.
- **Images are listed but cannot be read inside a container:** check that the
  container user can open the image bytes, not just list the directory. Some
  network mounts reject container root; use an authorized UID/GID and mount the
  inputs read-only rather than changing dataset permissions.
- **Package servers are unreachable:** use an accessible package mirror with
  the same pinned versions. Keep TLS verification and distribution package
  signature checks enabled; do not copy installed packages from a training env.
- **NumPy/OpenCV binary mismatch:** do not mix multiple OpenCV distributions or
  an unpinned NumPy 2 installation with this profile. Recreate the Conda
  environment and avoid changing a shared training environment.
- **Missing CUDA:** inspect the startup device log. The current runtime falls
  back to CPU if CUDA is unavailable. Confirm CUDA access before GPU inference.
- **Out of memory:** reduce `--chunk-size`, disable geometry/depth output for a
  trajectory-only run, and use `--max-frames` to shorten the sequence. Reducing chunk size does not
  bound the memory used by all accumulated outputs.
- **Viewer dependency tries to build `manifold3d` on an older Linux host:** use
  `python -m pip install --only-binary=manifold3d viser` to
  request compatible prebuilt dependencies, or use a newer supported system.
  Do not install arbitrary system libraries into a shared training environment.
- **Changing YAML has no effect:** pass `--config infer.yaml`; without
  it the CLI loads the package's bundled defaults.
- **Weights-only load error:** obtain the approved tensor-only release as
  described in the [Hugging Face checkpoint guide](https://huggingface.co/polar-explorer/Anchor3R/blob/main/USAGE.md).
  Do not add an unsafe-loading fallback.
