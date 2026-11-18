# SuperLocomotion

Depth-based terrain locomotion for Unitree G1, built on InstinctLab's Hiking
scene with SuperMimic world-model learning, TGI-aligned imagined policy updates
and Success-Guided Sampling (SGS).

This repository contains the current reference-free Hiking task. It does not
include the upstream shadowing, BeyondMimic, InstantTrack or motion-imitation
tasks. The deployed controller uses proprioceptive history and depth; the
physical world model additionally queries privileged terrain during training.

- [Environment requirements](#environment)
- [Clone and prepare dependencies](#1-prepare-linux-and-clone)
- [Install and activate the runtime](#2-install-the-pinned-runtime)
- [Short diagnostic](#3-check-a-small-run-first)
- [Train](#4-start-the-full-training-run)
- [Evaluate and record video](#5-evaluate-an-existing-trained-model)
- [Architecture, training and evaluation details](docs/hiking.md)
- [Final 2,048-environment configuration](configs/hiking/tgi_sgs_2048.json)
- [14,000-iteration training results](docs/results/hiking_14000.json)

Train with `scripts/instinct_rl/train_hiking_native_latent.py`; evaluate or
record video with `scripts/instinct_rl/evaluate_hiking_native_latent.py`.
The maintained route uses BF16 network execution and no adversarial motion prior
or imitation objective. Checkpoints, datasets, logs and environments stay local.

## Environment

| Component | Training runtime |
| --- | --- |
| Operating system | Linux x86_64 |
| GPU / driver | NVIDIA GPU, CUDA 13-compatible driver 580 or newer |
| Python | 3.12 |
| Isaac Sim | 6.2.0.0 |
| Isaac Lab | 41.0.2; exact source revision pinned below |
| Instinct-RL | 1.0.3; exact source revision pinned below |
| Newton | 1.8.0.dev0; exact source revision pinned below |
| PyTorch / TorchVision | 2.11.0+cu130 / 0.26.0+cu130 |
| Warp / NumPy | 1.17.0 / 2.3.1 |

The repository includes G1 robot assets, terrain definitions and the training
configuration. The simulator runtime must be installed separately. No
motion-imitation dataset is needed. Pretrained weights are not included:
a fresh clone can train from scratch; replaying the final learned gait requires
a separately supplied checkpoint.

## 1. Prepare Linux and clone

Use Linux x86_64, an NVIDIA GPU, and a CUDA 13-compatible NVIDIA driver (580 or
newer). Python 3.12 is required. Install Git, a C/C++ build toolchain, CMake and
OpenGL/EGL system libraries using your distribution's package manager. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) to create the Python
environment and install the pinned inventory.

For Ubuntu, install the basic system dependencies and uv first:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ca-certificates build-essential cmake libgl1 libegl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
nvidia-smi
```

`nvidia-smi` must show the NVIDIA GPU and its compatible driver. If uv is already
available in your shell, its installation step can be skipped.

```bash
git clone https://github.com/whu-qqf/SuperLocomotion.git
cd SuperLocomotion
export HIKING_REPO_ROOT="$PWD"
export HIKING_ISAACLAB_ROOT="$PWD/external/IsaacLab"
export HIKING_INSTINCT_RL_ROOT="$PWD/external/instinct_rl"
export HIKING_NEWTON_ROOT="$PWD/external/newton"
mkdir -p external

uv venv --python 3.12 .venv-hiking
source .venv-hiking/bin/activate
```

The following source revisions are the ones used on the training server.
Their availability in the public upstream repositories was checked when this
guide was written; floating branch heads may be incompatible.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$HIKING_ISAACLAB_ROOT"
git -C "$HIKING_ISAACLAB_ROOT" checkout b03345bbf2c9a54eb03f9bd823ae19c0ca3a014e

git clone https://github.com/project-instinct/instinct_rl.git "$HIKING_INSTINCT_RL_ROOT"
git -C "$HIKING_INSTINCT_RL_ROOT" checkout f476f7b402430ea21601c4fb9e3493d4daf60123

git clone https://github.com/newton-physics/newton.git "$HIKING_NEWTON_ROOT"
git -C "$HIKING_NEWTON_ROOT" checkout 71eeba16a717d4dcf0b9dd7056a7b6f863054dc3
```

## 2. Install the pinned runtime

Run the installation from the repository root with all four `HIKING_*_ROOT`
variables above still exported:

```bash
uv pip install --python .venv-hiking/bin/python --no-deps \
  -r requirements/hiking-runtime.txt \
  --default-index https://pypi.org/simple \
  --index https://pypi.nvidia.com \
  --index https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match

uv pip install --python .venv-hiking/bin/python --no-deps -e source/instinctlab
source tools/activate_hiking.sh
```

The inventory includes the resolved Isaac Sim 6.2.0.0 and CUDA 13.0 PyTorch
packages, plus editable source packages. This is a large installation, not a
small standalone Python dependency set. `--no-deps` deliberately consumes that
resolved inventory: the validated simulator uses dependency overrides that
vanilla pip resolution does not preserve. Do not mix this environment with
Isaac Lab 2.x / Isaac Sim 4.x or upgrade pinned dependencies independently.

The inventory comes from the training server. A complete installation on a
fresh machine has not yet been independently validated. The portable activation
script, CLI commands and short training workflow were checked with the existing
validated runtime. If installation fails on another machine, retain the package
installation error rather than replacing pinned versions arbitrarily.

For every new terminal, from the repository root:

```bash
source tools/activate_hiking.sh
export CUDA_VISIBLE_DEVICES=0
```

The activation script defaults to `.venv-hiking` and source checkouts under
`external/`. To use existing installations elsewhere, set `HIKING_VENV`,
`HIKING_ISAACLAB_ROOT` and `HIKING_INSTINCT_RL_ROOT` before sourcing it.
It configures the source search paths needed by the simulator bootstrap.
It does not depend on the original server's `logs/hiking_environment/` directory.

## 3. Check a small run first

```bash
python scripts/instinct_rl/train_hiking_native_latent.py --help

python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 8 --device cuda:0 --viz none \
  --tgi_aligned --diagnostic --policy_microbatch 8 --precision bf16 \
  --max_iterations 1 --output logs/hiking/diagnostic
```

Choose an output directory that does not exist yet. The diagnostic collects real
transitions, updates the world model/actor/critic, exports a controller and
verifies that the export matches. Success produces `checks.json` with
`controller_roundtrip: true` and `results.json` with `final_iteration: 1`.
It checks the dataflow; one iteration does not produce a walking policy.
The first scene setup also builds terrain and initializes physics kernels.

## 4. Start the full training run

```bash
python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 2048 --device cuda:0 --seed 1652 --viz none \
  --tgi_aligned --policy_microbatch 2048 --precision bf16 \
  --compile_mode eager --max_iterations 14000 \
  --output logs/hiking/train_2048
```

This is the current training recipe, including 300 initial world-model warmup
iterations. The full configuration needs substantially more GPU memory than the
8-environment diagnostic; a smaller diagnostic is not a memory qualification
for the full run. Record the effective configuration when changing batch sizes
or environment count.

## 5. Evaluate an existing trained model

Obtain a `model_N.pt` weights checkpoint separately, or use one produced by your
training run. Place it under `checkpoints/`; this directory is ignored by Git.
For the original final run, the required evaluation file is `model_14000.pt`.
`controller.pt` is a deployment export and cannot replace it in this evaluator.

```bash
python scripts/instinct_rl/evaluate_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --checkpoint checkpoints/model_14000.pt \
  --num_envs 32 --device cuda:0 --viz none \
  --terrain native_hiking --steps 1000 \
  --output logs/hiking/evaluation
```

To record a stair video, use a new output directory:

```bash
python scripts/instinct_rl/evaluate_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --checkpoint checkpoints/model_14000.pt \
  --num_envs 1 --device cuda:0 --viz newton_gl \
  --terrain native_hiking --terrain_type pyramid_stairs_inv --terrain_level 1 \
  --steps 1000 --video --video_length 1000 \
  --output logs/hiking/stairs_video
```

`PYGLET_HEADLESS=true` allows server video capture without a remote desktop.
To resume training instead, use the complete `latest.pt` checkpoint, including
replay/optimizer state. The [Hiking guide](docs/hiking.md) explains these artifacts,
the architecture, saved configuration and result interpretation.

## 6. Resume training

Use the complete `latest.pt` checkpoint, including replay and optimizer state.
The output directory must be new; `--max_iterations` is the number of additional
iterations for a resumed run.

```bash
python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 2048 --device cuda:0 --viz none --tgi_aligned \
  --resume --checkpoint /path/to/latest.pt \
  --max_iterations 1000 --output logs/hiking/continuation
```

| Artifact | Purpose |
| --- | --- |
| `model_N.pt` | Weights-only checkpoint for evaluation |
| `controller.pt` | Depth encoder and actor for deployment |
| `latest.pt` | Complete checkpoint for continuing training |
| `metrics.jsonl` | Training metrics, including episode duration and SGS statistics |

Logs, dependency checkouts, virtual environments and checkpoints are ignored by
Git. They remain local and are not downloaded when cloning this repository.

## Attribution and license

The code builds on [InstinctLab](https://github.com/project-instinct/InstinctLab)
and [SuperMimic](https://github.com/HybridRobotics/supermimic), with design ideas
from Terrain-Grounded Imagination. It is an experimental implementation, not a
complete reproduction of TGI. Gait quality and difficult-terrain success remain
areas for further work.

Original attribution and third-party notices are retained. The inherited
[CC BY-NC 4.0 license](LICENSE) applies to the project; individual components
retain their own license notices.
