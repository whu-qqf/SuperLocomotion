# Run after cloning

Cloning provides the source, G1 robot assets, terrain definitions and training
configuration. Install the simulator runtime separately. No motion-imitation
dataset is needed. Pretrained weights are not included, so a clone can train
from scratch but cannot immediately replay the final learned gait.

## 1. Prepare Linux and clone

Use Linux x86_64, an NVIDIA GPU, and a CUDA 13-compatible NVIDIA driver (580 or
newer). Python 3.12 is required. Install Git, a C/C++ build toolchain, CMake and
OpenGL/EGL system libraries using your distribution's package manager. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) to create the Python
environment and install the pinned inventory.

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
replay/optimizer state. The [Hiking guide](hiking.md) explains these artifacts,
the architecture, saved configuration and result interpretation.
