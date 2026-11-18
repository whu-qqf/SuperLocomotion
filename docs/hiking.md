# Hiking SuperMimic

## Current workflow

This branch trains Unitree G1 terrain locomotion in the native InstinctLab Hiking
scene. Training uses a learned physical world model and differentiable imagined
rollouts. The world model queries privileged terrain; the deployed depth encoder
and actor use only proprioceptive history and depth. There is no foot-placement
model or explicit imitation objective in the current training configuration.

The main entrypoints are:

| Purpose | Entry |
| --- | --- |
| Train or resume | `scripts/instinct_rl/train_hiking_native_latent.py` |
| Evaluate or record video | `scripts/instinct_rl/evaluate_hiking_native_latent.py` |
| Audit native camera input | `scripts/instinct_rl/audit_hiking_native_perception.py` |
| Audit imagined camera | `scripts/instinct_rl/audit_hiking_imagined_camera.py` |
| Audit recorded transitions | `scripts/instinct_rl/audit_hiking_replay.py` |

Only the current Hiking task family is registered. Other upstream task families and their training tools are excluded.
Experimental teacher, dual-model, substep-fitting, distillation and unfinished
keyboard entrypoints were removed from the active tree. Shared infrastructure and checkpoint compatibility required by the current learner remain.

## Environment

For a new checkout, follow the complete [clone-to-training guide](quickstart.md).

Use Linux, Python 3.12 and CUDA 13-compatible NVIDIA drivers. The validated local
runtime uses the following source revisions:

| Dependency | Revision / version |
| --- | --- |
| Isaac Lab | `b03345bbf2c9a54eb03f9bd823ae19c0ca3a014e` / 41.0.2 |
| Instinct-RL | `f476f7b402430ea21601c4fb9e3493d4daf60123` / 1.0.3 |
| Newton | `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3` / 1.8.0.dev0 |
| Isaac Sim | 6.2.0.0 |
| PyTorch / TorchVision | 2.11.0+cu130 / 0.26.0+cu130 |
| Warp / NumPy | 1.17.0 / 2.3.1 |

Follow the [upstream simulator installation instructions](https://github.com/project-instinct/InstinctLab#installation), using these revisions. Install this repository with `python -m pip install -e source/instinctlab`.
The runtime uses Isaac Lab's dependency overrides; vanilla pip resolution alone
has not been validated. A CUDA device is required for simulation. The
`requirements/hiking-runtime.txt` file records the validated local package
versions and source paths through environment variables; it is an installation
inventory for an already prepared simulator environment.

After installing the pinned runtime and source checkouts, the portable activation
script sets the source search paths:

```bash
source tools/activate_hiking.sh
# Select an available GPU according to your server's allocation.
export CUDA_VISIBLE_DEVICES=0
```

The activation script is included in Git; virtual environments and dependency
checkouts are not. Its default layout is documented in the quickstart guide.
No source changes relative to the pinned Isaac Lab and Newton archives were
found when preparing this release.

## Train

Start a fresh 14,000-iteration run with the final configuration:

```bash
python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 2048 --device cuda:0 --seed 1652 --viz none \
  --tgi_aligned --policy_microbatch 2048 --precision bf16 \
  --compile_mode eager --max_iterations 14000 \
  --output logs/hiking/new_run
```

The output directory must not already exist. `--tgi_aligned` enables the
terrain-conditioned world model, imagined camera, aligned objectives and SGS
sampling. It uses 300 world-model warmup iterations, three collected control
steps per iteration, eight model updates, actor batch 2048 and horizon 32.
There are 20 critic updates per actor update. BF16 controls neural execution;
the adversarial motion prior and imitation CLI options are not supported.

Each iteration collects 6,144 transitions. The environment control period is
0.02 seconds. Iterations therefore cannot be compared directly with another
implementation using a different collection budget.

The recorded final run used the same configuration, with an execution-only
continuation at iteration 600. [The saved configuration](../configs/hiking/tgi_sgs_2048.json)
is a reference record, not a CLI input file.

Resume from a complete checkpoint into a new output directory:

```bash
python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 2048 --device cuda:0 --viz none --tgi_aligned \
  --resume --checkpoint /path/to/latest.pt \
  --max_iterations 1000 --output logs/hiking/continuation
```

For a resumed run, `--max_iterations` is the number of additional iterations.
`latest.pt` includes replay, optimizers and SGS state. `model_N.pt` contains
weights for evaluation. `controller.pt` exports the depth encoder and actor;
it is not a training-resume or evaluator checkpoint.

## Evaluate and record video

Evaluate the weights-only final checkpoint in the native terrain scene:

```bash
python scripts/instinct_rl/evaluate_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --checkpoint /path/to/model_14000.pt --num_envs 32 \
  --device cuda:0 --viz none --terrain native_hiking --steps 1000 \
  --output logs/hiking/evaluation
```

Use `--terrain_type pyramid_stairs_inv --terrain_level 1` to pin the scene to
a stair type and difficulty row. For a video, use `--num_envs 1 --viz newton_gl
--video --video_length 1000`; `PYGLET_HEADLESS=true` supports server recording.
The evaluator writes `results.json` and `trace.pt`, including reports by terrain
type and column. Surviving the evaluation budget is not the same as reaching a
target or completing a stair traversal.

The release includes [training statistics](results/hiking_14000.json), not an
independent final benchmark. Final-window mean episode duration is about 17 s
against a 20 s time limit. SGS success estimates depend on terrain, direction,
spawn configuration and adaptive sampling. Difficult stairs and stepping stones
remain challenging; gait quality is not established by survival alone.

## Repository and local artifacts

- `source/instinctlab/instinctlab/tasks/parkour/supermimic/`: native learner,
  perception, physics, replay, rewards and audits.
- `source/instinctlab/instinctlab_supermimic/`: neural networks, depth controller,
  physical state and sensor rotation utilities.
- `source/instinctlab/test/`: current regression tests.
- `configs/hiking/` and `docs/results/`: compact configuration and result records.
- `logs/`, `.venv*/`, `external/`, checkpoints and datasets: ignored local artifacts.

The completed server run remains at
`logs/hiking_supermimic/sgs_tgi_accel_20261117/hiking_train_2048/`.
Final controller and model SHA-256 hashes are recorded in the result inventory.
Model files are intentionally not uploaded with the code repository.
Historical code and documents can be recovered locally from
`logs/repository_cleanup_20261118/archive/` and the pre-cleanup source snapshot.
The runtime source under `logs/hiking_environment/` is still required locally;
do not delete it as if it were disposable training output.

## Checks and attribution

Run the current Hiking tests in the prepared runtime:

```bash
python -m pytest -q source/instinctlab/test/test_hiking* \
  source/instinctlab/test/test_native_metrics_upload.py
pre-commit run --all-files
```

The Hiking-only release passed 170 tests, with one optional test skipped;
all repository pre-commit checks passed. Both maintained CLI help commands and
CUDA inference with the final exported controller were also checked.
The inherited motion-imitation task family, its tests and its dataset requirements
are excluded from this repository. No LAFAN dataset is needed for the current task.

This work builds on [InstinctLab](https://github.com/project-instinct/InstinctLab),
[SuperMimic](https://github.com/HybridRobotics/supermimic) and TGI design ideas.
It does not reproduce the complete TGI system. Original licenses, contributor
notices and vendored third-party license files are retained. The project remains
subject to the upstream [license](../LICENSE).
