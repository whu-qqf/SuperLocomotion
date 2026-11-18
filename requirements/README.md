# Hiking runtime inventory

`hiking-runtime.txt` records the resolved simulator environment prepared before
the experiments. It uses Isaac Lab's dependency overrides and is not intended
for unconstrained pip dependency resolution. Additional package requirements
for this repository live in `source/instinctlab/setup.py`.

Set these absolute paths in the shell before using the inventory:

- `HIKING_REPO_ROOT`: this repository.
- `HIKING_ISAACLAB_ROOT`: Isaac Lab at the pinned revision.
- `HIKING_NEWTON_ROOT`: Newton at the pinned revision.
- `HIKING_INSTINCT_RL_ROOT`: Instinct-RL at the pinned revision.

In a prepared Python 3.12 environment, the original installation consumed the
complete resolved list with `uv pip install --no-deps -r requirements/hiking-runtime.txt`,
using PyPI, NVIDIA's Python index and PyTorch's CUDA 13.0 wheel index. Consult the
simulator installation instructions before reproducing that environment.
Use an editable installation of this repository afterward. This inventory was
made portable during cleanup; installation on a fresh machine has not been tested.
