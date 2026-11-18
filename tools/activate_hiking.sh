#!/usr/bin/env bash
# Source this file after installing the pinned Hiking runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run: source tools/activate_hiking.sh" >&2
    exit 1
fi

_hiking_script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HIKING_REPO_ROOT="${HIKING_REPO_ROOT:-$_hiking_script_root}"
export HIKING_ISAACLAB_ROOT="${HIKING_ISAACLAB_ROOT:-$HIKING_REPO_ROOT/external/IsaacLab}"
export HIKING_INSTINCT_RL_ROOT="${HIKING_INSTINCT_RL_ROOT:-$HIKING_REPO_ROOT/external/instinct_rl}"
export HIKING_NEWTON_ROOT="${HIKING_NEWTON_ROOT:-$HIKING_REPO_ROOT/external/newton}"
_hiking_venv="${HIKING_VENV:-$HIKING_REPO_ROOT/.venv-hiking}"

if [[ ! -f "$_hiking_venv/bin/activate" || ! -d "$HIKING_ISAACLAB_ROOT/source/isaaclab" || ! -d "$HIKING_INSTINCT_RL_ROOT/instinct_rl" ]]; then
    echo "Hiking environment or source checkout missing. Follow docs/quickstart.md first." >&2
    unset _hiking_script_root _hiking_venv
    return 1
fi

source "$_hiking_venv/bin/activate"
export ISAACLAB_PATH="$HIKING_ISAACLAB_ROOT"
_hiking_source_paths="$HIKING_REPO_ROOT/source/instinctlab:$HIKING_INSTINCT_RL_ROOT"
for _hiking_source_root in "$HIKING_ISAACLAB_ROOT"/source/*; do
    case "$_hiking_source_root" in
        */isaaclab_mimic|*/isaaclab_teleop) continue ;;
    esac
    if [[ -f "$_hiking_source_root/pyproject.toml" ]]; then
        _hiking_source_paths+=":$_hiking_source_root"
    fi
done
export PYTHONPATH="$_hiking_source_paths${PYTHONPATH:+:$PYTHONPATH}"
export PYGLET_HEADLESS="${PYGLET_HEADLESS:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OMNI_KIT_ALLOW_ROOT=1
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_ACCEPT_EULA=Y
export ACCEPT_EULA=Y
unset _hiking_script_root _hiking_venv _hiking_source_paths _hiking_source_root
