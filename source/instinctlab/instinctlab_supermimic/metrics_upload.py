"""Stream native JSONL metrics to SwanLab from an independent Python environment."""

import argparse
import json
import math
import time
from pathlib import Path


def scalar_metrics(record, prefix=""):
    result = {}
    for name, value in record.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            result.update(scalar_metrics(value, key))
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite training metric: {key}")
            result[key] = int(value) if isinstance(value, bool) else value
    return result


def format_iteration(record, target, num_envs=None, compact=False):
    """Render real episode statistics separately from imagined policy diagnostics."""
    if compact:
        selected = {
            "model/physical",
            "model/contact",
            "model/latent",
            "model/reconstruction_l1",
            "policy/actor_grad_norm",
            "policy/final_survival",
            "collection/episode/mean_episode_length",
        }
        scalars = scalar_metrics(record)
        values = [
            f"{key}={value:.4g}"
            for key, value in scalars.items()
            if key in selected or (key.startswith("evaluation/holdout/") and key.endswith("root_rmse_m"))
        ]
        stage = "world_model" if record.get("model_only") else "policy"
        return f"Learning iteration {record['iteration']}/{target} [{stage}] " + "  ".join(values)
    lines = ["#" * 80, f"Learning iteration {record['iteration']}/{target}", ""]
    performance = record.get("performance")
    if performance:
        lines.append(
            f"{'Computation:':>48} {performance['transitions_per_second']:.0f} steps/s "
            f"(collection: {performance['collection_seconds']:.3f}s, "
            f"learning: {performance['learning_seconds']:.3f}s)"
        )
    episode = record.get("collection", {}).get("episode")
    if episode is None:
        lines.append(f"{'Real episode statistics:':>48} not collected in this earlier segment")
    elif episode["mean_task_reward"] is None:
        lines.append(f"{'Real episode statistics:':>48} waiting for completed episodes")
    names = {
        "policy/critic_loss": "Value function loss",
        "policy/actor_loss": "Actor objective loss (BPTT)",
        "policy/task_reward": "Imagined task reward",
        "policy/style_reward": "Imagined style reward",
        "policy/final_survival": "Imagined final survival",
        "physical/physical": "Dynamics prediction loss",
        "perception/depth": "Depth reconstruction loss",
        "collection/episode/mean_task_reward": "Mean task episode reward (last 100)",
        "collection/episode/mean_episode_length": "Mean episode length (last 100)",
        "performance/collection_action_noise_std": "Collection action noise std",
    }
    for key, value in scalar_metrics(record).items():
        if key == "iteration" or (key.startswith("performance/") and key not in names):
            continue
        name = names.get(key, key.removeprefix("collection/episode/metrics/"))
        lines.append(f"{name + ':':>48} {value:.4f}")
    lines.append("-" * 80)
    total = performance["total_timesteps"] if performance else None
    if total is None and num_envs is not None:
        total = record["collection"]["real_controls"] * num_envs
    if total is not None:
        lines.append(f"{'Total timesteps:':>48} {total}")
    if performance:
        for label, key in (
            ("Iteration time", "iteration_seconds"),
            ("Total time (current process)", "session_total_seconds"),
            ("ETA", "eta_seconds"),
        ):
            lines.append(f"{label + ':':>48} {performance[key]:.2f}s")
    return "\n".join(lines)


def follow_metrics(path, alive, poll_seconds=1):
    """Wait for initialization, preserve partial writes, and drain after producer exit."""
    path = Path(path)
    while not path.exists():
        if not alive():
            raise RuntimeError("Training ended before creating metrics.jsonl.")
        time.sleep(poll_seconds)
    with path.open() as stream:
        while True:
            running = alive()
            position = stream.tell()
            line = stream.readline()
            if line.endswith("\n"):
                yield json.loads(line)
            elif not running:
                if line:
                    raise ValueError("Training ended with an incomplete metrics record.")
                return
            else:
                stream.seek(position)
                time.sleep(poll_seconds)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume the existing cloud run and backfill text logs.")
    parser.add_argument("--compact", action="store_true", help="Print key metrics only; upload all scalar curves.")
    args = parser.parse_args()
    import swanlab

    identity = process_identity(args.pid)
    if identity is None:
        raise RuntimeError("Training process is not alive.")
    config = json.loads(args.config.read_text())
    # The trainer creates its output with exist_ok=False. Never create it first.
    while not args.output.is_dir():
        if process_identity(args.pid) != identity:
            raise RuntimeError("Training ended before creating its output directory.")
        time.sleep(1)
    status_path = args.output / "swanlab.json"
    old_status = json.loads(status_path.read_text()) if args.resume else None
    if status_path.exists() and not args.resume:
        raise ValueError("This output already has a SwanLab run; use --resume.")
    resume_step = old_status["last_logged_iteration"] if old_status else config["start_iteration"]
    run = swanlab.init(
        project=args.project,
        workspace=args.workspace,
        experiment_name=args.experiment,
        config=config,
        logdir=str(args.output / "swanlog"),
        mode="cloud",
        public=False,
        id=old_status["run_id"] if old_status else None,
        resume="must" if old_status else "never",
    )
    status = dict(
        run_id=run.id,
        url=run.url if hasattr(run, "url") else swanlab.get_url(),
        mode="cloud",
        logged_records=resume_step - config["start_iteration"],
        last_logged_iteration=resume_step,
        text_records=0,
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print("SWANLAB_RUN", json.dumps(status), flush=True)
    previous = config["start_iteration"]
    for record in follow_metrics(args.output / "metrics.jsonl", lambda: process_identity(args.pid) == identity):
        step = record["iteration"]
        if step != previous + 1:
            raise ValueError(f"Training metric sequence differs: {step} after {previous}.")
        scalars = scalar_metrics(record)
        scalars.update(record.get("collection", {}).get("episode", {}).get("metrics", {}))
        if step > resume_step:
            swanlab.log(scalars, step=step)
            status.update(last_logged_iteration=step, logged_records=step - config["start_iteration"])
        print(
            format_iteration(record, config["target_iteration"], config.get("num_envs"), compact=args.compact),
            flush=True,
        )
        previous = step
        status.update(last_text_iteration=step, text_records=status["text_records"] + 1)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    results = json.loads((args.output / "results.json").read_text())
    if previous != config["target_iteration"] or results["iterations"] != config["additional_iterations"]:
        raise RuntimeError("Training stopped before the requested iteration budget.")
    swanlab.finish()
    status["finished"] = True
    status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
