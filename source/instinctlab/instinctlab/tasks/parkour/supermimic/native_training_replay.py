"""Real native transitions and tick-aligned starts for the dual-model runner.

Physical targets are copied from the pre-reset recorder. Actor starts retain
the actual noisy observation histories, not histories regenerated from F.
Only P replay owns images and episode prefixes; no old P state is cached here.
"""

import torch
from collections import deque

from instinctlab_perception.replay import reconstruct_prefixes

from .native_imagination import NativeImaginationStart
from .native_task import AMP_WIDTHS, POLICY_WIDTHS


def history_frames(flat, widths, length):
    """Undo native term-major flattening without changing noise or reset fills."""
    if flat.ndim != 2 or flat.shape[-1] != sum(widths) * length:
        raise ValueError("Native history does not match the observation layout.")
    return torch.cat(
        [
            term.reshape(len(flat), length, width)
            for term, width in zip(flat.split([width * length for width in widths], -1), widths)
        ],
        -1,
    )


def copy_cpu(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().clone()
    if isinstance(tree, dict):
        return {name: copy_cpu(value) for name, value in tree.items()}
    if isinstance(tree, list):
        return [copy_cpu(value) for value in tree]
    raise TypeError(f"Unsupported native replay value: {type(tree)}")


def select_rows(tree, indices):
    if isinstance(tree, torch.Tensor):
        return tree[indices]
    if isinstance(tree, dict):
        return {name: select_rows(value, indices) for name, value in tree.items()}
    return [select_rows(value, indices) for value in tree]


def stack_rows(trees, device):
    first = trees[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(trees).to(device)
    if isinstance(first, dict):
        return {name: stack_rows([tree[name] for tree in trees], device) for name in first}
    return [stack_rows([tree[index] for tree in trees], device) for index in range(len(first))]


class NativeTrainingReplay:
    def __init__(self, num_envs, capacity_steps=512, action_steps=5):
        if min(num_envs, capacity_steps, action_steps) < 1:
            raise ValueError("Native replay dimensions must be positive.")
        self.num_envs, self.capacity_steps, self.action_steps = num_envs, capacity_steps, action_steps
        self.rows = deque(maxlen=capacity_steps)
        self.starts = deque()
        self.controls = 0
        self.segment = 0

    def begin_collection_segment(self):
        """A fresh simulator reset on resume is an explicit replay boundary."""
        self.segment += 1

    def add_transition(self, transition, context, episode_id, contacts, auxiliary, learning=None, task_reward=None):
        if len(transition.state) != self.num_envs or episode_id.shape != (self.num_envs,):
            raise ValueError("Native transition batch differs from replay environment count.")
        if task_reward is not None and (task_reward.shape != (self.num_envs,) or not torch.isfinite(task_reward).all()):
            raise ValueError("Native real task rewards must be finite per-environment scalars.")
        row = dict(
            state=transition.state,
            next_state=transition.next_state,
            actions=transition.action,
            context=context,
            contacts=contacts,
            auxiliary=auxiliary,
            terminated=transition.terminated,
            truncated=transition.truncated,
            episode_id=episode_id,
            control_step=transition.episode_step,
            segment=torch.full_like(episode_id, self.segment),
            task_reward=(
                torch.zeros(self.num_envs, device=transition.state.device) if task_reward is None else task_reward
            ),
            task_reward_valid=torch.full_like(episode_id, task_reward is not None, dtype=torch.bool),
        )
        if learning is not None:
            row["learning"] = learning
        row = copy_cpu(row)
        if self.rows and int(self.rows[-1]["segment"][0]) == self.segment:
            last = self.rows[-1]
            done = last["terminated"] | last["truncated"]
            same = row["episode_id"] == last["episode_id"]
            consecutive = row["control_step"] == last["control_step"] + 1
            reset = (row["episode_id"] > last["episode_id"]) & (row["control_step"] == 0)
            if not torch.where(done, reset, same & consecutive).all():
                raise ValueError("Native replay episode clocks disagree with recorded terminations.")
        self.rows.append(row)
        self.controls += 1
        oldest = self.controls - len(self.rows)
        while self.starts and self.starts[0]["collection_step"] < oldest:
            self.starts.popleft()

    def add_starts(self, tick, task, physical_context, policy_observation, amp_observation):
        """Call after adding the P tick, before submitting that state's action.

        The task context, previous action and sensor drifts are all pre-step.
        No extra observation-manager compute call is made here.
        """
        if tick is None:
            return
        if (tick.control_step.remainder(self.action_steps) != 0).any():
            raise ValueError("Actor starts must coincide with perception ticks.")
        env = task.env
        ids = tick.env
        sensor = env.scene["contact_forces"].data
        foot_ids = task.foot_contact_ids
        timers = torch.cat(
            (sensor.current_air_time.torch[:, foot_ids], sensor.current_contact_time.torch[:, foot_ids]), -1
        )
        values = dict(
            state=task.physics.recorder.state(),
            previous_action=env.action_manager.action,
            timers=timers,
            physical_context=physical_context,
            task_context=task.context(),
            command=task.command(),
            sensor_context=task.sensor_context(),
            policy_history=history_frames(policy_observation, POLICY_WIDTHS, 8),
            amp_history=history_frames(amp_observation, AMP_WIDTHS, 10),
        )
        values = copy_cpu(select_rows(values, ids))
        for index, env_id in enumerate(ids.tolist()):
            self.starts.append(
                dict(
                    collection_step=self.controls,
                    env=env_id,
                    episode_id=int(tick.episode_id[index]),
                    control_step=int(tick.control_step[index]),
                    values=select_rows(values, index),
                )
            )

    def sample_physics(self, batch_size, horizon, train_envs, device, generator=None, extra_fields=()):
        """Uniform real starts; include terminal targets, mask everything after.

        Explicit episode identity and step continuity prevent reset crossing.
        Partial tails are valid windows and have the same explicit mask.
        """
        if batch_size < 1 or horizon < 1 or not self.rows or not train_envs:
            raise ValueError("Physical replay sampling needs data and positive batch/horizon.")
        rows = list(self.rows)
        envs = tuple(train_envs)
        picks = torch.randint(len(rows) * len(envs), (batch_size,), generator=generator)
        starts = picks.div(len(envs), rounding_mode="floor")
        offsets = torch.arange(horizon)
        indices = starts[:, None] + offsets
        row_indices = indices.clamp_max(len(rows) - 1)
        env_indices = torch.tensor(envs)[picks.remainder(len(envs))]
        clocks = {
            name: torch.stack([row[name] for row in rows])
            for name in ("episode_id", "segment", "control_step", "terminated", "truncated")
        }
        selected = {name: value[row_indices, env_indices[:, None]] for name, value in clocks.items()}
        valid = indices < len(rows)
        for name in ("episode_id", "segment"):
            valid &= selected[name] == selected[name][:, :1]
        valid &= selected["control_step"] == selected["control_step"][:, :1] + offsets
        # The terminal transition itself is a valid target; only its successors stop.
        done = selected["terminated"] | selected["truncated"]
        valid[:, 1:] &= ~done[:, :-1]
        mask = valid.long().cumprod(1).bool()

        # Gather once per retained time row, rather than once per sampled timestep.
        # Images stay on CPU until selected: stacking the entire replay duplicates
        # several GB of camera histories and needlessly transfers unselected data.
        flat_rows = row_indices.flatten()
        order = flat_rows.argsort()
        counts = flat_rows.bincount(minlength=len(rows))
        groups = order.split(counts.tolist())
        flat_envs = env_indices[:, None].expand(-1, horizon).reshape(-1)

        def gather(values):
            first = values[0]
            output = first.new_empty((batch_size * horizon, *first.shape[1:]))
            for value, positions in zip(values, groups):
                if len(positions):
                    output[positions] = value[flat_envs[positions]]
            return output.reshape(batch_size, horizon, *first.shape[1:]).to(device)

        fields = ("state", "next_state", "actions", "context", "contacts", "auxiliary") + tuple(extra_fields)
        result = {name: gather([row[name] for row in rows]) for name in fields}
        result["mask"] = mask.to(device)
        return result

    def sample_real(self, batch_size, train_envs, device, generator=None, recent_steps=None):
        """True policy/reference windows and physical failure supervision."""
        if batch_size < 1 or not self.rows or not train_envs:
            raise ValueError("Real-data sampling needs a nonempty training pool.")
        rows = list(self.rows)
        if recent_steps is not None:
            if recent_steps < 1:
                raise ValueError("Recent real replay window must be positive.")
            rows = rows[-recent_steps:]
        envs = tuple(train_envs)
        picks = torch.randint(len(rows) * len(envs), (batch_size,), generator=generator)
        samples = []
        for pick in picks.tolist():
            row, env = divmod(pick, len(envs))
            if "learning" not in rows[row]:
                raise RuntimeError("Native replay row lacks real style/failure supervision.")
            samples.append(select_rows(rows[row], envs[env]))
        return stack_rows(samples, device)

    def sample_starts(self, perception_replay, model, batch_size, horizon, dt, generator=None):
        """Rebuild each selected posterior using current P, including its tick."""
        if batch_size < 1 or horizon < 1 or dt <= 0:
            raise ValueError("Actor replay sampling needs positive batch, horizon and dt.")
        candidates, prefixes = [], []
        for record in self.starts:
            if record["env"] not in perception_replay.train_envs:
                continue
            if float(record["values"]["command"]["time_left"]) <= horizon * dt:
                continue
            # Whole P episodes may be evicted before physical rows. Never
            # replace a missing prefix by an initial or cached hidden state.
            if record["episode_id"] not in perception_replay.episodes[record["env"]]:
                continue
            prefix = perception_replay.prefix(record["env"], record["episode_id"], record["control_step"])
            candidates.append(record)
            prefixes.append(prefix)
        if not candidates:
            raise RuntimeError("No tick-aligned native starts with complete P prefixes and command horizon.")
        picks = torch.randint(len(candidates), (batch_size,), generator=generator).tolist()
        device = next(model.parameters()).device
        values = stack_rows([candidates[index]["values"] for index in picks], device)
        memory = reconstruct_prefixes(model, [prefixes[index] for index in picks])
        return NativeImaginationStart(
            **values,
            perception_state=memory,
            control_step=torch.tensor([candidates[index]["control_step"] for index in picks], device=device),
        )

    def sample_value_windows(self, perception_replay, batch_size, horizon, envs, generator=None):
        """Measured rewards from a tick to a tick or a physical termination.

        Unknown legacy rewards, timeouts, reset/collection boundaries and
        missing P prefixes cannot become critic labels. Sampling a shuffled
        candidate list keeps the conditional distribution uniform without
        constructing every long window on each update.
        """
        if batch_size < 1 or horizon < 1 or horizon % self.action_steps:
            raise ValueError("Real value windows need a positive batch and tick-aligned horizon.")
        rows = list(self.rows)
        oldest = self.controls - len(rows)
        pool = set(envs)
        starts = [
            r
            for r in self.starts
            if r["env"] in pool
            and r["episode_id"] in perception_replay.episodes[r["env"]]
            and 0 <= r["collection_step"] - oldest < len(rows)
            and rows[r["collection_step"] - oldest]["task_reward_valid"][r["env"]]
        ]
        endpoints = {(r["collection_step"], r["env"]): r for r in self.starts}
        if not starts:
            raise RuntimeError("No measured real value windows; collect rewards and complete P prefixes first.")
        done = torch.stack([r["terminated"] | r["truncated"] for r in rows])
        selected = []
        for index in torch.randperm(len(starts), generator=generator).tolist():
            start = starts[index]
            first, env = start["collection_step"] - oldest, start["env"]
            available = min(horizon, len(rows) - first)
            stops = done[first : first + available, env].nonzero().flatten()
            length = int(stops[0]) + 1 if len(stops) else available
            last = rows[first + length - 1]
            terminal = bool(last["terminated"][env])
            if last["truncated"][env] or (not terminal and length != horizon):
                continue
            if (
                last["episode_id"][env] != start["episode_id"]
                or last["segment"][env] != rows[first]["segment"][env]
                or last["control_step"][env] != start["control_step"] + length - 1
            ):
                continue
            window = rows[first : first + length]
            if not all(r["task_reward_valid"][env] for r in window):
                continue
            end = None if terminal else endpoints.get((start["collection_step"] + horizon, env))
            if not terminal and (end is None or end["episode_id"] != start["episode_id"]):
                continue
            selected.append(
                dict(
                    start=start,
                    end=end,
                    task_reward=torch.stack([r["task_reward"][env] for r in window]),
                    amp_windows=torch.stack([r["learning"]["policy_window"][env] for r in window]),
                    terminal_state=last["next_state"][env] if terminal else None,
                    terminal_previous_amp=(
                        (
                            start["values"]["amp_history"]
                            if length == 1
                            else history_frames(window[-2]["learning"]["policy_window"][env : env + 1], AMP_WIDTHS, 10)[
                                0
                            ]
                        )
                        if terminal
                        else None
                    ),
                )
            )
            if len(selected) == batch_size:
                break
        if not selected:
            raise RuntimeError("No complete measured value windows; collect beyond the requested return horizon.")
        if len(selected) < batch_size:
            picks = torch.randint(len(selected), (batch_size,), generator=generator).tolist()
            selected = [selected[i] for i in picks]
        return selected

    def state_dict(self):
        return dict(
            config=(self.num_envs, self.capacity_steps, self.action_steps),
            rows=list(self.rows),
            starts=list(self.starts),
            controls=self.controls,
            segment=self.segment,
        )

    def load_state_dict(self, state):
        if state["config"] != self.state_dict()["config"]:
            raise ValueError("Incompatible native training replay checkpoint.")
        # Older checkpoints did not retain rewards. Their unknown returns
        # must never become zero-reward critic supervision on resume.
        rows = []
        for saved in state["rows"]:
            row = dict(saved)
            if "task_reward" not in row:
                row["task_reward"] = torch.zeros(self.num_envs)
                row["task_reward_valid"] = torch.zeros(self.num_envs, dtype=torch.bool)
            rows.append(row)
        self.rows = deque(rows, maxlen=self.capacity_steps)
        self.starts = deque(state["starts"])
        self.controls = state["controls"]
        self.segment = state["segment"]
