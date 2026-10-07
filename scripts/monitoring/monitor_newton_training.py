#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Incrementally read TensorBoard TFRecords, persisting byte offsets between calls.

Run on the training node with its Isaac Lab Python. First call scans history;
subsequent calls read only appended complete records. Partial writes are retried.
Use one state file per run; JSON values carry the actual zero-based event step.
"""

import argparse
import json
import struct
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event

TAGS = {
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_std",
    "Policy/std_min",
    "Policy/std_max",
    "Audit/raw_action_abs_gt3_fraction",
    "Numerics/invalid_world_resets",
    "Numerics/invalid_world_100_update_rate_per_million",
    "Numerics/invalid_world_100_update_count",
    "Curriculum/arm_intensity",
    "Curriculum/reset_intensity",
    "Perf/collection_time",
    "Perf/learning_time",
}
TAGS.update(
    f"Robustness/all/all/{axis}_error_{unit}_mae"
    for axis, unit in (("vx", "mps"), ("vy", "mps"), ("yaw_rate", "radps"))
)


def scan(directory, state):
    read_bytes = 0
    state.setdefault("rolling_gate_breaches", {})
    for path in sorted(directory.glob("events.out.tfevents.*")):
        key = str(path.resolve())
        stat = path.stat()
        cursor = state["files"].setdefault(key, {"offset": 0, "inode": stat.st_ino})
        if cursor["inode"] != stat.st_ino or stat.st_size < cursor["offset"]:
            raise RuntimeError(f"Event file replaced/truncated: {path}; use a fresh state file")
        with path.open("rb") as f:
            f.seek(cursor["offset"])
            while True:
                start = f.tell()
                header = f.read(12)
                if len(header) != 12:
                    break
                size = struct.unpack("<Q", header[:8])[0]
                if size > 64 * 1024 * 1024:
                    raise ValueError(f"Invalid TFRecord length at {path}:{start}")
                data, footer = f.read(size), f.read(4)
                if len(data) != size or len(footer) != 4:
                    break
                event = Event.FromString(data)
                for value in event.summary.value:
                    if value.tag not in TAGS or not value.HasField("simple_value"):
                        continue
                    state["step"] = max(state["step"], event.step)
                    state["latest"][value.tag] = {"step": event.step, "value": value.simple_value}
                    if value.tag == "Numerics/invalid_world_resets":
                        state["invalid"][str(event.step)] = value.simple_value
                        state["invalid"] = {k: v for k, v in state["invalid"].items() if int(k) > event.step - 100}
                        if len(state["invalid"]) == 100 and event.step >= 500:
                            denominator = 100 * state["identity"][1] * state["identity"][2]
                            rate = sum(state["invalid"].values()) * 1e6 / denominator
                            if rate > 10:
                                state["rolling_gate_breaches"][str(event.step)] = rate
                    if value.tag == "Numerics/invalid_world_100_update_rate_per_million" and event.step >= 500:
                        if value.simple_value > 10:
                            state["gate_breaches"][str(event.step)] = value.simple_value
                cursor["offset"] = f.tell()
                read_bytes += f.tell() - start
    state["invalid"] = {k: v for k, v in state["invalid"].items() if int(k) > state["step"] - 100}
    return read_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps-per-update", type=int, default=24)
    a = p.parse_args()
    state = (
        json.loads(a.state.read_text())
        if a.state.exists()
        else dict(files={}, latest={}, invalid={}, gate_breaches={}, step=-1)
    )
    identity = [str(a.events.resolve()), a.num_envs, a.steps_per_update]
    if state.setdefault("identity", identity) != identity:
        raise ValueError("State belongs to a different run/protocol")
    read_bytes = scan(a.events, state)
    a.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.state.with_suffix(a.state.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(a.state)
    count = sum(state["invalid"].values())
    n = len(state["invalid"])
    output = dict(
        updates_completed=state["step"] + 1,
        bytes_read_this_call=read_bytes,
        latest=state["latest"],
        invalid_last_window=dict(
            updates=n,
            resets=count,
            env_steps=n * a.num_envs * a.steps_per_update,
            per_million=count * 1e6 / (n * a.num_envs * a.steps_per_update) if n else None,
            complete_100_updates=n == 100,
        ),
        aligned_gate_breaches=state["gate_breaches"],
        rolling_gate_breaches=state["rolling_gate_breaches"],
        note="Raw action saturation is diagnostic only. Process/log inspection is required to rule out a crash.",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
