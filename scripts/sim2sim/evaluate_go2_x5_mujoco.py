#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""CPU deployment probe, gated by identical-state Isaac Lab observation packets.

The read-only MJCF supplies inertias/contact geometry/passive joint properties.
Runtime overrides supply the training timestep, motor effort limits and PD law.
Optional arm motion reuses the training BoundedArmMotion kernel at final intensity.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

LEGS = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
ARMS = [f"joint{i}" for i in range(1, 7)]
TERMS = dict(
    base_lin_vel=3,
    base_ang_vel=3,
    projected_gravity=3,
    velocity_commands=6,
    actions=12,
    joint_pos=12,
    joint_vel=12,
    arm_joint_pos=6,
    arm_joint_vel=6,
)
DEFAULT = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5] + [0.0] * 6)
KP = np.array([25.0] * 12 + [50.0, 50.0, 80.0, 30.0, 20.0, 20.0])
KD = np.array([0.5] * 12 + [5.0, 10.0, 10.0, 2.5, 2.0, 1.0])
LIMIT = np.array([33.5] * 12 + [20.0] * 6)
SCALE = np.array([0.125, 0.25, 0.25] * 4)
SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "vx03": (0.3, 0.0, 0.0),
    "vx06": (0.6, 0.0, 0.0),
    "vy03": (0.0, 0.3, 0.0),
    "vy-03": (0.0, -0.3, 0.0),
    "yaw08": (0.0, 0.0, 0.8),
    "yaw-08": (0.0, 0.0, -0.8),
}


class Plant:
    def __init__(self, xml):
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.option.timestep = 0.005
        # The robot-only deployment XML has no floor.
        spec.worldbody.add_geom(
            name="stage_d_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100.0, 100.0, 0.1], friction=[0.4, 0.02, 0.01]
        )
        self.model = m = spec.compile()
        self.names = LEGS + ["x5_" + n for n in ARMS]
        self.jids = np.array([m.joint(n).id for n in self.names])
        self.qids, self.vids = m.jnt_qposadr[self.jids], m.jnt_dofadr[self.jids]
        self.aids = np.array([np.flatnonzero(m.actuator_trnid[:, 0] == j).item() for j in self.jids])
        assert np.all(m.actuator_gaintype[self.aids] == 0) and np.all(m.actuator_biastype[self.aids] == 0)
        assert np.all(m.actuator_gainprm[self.aids, 0] == 1.0)
        self.original_ranges = m.actuator_ctrlrange[self.aids].copy()
        m.actuator_ctrlrange[self.aids] = np.stack([-LIMIT, LIMIT], axis=-1)
        self.base = m.body("base_link").id
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def kinematics(self, d):
        mujoco.mj_jacBodyCom(self.model, d, self.jp, self.jr, self.base)
        rotation = d.xmat[self.base].reshape(3, 3)
        return rotation.T @ (self.jp @ d.qvel), rotation.T @ (self.jr @ d.qvel), rotation.T @ [0.0, 0.0, -1.0]

    def observe(self, d, command, previous, default=DEFAULT):
        lin, ang, gravity = self.kinematics(d)
        q, v = d.qpos[self.qids] - default, d.qvel[self.vids]
        return np.concatenate(
            [
                2 * lin,
                0.25 * ang,
                gravity,
                command,
                previous,
                q[:12],
                0.05 * v[:12],
                np.clip(q[12:], -3 * 3.14159, 3 * 3.14159),
                0.1 * np.clip(v[12:], -50.0, 50.0),
            ]
        ).astype(np.float32)

    def write_packet(self, d, packet, row, indices):
        mujoco.mj_resetData(self.model, d)
        pose, vel = packet["root_pose"][row], packet["root_velocity"][row]
        d.qpos[:3] = pose[:3]
        d.qpos[3:7] = pose[[6, 3, 4, 5]]  # Lab xyzw -> MuJoCo wxyz
        d.qpos[self.qids] = packet["joint_pos"][row, indices]
        mujoco.mj_forward(self.model, d)
        rotation = d.xmat[self.base].reshape(3, 3)
        # Lab root velocity is COM velocity; freejoint translation is link-origin velocity.
        offset = rotation @ self.model.body_ipos[self.base]
        d.qvel[:3] = vel[:3] - np.cross(vel[3:], offset)
        d.qvel[3:6] = rotation.T @ vel[3:]
        d.qvel[self.vids] = packet["joint_vel"][row, indices]
        mujoco.mj_forward(self.model, d)


def equivalence(plant, dirs, out):
    rows, sources = [], []
    for folder in dirs:
        manifest = json.loads((folder / "manifest.json").read_text())
        indices = [manifest["joint_names"].index(n) for n in LEGS + ARMS]
        for path in sorted(folder.glob("seed_*_obs_packets.npz")):
            packet = np.load(path)
            maxima = np.zeros(len(TERMS))
            d = mujoco.MjData(plant.model)
            for row in range(len(packet["observation"])):
                plant.write_packet(d, packet, row, indices)
                obs = plant.observe(
                    d, packet["command"][row], packet["previous_action"][row], packet["default_joint_pos"][row, indices]
                )
                start = 0
                for t, width in enumerate(TERMS.values()):
                    maxima[t] = max(
                        maxima[t],
                        np.max(np.abs(obs[start : start + width] - packet["observation"][row, start : start + width])),
                    )
                    start += width
            for term, err in zip(TERMS, maxima):
                rows.append(
                    dict(
                        source=str(path),
                        term=term,
                        samples=len(packet["observation"]),
                        max_abs_diff=float(err),
                        threshold=1e-4,
                        passed=bool(err <= 1e-4),
                    )
                )
            sources.append(str(path))
    assert rows, "No Isaac Lab rollout observation packets"
    write_csv(out / "obs_equivalence.csv", rows)
    receipt = dict(
        passed=all(r["passed"] for r in rows),
        sources=sources,
        mujoco_version=mujoco.__version__,
        max_abs_diff=max(r["max_abs_diff"] for r in rows),
    )
    (out / "obs_equivalence.json").write_text(json.dumps(receipt, indent=2))
    assert receipt["passed"], "STOP: identical-state observation difference exceeds 1e-4; policy evaluation forbidden"
    return receipt


def write_csv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def arm_controller(recipe, replicas, seed):
    import torch

    path = (
        Path(__file__).resolve().parents[2]
        / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/shared/robustness_math.py"
    )
    spec = importlib.util.spec_from_file_location("arm_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = dict(recipe)
    args["lower"] = torch.tensor(args["lower"]).repeat(replicas, 1)
    args["upper"] = torch.tensor(args["upper"]).repeat(replicas, 1)
    torch.manual_seed(seed)
    controller = module.BoundedArmMotion(**args)
    controller.reset(torch.arange(replicas), torch.tensor(DEFAULT[12:], dtype=torch.float32).repeat(replicas, 1))
    return controller


def evaluate(plant, bundle, label, replicas, recipe=None):
    import torch

    torch.set_num_threads(1)
    policy = torch.jit.load(str(bundle / "policy.pt"), map_location="cpu").eval()
    config = next(iter(yaml.safe_load((bundle / "config.yaml").read_text()).values()))
    lower = np.asarray(config["clip_actions_lower"])
    upper = np.asarray(config["clip_actions_upper"])
    assert lower.shape == upper.shape == (12,) and np.all(lower < upper)
    result = []
    for name, velocity in SCENARIOS.items():
        for seed in (42, 73):
            rng = np.random.default_rng(seed)
            data = [mujoco.MjData(plant.model) for _ in range(replicas)]
            # X5 task overrides the asset's five-tick delay to two ticks.
            lags = rng.integers(0, 3, (replicas, 2))
            cmd = np.array([*velocity, 0.33, 0.0, 0.0])
            motion = arm_controller(recipe, replicas, seed) if recipe else None
            previous = np.zeros((replicas, 12))
            history = np.tile(DEFAULT, (3, replicas, 1))
            alive = np.ones(replicas, dtype=bool)
            falls = np.zeros(replicas, dtype=bool)
            for d in data:
                d.qpos[:3] = [0.0, 0.0, 0.36]
                d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                d.qpos[plant.qids] = DEFAULT
                mujoco.mj_forward(plant.model, d)
            errors, drift, velocity_sq, clips, arm_errors = [], [], [], [], []
            tick = 0
            for step in range(1000):
                obs = np.stack([plant.observe(d, cmd, previous[i]) for i, d in enumerate(data)])
                with torch.inference_mode():
                    raw = policy(torch.from_numpy(obs)).numpy()
                assert np.isfinite(raw).all()
                previous = np.clip(raw, lower, upper)
                target = np.tile(DEFAULT, (replicas, 1))
                target[:, :12] += previous * SCALE
                if motion is not None:
                    actual = torch.tensor(np.stack([d.qpos[plant.qids[12:]] for d in data]), dtype=torch.float32)
                    target[:, 12:] = motion.step(1.0, actual).numpy()
                for _ in range(4):
                    if tick == 0:
                        # Isaac DelayBuffer fills a reset history with its first
                        # input; it does not inject default targets at startup.
                        history[:] = target
                    history[tick % 3] = target
                    for i, d in enumerate(data):
                        if not alive[i]:
                            continue
                        qtarget = np.r_[
                            history[(tick - lags[i, 0]) % 3, i, :12], history[(tick - lags[i, 1]) % 3, i, 12:]
                        ]
                        torque = np.clip(KP * (qtarget - d.qpos[plant.qids]) - KD * d.qvel[plant.vids], -LIMIT, LIMIT)
                        d.ctrl[plant.aids] = torque
                        mujoco.mj_step(plant.model, d)
                    tick += 1
                for i, d in enumerate(data):
                    if not alive[i]:
                        continue
                    # mj_step integrates qpos/qvel after computing body frames.
                    # Refresh those frames before sampling the policy state.
                    mujoco.mj_forward(plant.model, d)
                    lin, ang, gravity = plant.kinematics(d)
                    fallen = d.qpos[2] < 0.12 or -gravity[2] < np.cos(np.deg2rad(75))
                    falls[i] |= fallen
                    if fallen:
                        alive[i] = False
                        continue
                    if step >= 199:  # t >= 4 seconds, transient excluded
                        errors.append(np.abs(np.r_[lin[:2], ang[2]] - velocity))
                        drift.append([np.linalg.norm(lin[:2]), abs(ang[2])])
                        velocity_sq.append(d.qvel[plant.vids] ** 2)
                        clips.append((raw[i] < lower) | (raw[i] > upper))
                        arm_errors.append(np.abs(target[i, 12:] - d.qpos[plant.qids[12:]]))
            err = np.mean(errors, axis=0) if errors else np.full(3, np.nan)
            ds = np.mean(drift, axis=0) if drift else [np.nan, np.nan]
            vs = np.mean(velocity_sq, axis=0) if velocity_sq else np.full(18, np.nan)
            result.append(
                dict(
                    policy=label,
                    scenario=name,
                    seed=seed,
                    replicas=replicas,
                    seconds=20.0,
                    falls=int(falls.sum()),
                    valid_post4s_samples=len(errors),
                    mae_vx_m_s=err[0],
                    mae_vy_m_s=err[1],
                    mae_wz_rad_s=err[2],
                    stand_drift_m_s=ds[0] if name == "stand" else "",
                    stand_drift_rad_s=ds[1] if name == "stand" else "",
                    stand_leg_qvel_rms_rad_s=np.sqrt(vs[:12].mean()) if name == "stand" else "",
                    stand_arm_qvel_rms_rad_s=np.sqrt(vs[12:].mean()) if name == "stand" else "",
                    raw_action_clip_fraction=float(np.mean(clips)) if clips else np.nan,
                    scalar_action_denominator=len(clips) * 12,
                    arm_target_mae_rad=float(np.mean(arm_errors)) if arm_errors else np.nan,
                    arm_scalar_samples=len(arm_errors) * 6,
                    arm_motion_intensity=1.0 if recipe else 0.0,
                    extension_replicas=int((motion.mode == 4).sum()) if motion else 0,
                )
            )
            print(json.dumps(result[-1]), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--isaac-packets", type=Path, nargs="+", required=True)
    p.add_argument("--bundle", type=Path, action="append", default=[])
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--arm-recipe", type=Path, help="Training BoundedArmMotion kwargs and actual soft limits JSON")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(a.arm_recipe.read_text()) if a.arm_recipe else None
    plant = Plant(a.xml)
    equivalence(plant, a.isaac_packets, a.output)
    rows = []
    for bundle in a.bundle:
        rows.extend(evaluate(plant, bundle, bundle.name, a.replicas, recipe))
        write_csv(a.output / "mujoco_eval.csv", rows)


if __name__ == "__main__":
    main()
