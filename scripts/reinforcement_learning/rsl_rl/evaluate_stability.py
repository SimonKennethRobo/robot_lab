"""Paired, bounded GO2-X5 checkpoint evaluation with terminal-state traces.

All checkpoints see the same seeded terrain/reset, command schedule, and pushes.
Each environment contributes its first rollout only; auto-reset samples are masked.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoints', nargs='+', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seeds', nargs='+', type=int, default=[42])
parser.add_argument('--replicas', type=int, default=16)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--task', default='RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import hashlib
import json
import numpy as np
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity_pose.robustness_runner import LocomotionOnPolicyRunner

SCENARIOS = ('arm_full_extension_eval', 'forward_03', 'forward_06', 'lateral_left', 'lateral_right', 'push')


def summarize(trace, replicas, dt):
    results = {}
    time = (np.arange(len(trace['valid'])) + 1) * dt
    for i, name in enumerate(SCENARIOS):
        sl = slice(i * replicas, (i + 1) * replicas)
        valid = trace['valid'][:, sl]
        measured = valid & (time[:, None] >= 4.0)
        reached = trace['arm_reached'][:, sl] & valid & ~trace['fallen'][:, sl]
        continuous = np.zeros(replicas)
        longest = continuous.copy()
        for mask in reached:
            continuous = np.where(mask, continuous + dt, 0)
            longest = np.maximum(longest, continuous)
        contact = trace['contact'][:, sl]
        losses = (contact[:-1] & ~contact[1:]) & (measured[:-1] & measured[1:])[..., None]
        speed = np.abs(trace['leg_velocity'][:, sl])
        rms = np.sqrt((speed ** 2).mean(-1))
        others = np.median(rms[..., [0, 2, 3]], axis=-1)
        ratio = (rms[..., 1] / np.maximum(others, 0.1))[measured & (others > 0.1)]
        arrival_mask = trace['arm_arrival'][:, sl] & valid
        arrival = trace['arm_arrival_time_s'][:, sl][arrival_mask]
        def avg(key):
            values = trace[key][:, sl][measured]
            return float(np.abs(values).mean()) if values.size else None
        results[name] = {
            'episodes': replicas, 'post4s_samples': int(measured.sum()),
            'failure_fraction': float((trace['failed'][:, sl] & valid).any(0).mean()),
            'fall_event_fraction': float((trace['fallen'][:, sl] & valid).any(0).mean()),
            'truncated_fraction': float((trace['truncated'][:, sl] & valid).any(0).mean()),
            'arrival_fraction': float(arrival_mask.any(0).mean()),
            'time_to_target_s_mean': float(arrival.mean()) if arrival.size else None,
            'upright_extension_hold_10s_fraction': float((longest >= 10 - 1e-5).mean()),
            'extension_target_error_rad_mean': avg('arm_error_rad'),
            'ee_forward_error_m_mean': avg('ee_forward_error_m'),
            'arm_torque_saturation_fraction': avg('arm_torque_saturation'),
            'leg_joint_speed_peak_radps_FL_FR_RL_RR': np.where(measured[..., None, None], speed, 0).max(axis=(0, 1, 3)).tolist(),
            'FR_over_median_other_legs_p95': float(np.quantile(ratio, .95)) if ratio.size else None,
            'contact_losses_FL_FR_RL_RR': losses.sum(axis=(0, 1)).tolist(),
            'four_feet_contact_fraction': float(contact.all(-1)[measured].mean()) if measured.any() else None,
            'velocity_error_mae_vx_vy_wz': np.abs(trace['velocity_error'][:, sl][measured]).mean(0).tolist() if measured.any() else None,
        }
    return results


def main():
    if args.steps < 1 or args.replicas < 1:
        raise ValueError('steps and replicas must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    n = len(SCENARIOS) * args.replicas
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n)
    cfg.seed = args.seeds[0]
    cfg.inference_mode = True
    cfg.log_dir = str(output / 'runtime')
    cfg.episode_length_s = args.steps * cfg.sim.dt * cfg.decimation + 5.0
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.scene.terrain.max_init_terrain_level = None
    r = cfg.robustness
    r.domain_rand = 'none'
    r.iteration_override = r.arm_end
    r.arm_full_extension_fraction = 1.0
    r.arm_full_extension_standing_fraction = 0.0
    r.arm_full_extension_hold_s = cfg.episode_length_s + 1.0
    r.arm_init_joint_noise = 0.0
    r.easy_reset_degrees = r.hard_reset_degrees = 0.0
    r.height_range = (0.33, 0.33)
    r.roll_range = r.pitch_range = (0.0, 0.0)
    cmd = cfg.commands.base_velocity_pose
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.ranges.lin_vel_x = cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0.0, 0.0)
    agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
    agent.device = args.device
    agent.logger = 'tensorboard'
    env = gym.make(args.task, cfg=cfg).unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    try:
        runner = LocomotionOnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
        runtime = env._locomotion
        runtime.capture_evaluation = True
        manifest = {
            'task': args.task, 'seeds': args.seeds, 'replicas': args.replicas, 'steps': args.steps,
            'dt': env.step_dt, 'scenarios': SCENARIOS, 'robustness': r.to_dict(),
            'contract': runner._contract(), 'terrain_seed': cfg.seed,
            'legs': ['FL', 'FR', 'RL', 'RR'], 'domain_rand': 'none',
            'commands': '0-4s stand; 4-12s requested speed; 12-14s linear deceleration; 14s+ stand',
            'push_schedule_s': [8, 12], 'first_rollout_only': True,
            'hold_definition': 'continuous actual arm error <0.15 rad while upright',
        }
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        all_results = []
        for checkpoint in args.checkpoints:
            label = Path(checkpoint).stem
            digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            runner.load(checkpoint, load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            for seed in args.seeds:
                with torch.inference_mode():
                    env.reset(seed=seed)
                    runtime.sums.zero_(); runtime.squares.zero_(); runtime.value_counts.zero_()
                    runtime.counts.zero_(); runtime.episodes.zero_(); runtime.window_maxima.clear()
                    alive = torch.ones(n, dtype=torch.bool, device=env.device)
                    frames = []
                    command = env.command_manager.get_term('base_velocity_pose')
                    push_ids = torch.arange(5 * args.replicas, n, device=env.device)
                    generator = torch.Generator().manual_seed(seed + 100000)
                    push_delta = (torch.rand(2, args.replicas, 6, generator=generator) * 2 - 1).to(env.device) * 0.6
                    push_delta[..., 2] = 0
                    initial = torch.cat((env.scene['robot'].data.root_state_w, env.scene['robot'].data.joint_pos), -1).cpu().numpy()
                    for step in range(args.steps):
                        t = step * env.step_dt
                        scale = 1.0 if 4 <= t < 12 else max(0.0, (14 - t) / 2) if 12 <= t < 14 else 0.0
                        command.vel_command_b[:] = 0.0
                        command.height_command[:] = 0.33
                        command.pose_command[:] = 0.0
                        command.vel_command_b[args.replicas:2*args.replicas, 0] = .3 * scale
                        command.vel_command_b[2*args.replicas:3*args.replicas, 0] = .6 * scale
                        command.vel_command_b[3*args.replicas:4*args.replicas, 1] = .3 * scale
                        command.vel_command_b[4*args.replicas:5*args.replicas, 1] = -.3 * scale
                        for event, second in enumerate((8, 12)):
                            if step == round(second / env.step_dt):
                                robot = env.scene['robot']
                                velocity = robot.data.root_vel_w[push_ids].clone() + push_delta[event]
                                robot.write_root_velocity_to_sim(velocity, env_ids=push_ids)
                        obs = wrapped.get_observations()
                        _, _, dones, _ = wrapped.step(policy(obs))
                        runner.alg.policy.reset(dones)
                        frame = dict(runtime.last_evaluation)
                        frame['valid'] = alive.clone()
                        frame['truncated'] = env.reset_time_outs.clone()
                        frames.append(frame)
                        alive &= ~dones.bool()
                        if step % 250 == 0:
                            print(f'EVAL_PROGRESS {label} seed={seed} step={step}/{args.steps}', flush=True)
                    trace = {key: torch.stack([f[key] for f in frames]).cpu().numpy() for key in frames[0]}
                    np.savez_compressed(output / f'{label}_s{seed}_trace.npz', **trace, initial=initial, push_delta=push_delta.cpu().numpy())
                    result = {'checkpoint': str(Path(checkpoint).resolve()), 'sha256': digest, 'seed': seed,
                              'scenarios': summarize(trace, args.replicas, env.step_dt)}
                    all_results.append(result)
                    (output / 'results.json').write_text(json.dumps(all_results, indent=2, allow_nan=False))
                    print('EVAL_RESULT', json.dumps(result, allow_nan=False), flush=True)
        # Identical seeds must give identical pre-policy state across checkpoints.
        for seed in args.seeds:
            initial_states = [np.load(output / f'{Path(p).stem}_s{seed}_trace.npz')['initial'] for p in args.checkpoints]
            for state in initial_states[1:]:
                np.testing.assert_allclose(state, initial_states[0], rtol=0, atol=1e-6)
        (output / 'success.json').write_text(json.dumps({'completed': len(all_results), 'paired_initial_states_verified': True}))
        print('STABILITY_EVAL_PASS', flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
