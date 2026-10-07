# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""VelocityPose hooks around the installed Isaac Lab RSL-RL entrypoints."""

import os
from functools import partial


def run(argv, *, play=False):
    from isaaclab_rl.entrypoints.backends import play_rsl_rl, train_rsl_rl

    from .robustness_runner import LocomotionOnPolicyRunner

    backend = play_rsl_rl if play else train_rsl_rl
    original = {
        name: getattr(backend, name)
        for name in ("setup_preset_cli", "resolve_task_config", "create_isaaclab_env", "create_rsl_rl_runner")
    }
    if play:
        original["run_playback"] = backend.run_playback
    context = {}

    def setup(parser, argv):
        if play:
            parser.add_argument("--keyboard", action="store_true")
            parser.add_argument("--robustness_iteration", type=int)
            parser.add_argument("--domain_rand", choices=["none", "benchmark", "sim2real"], default="benchmark")
            parser.add_argument("--arm_mode", choices=["random", "structured", "hold"])
            parser.add_argument("--debug_vis", action="store_true")
            parser.add_argument("--max_steps", type=int, default=0)
            parser.add_argument("--camera_follow_env0", action="store_true")
            parser.add_argument("--curriculum_stage", type=int, choices=[1, 2, 3, 4, 5], default=4)
        else:
            parser.add_argument("--tensorboard-log-dir")
        args, remaining = original["setup_preset_cli"](parser, argv)
        context["args"] = args
        return args, remaining

    def resolve(*args, **kwargs):
        cfg, agent = original["resolve_task_config"](*args, **kwargs)
        context["agent"] = agent
        return cfg, agent

    def create_env(task, cfg, args, **kwargs):
        if getattr(cfg, "robustness", None) is not None:
            cfg.robustness.steps_per_iteration = context["agent"].num_steps_per_env
            cfg.robustness.metrics_interval = context["agent"].num_steps_per_env
            if play:
                cfg.inference_mode = True
                cfg.inference_stage = args.curriculum_stage
                cfg.robustness.domain_rand = args.domain_rand
                if args.robustness_iteration is not None:
                    cfg.robustness.iteration_override = args.robustness_iteration
                if args.arm_mode is not None:
                    cfg.robustness.arm_mode = ("random", "structured", "hold").index(args.arm_mode)
                cfg.commands.base_velocity_pose.debug_vis = args.debug_vis
        if play and args.keyboard:
            cfg.scene.num_envs = 1
            cfg.terminations.time_out = None
        if play and args.camera_follow_env0:
            from isaaclab_visualizers.kit import KitVisualizerCfg

            camera = cfg.sim.default_visualizer_cfg
            camera.eye, camera.lookat = (-3.0, 0.0, 0.6), (0.0, 0.0, 0.25)
            cfg.sim.visualizer_cfgs = [
                KitVisualizerCfg(origin_type="asset", origin_track_path="robot", eye=camera.eye, lookat=camera.lookat)
            ]
        return original["create_isaaclab_env"](task, cfg, args, **kwargs)

    def create_runner(env, cfg, **kwargs):
        if getattr(env.unwrapped.cfg, "robustness", None) is not None:
            runner = LocomotionOnPolicyRunner(env, cfg.to_dict(), device=cfg.device, **kwargs)
        else:
            runner = original["create_rsl_rl_runner"](env, cfg, **kwargs)
        env.unwrapped._rsl_rl_runner = runner
        context.update(env=env, runner=runner)
        if play:
            runner.load = partial(
                runner.load,
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True},
                map_location=cfg.device,
            )
        elif context["args"].tensorboard_log_dir:
            if cfg.logger != "tensorboard":
                raise ValueError("--tensorboard-log-dir requires --logger tensorboard")
            native_writer = runner.logger.init_logging_writer

            def init_writer():
                checkpoint_dir = runner.logger.log_dir
                runner.logger.log_dir = os.path.abspath(context["args"].tensorboard_log_dir)
                try:
                    native_writer()
                finally:
                    runner.logger.log_dir = checkpoint_dir

            runner.logger.init_logging_writer = init_writer
        return runner

    def run_playback(step, *, dt, args_cli, env_cfg):
        env = context["env"]
        policy = context["runner"].get_inference_policy(device=env.unwrapped.device)
        playback(env, policy, args_cli, env_cfg)

    backend.setup_preset_cli = setup
    backend.resolve_task_config = resolve
    backend.create_isaaclab_env = create_env
    backend.create_rsl_rl_runner = create_runner
    if play:
        backend.run_playback = run_playback
    try:
        backend.run(argv)
    finally:
        for name, value in original.items():
            setattr(backend, name, value)


def playback(env, policy, args_cli, env_cfg):
    import time

    import torch

    from isaaclab_rl.entrypoints.common import video_playback_steps

    obs = env.get_observations()
    controller = None
    if args_cli.keyboard:
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        command_cfg = getattr(env_cfg.commands, "base_velocity_pose", None) or env_cfg.commands.base_velocity
        controller = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=command_cfg.ranges.lin_vel_x[1],
                v_y_sensitivity=command_cfg.ranges.lin_vel_y[1],
                omega_z_sensitivity=command_cfg.ranges.ang_vel_z[1],
            )
        )
    limit = video_playback_steps(args_cli, env_cfg)
    if args_cli.max_steps > 0:
        limit = min(limit, args_cli.max_steps) if limit is not None else args_cli.max_steps
    timestep = 0
    try:
        while limit is None or timestep < limit:
            start = time.time()
            with torch.inference_mode():
                if controller is not None:
                    name = "base_velocity_pose" if hasattr(env_cfg.commands, "base_velocity_pose") else "base_velocity"
                    term = env.unwrapped.command_manager.get_term(name)
                    term.vel_command_b[:] = torch.as_tensor(
                        controller.advance(), dtype=torch.float32, device=env.device
                    )
                    obs = env.get_observations()
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            timestep += 1
            delay = env.unwrapped.step_dt - (time.time() - start)
            if args_cli.real_time and delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    print(f"[INFO] Playback completed: {timestep} steps; environment policy steps: {env.unwrapped.common_step_counter}")
