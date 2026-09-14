# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from robustness_cli import normalize_robustness_float_overrides  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--robustness_iteration",
    type=int,
    default=None,
    help="Fixed robustness evaluation iteration; default restores checkpoint progress.",
)
parser.add_argument("--domain_rand", choices=["none", "benchmark", "sim2real"], default="benchmark")
parser.add_argument("--arm_mode", choices=["random", "structured", "hold"], default=None)
parser.add_argument("--debug_vis", action="store_true", help="Show pose markers for at most 8 environments.")
parser.add_argument(
    "--max_steps", type=int, default=0, help="Stop after this many playback steps; 0 runs continuously."
)
parser.add_argument(
    "--curriculum_stage",
    type=int,
    default=4,
    choices=[1, 2, 3, 4, 5],
    help=(
        "Curriculum stage to use for inference (1=base, 2=small range, 3=medium range, 4=max range, 5=extreme)."
        " Default: 4"
    ),
)
parser.add_argument(
    "--arm_actions_idx",
    type=int,
    default=None,
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    help=(
        "Force specific arm motion mode for all environments (0=circular, 1=figure_eight, 2=sinusoidal, 3=random_walk,"
        " 4=reach_points, 5=fishing, 6=grasping, 7=swinging, 8=probing). If not specified, uses random cyclic"
        " assignment."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
normalized_hydra_args = normalize_robustness_float_overrides(hydra_args)
if normalized_hydra_args != hydra_args:
    print("[INFO] Normalized integer robustness overrides to float literals for IsaacLab.")
hydra_args = normalized_hydra_args
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None and getattr(env_cfg, "robustness", None) is None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = (
        None if getattr(env_cfg, "robustness", None) is None else env_cfg.events.randomize_push_robot
    )
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        keyboard_command = (
            env_cfg.commands.base_velocity_pose
            if hasattr(env_cfg.commands, "base_velocity_pose")
            else env_cfg.commands.base_velocity
        )
        keyboard_command.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=keyboard_command.ranges.lin_vel_x[1],
            v_y_sensitivity=keyboard_command.ranges.lin_vel_y[1],
            omega_z_sensitivity=keyboard_command.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)

        def keyboard_commands(env):
            velocity = torch.tensor(controller.advance(), dtype=torch.float32, device=env.device).unsqueeze(0)
            if hasattr(env.cfg.commands, "base_velocity_pose"):
                term = env.command_manager.get_term("base_velocity_pose")
                term.vel_command_b[:] = velocity
                return term.command
            return velocity

        env_cfg.observations.policy.velocity_commands = ObsTerm(func=keyboard_commands)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # CRITICAL: Set inference stage BEFORE environment creation
    # This allows DogArmCompositeAction to initialize with correct stage
    env_cfg.inference_mode = True
    env_cfg.inference_stage = args_cli.curriculum_stage

    if getattr(env_cfg, "robustness", None) is not None:
        env_cfg.robustness.domain_rand = args_cli.domain_rand
        env_cfg.robustness.steps_per_iteration = agent_cfg.num_steps_per_env
        if args_cli.robustness_iteration is not None:
            env_cfg.robustness.iteration_override = args_cli.robustness_iteration
        if args_cli.arm_actions_idx is not None:
            raise ValueError(
                "The bounded arm generator uses --arm_mode random|structured|hold; the legacy 9 modes are no longer"
                " used."
            )
        if args_cli.arm_mode is not None:
            env_cfg.robustness.arm_mode = ("random", "structured", "hold").index(args_cli.arm_mode)
        env_cfg.commands.base_velocity_pose.debug_vis = args_cli.debug_vis
        env_cfg.log_dir = os.path.join(log_dir, "evaluation")
    elif args_cli.arm_actions_idx is not None:
        env_cfg.fixed_arm_mode_idx = args_cli.arm_actions_idx

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Also set on unwrapped env for curriculum to read
    env.unwrapped._is_inference_mode = True  # type: ignore[attr-defined]
    env.unwrapped._inference_curriculum_stage = args_cli.curriculum_stage  # type: ignore[attr-defined]

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Mark as inference mode for curriculum (no runner injection)
    # Set the curriculum stage for inference based on command line argument
    env.unwrapped._is_inference_mode = True  # type: ignore[attr-defined]
    env.unwrapped._inference_curriculum_stage = args_cli.curriculum_stage  # type: ignore[attr-defined]

    stage_descriptions = {
        1: "Stage 1 (Base): Height/pose fixed at default, velocity control only",
        2: "Stage 2 (Small): ±3cm height, ±8° roll",
        3: "Stage 3 (Medium): ±10cm height, ±20° roll, ±12° pitch",
        4: "Stage 4 (Maximum): ±15cm height, ±30° roll, ±15° pitch",
        5: "Stage 5 (Extreme): Same as Stage 4 but with 1.5x arm motion amplitude",
    }
    if getattr(env_cfg, "robustness", None) is None:
        print(f"[INFO] Curriculum Stage for Inference: {stage_descriptions[args_cli.curriculum_stage]}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner_type = OnPolicyRunner
        if getattr(env_cfg, "robustness", None) is not None:
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.robustness_runner import (
                LocomotionOnPolicyRunner,
            )

            runner_type = LocomotionOnPolicyRunner
        runner = runner_type(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        timestep += 1
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[INFO] Playback completed: {timestep} steps; environment policy steps: {env.unwrapped.common_step_counter}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
