# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Blind locomotion on 50% plane, 25% +/-2 cm and 25% +/-4 cm ground."""

from isaaclab import sim as sim_utils
from isaaclab.terrains import HfRandomUniformTerrainCfg, MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2X5VelocityPoseRoughEnvCfg


def bounded_random_uniform_terrain(difficulty, cfg):
    """Retain smooth interpolation while bounding its cubic overshoot."""
    meshes, origin = random_uniform_terrain(difficulty, cfg)
    for mesh in meshes:
        mesh.vertices[:, 2] = mesh.vertices[:, 2].clip(*cfg.noise_range)
    origin[2] = origin[2].clip(*cfg.noise_range)
    return meshes, origin


@configclass
class BoundedRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
    function = bounded_random_uniform_terrain


@configclass
class UnitreeGo2X5VelocityPoseMildEnvCfg(UnitreeGo2X5VelocityPoseRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Stronger leg joint-limit penalty (Rough sets -5.0): the 4096-env Newton 25k run
        # drifted to action std ~24 with ~35 % raw saturation by update 3.3k.
        self.rewards.joint_pos_limits.weight = -10.0
        # At clip 3 (max offset 0.75 rad) the joints never reach the soft limits, so the
        # penalty is inert (~-1e-3/episode) and the mean drifts past the clip. Clip 10
        # (2.5 rad, as in the WTW feat/loco recipe) lets it act on a drifting mean.
        self.actions.joint_pos.clip = {".*": (-10.0, 10.0)}
        self.scene.terrain.terrain_type = "generator"
        # Curriculum=True allocates deterministic column types in these proportions.
        # No terrain-level curriculum runs; rows all have the same mild difficulty.
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=1234,
            size=(8.0, 8.0),
            border_width=10.0,
            num_rows=4,
            num_cols=20,
            curriculum=True,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            use_cache=False,
            sub_terrains={
                "plane": MeshPlaneTerrainCfg(proportion=0.5),
                "noise_2cm": BoundedRandomUniformTerrainCfg(
                    proportion=0.25,
                    noise_range=(-0.02, 0.02),
                    noise_step=0.005,
                    downsampled_scale=0.4,
                    border_width=0.25,
                ),
                "noise_4cm": BoundedRandomUniformTerrainCfg(
                    proportion=0.25,
                    noise_range=(-0.04, 0.04),
                    noise_step=0.005,
                    downsampled_scale=0.4,
                    border_width=0.25,
                ),
            },
        )
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35))

        if self.__class__.__name__ == "UnitreeGo2X5VelocityPoseMildEnvCfg":
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.newton_arm import configure_newton_x5

            configure_newton_x5(self)
