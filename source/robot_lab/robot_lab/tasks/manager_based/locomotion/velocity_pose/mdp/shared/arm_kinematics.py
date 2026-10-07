# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Forward kinematics from the robot's exact USD, ending at the link6 origin."""

import torch

from isaaclab.utils.math import matrix_from_quat


class ArmKinematics:
    def __init__(self, usd_path, joint_names, device, root_name="base", tip_name="link6"):
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(usd_path)
        by_child = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                child = joint.GetBody1Rel().GetTargets()
                if child:
                    by_child[str(child[0])] = joint
        tips = [path for path in by_child if path.rsplit("/", 1)[-1] == tip_name]
        if len(tips) != 1:
            raise ValueError(f"Expected one {tip_name} in {usd_path}: {tips}")
        chain = []
        path = tips[0]
        while path.rsplit("/", 1)[-1] != root_name:
            joint = by_child[path]
            parent = joint.GetBody0Rel().GetTargets()
            if not parent:
                raise ValueError(f"Broken arm chain at {joint.GetPath()}")
            name = joint.GetPrim().GetName()
            index = joint_names.index(name) if name in joint_names else None
            if index is None and not joint.GetPrim().IsA(UsdPhysics.FixedJoint):
                raise ValueError(f"Unmapped movable arm joint: {name}")

            def transform(pos, rot):
                q = torch.tensor([*rot.GetImaginary(), rot.GetReal()], device=device)
                result = torch.eye(4, device=device)
                result[:3, :3] = matrix_from_quat(q)
                result[:3, 3] = torch.tensor(tuple(pos), device=device)
                return result

            before = transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            after = torch.linalg.inv(transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()))
            axis = None if index is None else "XYZ".index(UsdPhysics.RevoluteJoint(joint.GetPrim()).GetAxisAttr().Get())
            chain.append((index, axis, before, after))
            path = str(parent[0])
        self.chain = list(reversed(chain))
        if sorted(index for index, _, _, _ in self.chain if index is not None) != list(range(len(joint_names))):
            raise ValueError("USD arm chain does not contain all configured arm joints")

    def position(self, angles):
        n = angles.shape[0]
        result = torch.eye(4, device=angles.device, dtype=angles.dtype).expand(n, 4, 4)
        for index, axis, before, after in self.chain:
            result = result @ before
            if index is not None:
                rotation = torch.eye(4, device=angles.device, dtype=angles.dtype).repeat(n, 1, 1)
                i, j = (axis + 1) % 3, (axis + 2) % 3
                c, s = angles[:, index].cos(), angles[:, index].sin()
                rotation[:, i, i] = rotation[:, j, j] = c
                rotation[:, i, j], rotation[:, j, i] = -s, s
                result = result @ rotation
            result = result @ after
        return result[:, :3, 3]
