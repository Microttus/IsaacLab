from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def cylinder_position_in_world_frame(
        env: ManagerBasedRLEnv,
        center_cfg: SceneEntityCfg = SceneEntityCfg("center"),
        pipe_cfg: SceneEntityCfg = SceneEntityCfg("pipe"),
        pin_cfg: SceneEntityCfg = SceneEntityCfg("pin"),
) -> torch.Tensor:
    center: RigidObject = env.scene[center_cfg.name]
    pipe: RigidObject = env.scene[pipe_cfg.name]
    pin: RigidObject = env.scene[pin_cfg.name]

    return torch.cat((center.data.root_pos_w, pipe.data.root_pos_w, pin.data.root_pos_w), dim=1)

def pin_pos_in_env_frame(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("pin"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    return torch.cat(asset.data.root_pos_w, dim=1)

def gripper_pos(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

'''
def object_at_target(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        target_pos: torch.Tensor,
        threshold: float = 0.05,
) -> torch.Tensor:
    """Check if the specified object is in the target position.

    Args:
    :param env: The ManagerBasedRLEnv environment
    :param object_cfg: SceneEntityCfg specifying the rigid object name
    :param target_pos: A torch.Tensor specifying the desired target position for each environment,
                      shape (num_envs, 3) if multi-env, or (1, 3) for single-env.
    :param threshold: Distance threshold to consider the object "at" the target.
    :return: A boolean tensor of shape (num_envs,) where True means the object is at the target.
    """

    object_cyl: RigidObject = env.scene[object_cfg.name]
    object_pos = object_cyl.data.root_pos_w
    distance = torch.linalg.vector_norm(object_pos - target_pos.to(env.device), dim=1)

    return distance < threshold
'''

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped