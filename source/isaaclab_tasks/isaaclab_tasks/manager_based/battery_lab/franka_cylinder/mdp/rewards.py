from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_error_reward(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        target_pos=(0.0, 0.0, 0.0)
):
    """
    Example: penalty = - distance(object_pos, target_pos).
    """
    # 1) Rigid object handle
    obj: RigidObject = env.scene[object_cfg.name]
    # 2) Current position
    curr_pos_w = obj.data.root_pos_w  # shape (num_envs, 3)
    # 3) Convert your param to a (num_envs, 3) tensor (if using multiple envs):
    target_pos_t = torch.tensor(target_pos, device=env.device).expand(curr_pos_w.shape[0], -1)
    # 4) Distance
    dist = torch.norm(curr_pos_w - target_pos_t, dim=1)
    # 5) Return the absolut distance
    return torch.sum(torch.abs(dist))

def object_distance_error_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    return torch.sum(torch.abs(pose_diff))