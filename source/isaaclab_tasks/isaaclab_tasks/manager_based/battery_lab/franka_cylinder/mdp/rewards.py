from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



"""
Custom position tracking rewards
"""


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

def object_frame_distance_lin_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Computes a reward based on the distance between an object and the end effector.
    The reward is higher when the object is closer to the target (end effector).

    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot (not used directly here, but might be needed in other variants).
        ee_frame_cfg: Configuration for the end-effector frame.
        object_cfg: Configuration for the object.
        scale: Scaling factor for the decay; higher values make the reward drop off faster with distance.

    Returns:
        A tensor of shape [num_envs] with rewards, where values are in (0, 1] (closer = higher reward).
    """
    # Retrieve the scene elements.
    robot: Articulation = env.scene[robot_cfg.name]         # (Unused in this simple version.)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Get the world positions.
    object_pos = obj.data.root_pos_w      # Expected shape: [num_envs, dims]
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]  # Expected shape: [num_envs, dims]

    # Compute the Euclidean distance between the object and the end effector.
    distance = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)  # Shape: [num_envs]

    # Convert the distance into a reward. Using an exponential decay yields a reward close to 1 when distance is small,
    # and close to 0 when the distance is large.
    reward = torch.exp(-scale * distance)

    return reward


def object_target_distance_lin_reward(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        target: tuple[float, float, float],
        scale: float = 1.0,
) -> torch.Tensor:
    """
    Computes a reward based on the Euclidean distance between an object and a target position.
    The reward is high when the object is near the target and decays exponentially with distance.

    The reward is defined as:
        reward = exp(-scale * distance)
    where `distance` is the Euclidean distance between the object's position and the target.

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object.
        target: A tuple (x, y, z) specifying the target position.
        scale: A scaling factor that controls how rapidly the reward decays with distance.

    Returns:
        A tensor of shape [num_envs] containing the reward for each environment.
    """
    # Retrieve the object from the environment.
    obj: RigidObject = env.scene[object_cfg.name]

    # Get the object's world position. Expected shape: [num_envs, dims]
    object_pos = obj.data.root_pos_w

    # Convert the target tuple into a tensor with the same dtype and device as object_pos.
    target_tensor = torch.tensor(target, dtype=object_pos.dtype, device=object_pos.device)

    # Compute the Euclidean distance between each object's position and the target.
    # Broadcasting ensures that target_tensor of shape [dims] subtracts from object_pos [num_envs, dims]
    distance = torch.linalg.vector_norm(object_pos - target_tensor, dim=1)

    # Compute the reward using an exponential decay: the closer the object is to the target, the higher the reward.
    reward = torch.exp(-scale * distance)

    return reward

def object_2_distance_lin_reward(
    env: ManagerBasedRLEnv,
    main_object_cfg: SceneEntityCfg,
    ref_object_cfg: SceneEntityCfg,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Computes a reward based on the distance between two objects.
    The reward is higher when the object is further apart from each other.

    Args:
        env: The RL environment instance.
        main_object_cfg: Configuration for the main movable object.
        ref_object_cfg: Configuration for the reference object.
        scale: Scaling factor for the decay; higher values make the reward drop off faster with distance.

    Returns:
        A tensor of shape [num_envs] with rewards, where values are in (0, 1) (higher = higher reward).
    """
    # Retrieve the scene elements.
    main_obj: RigidObject = env.scene[main_object_cfg.name]
    ref_obj: RigidObject = env.scene[ref_object_cfg.name]

    # Get the world positions.
    main_object_pos = main_obj.data.root_pos_w      # Expected shape: [num_envs, dims]
    ref_object_pos = ref_obj.data.root_pos_w

    # Compute the Euclidean distance between the object and the end effector.
    distance = torch.linalg.vector_norm(main_object_pos - ref_object_pos, dim=1)  # Shape: [num_envs]

    # Convert the distance into a reward. Using an exponential decay yields a reward close to 1 when distance is small,
    # and close to 0 when the distance is large.
    reward = 1 - torch.exp(-scale * distance)

    return reward