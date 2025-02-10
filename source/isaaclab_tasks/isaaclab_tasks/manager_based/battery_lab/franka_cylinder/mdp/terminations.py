from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def object_speed_over_max(
        env: ManagerBasedRLEnv,
        max_speed: float,
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor :
    """ Terminate is the object has a velocity above the threshold.

    Note:
        Just trying to get the velocity above the threshold. XD
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # violation?
    over_speed = torch.any(asset.data.root_vel_w > max_speed, dim=1)

    return over_speed


def object_outside_bounds(
        env: ManagerBasedRLEnv,
        bounds: tuple[float, float],
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Terminate if the object's x or y position is outside the given bounds.

    Assumes asset.data.com_pos_b is a tensor that might have shape [num_envs, 1, 3]
    (an extra singleton dimension) and returns a Boolean tensor of shape [num_envs],
    where True indicates that the object is out-of-bounds.

    Parameters:
        env: The environment instance.
        bounds: A tuple (lower_bound, upper_bound) to check against.
        asset_cfg: Configuration for the asset.

    Returns:
        A Boolean tensor of shape [num_envs] indicating termination.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.com_pos_b  # e.g., shape: [64, 1, 3]

    # Remove an extra singleton dimension if present.
    if pos.ndim == 3 and pos.shape[1] == 1:
        pos = pos.squeeze(1)  # Now pos.shape becomes [64, 3]

    # Ensure pos has a batch dimension.
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)

    # Check that we have at least 2 coordinates (x and y).
    if pos.shape[1] < 2:
        raise ValueError(f"Expected at least 2 coordinates for x and y, but got shape {pos.shape}")

    # Check x and y coordinates.
    out_x = (pos[:, 0] < bounds[0]) | (pos[:, 0] > bounds[1])
    out_y = (pos[:, 1] < bounds[0]) | (pos[:, 1] > bounds[1])

    # Combine the two conditions into a single termination flag per environment.
    termination_flags = out_x | out_y  # shape: [num_envs]

    return termination_flags


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminate when the asset's joint positions are outside the soft joint limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)

    return torch.logical_or(out_of_upper_limits[:, asset_cfg.joint_ids], out_of_lower_limits[:, asset_cfg.joint_ids])

