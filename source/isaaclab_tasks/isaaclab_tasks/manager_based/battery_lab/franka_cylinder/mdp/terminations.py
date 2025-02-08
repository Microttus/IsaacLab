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
) -> torch.Tensor :
    """
    Terminate is the object has a position outside the bounds.

    Note:
        This function is a custom function allowing the user to set custom bounds for termination.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # violation?
    out_of_upper_limit = torch.any(asset.data.root_pos_w > bounds[1], dim=1)
    out_of_lower_limit = torch.any(asset.data.root_pos_w < bounds[0], dim=1)

    return torch.logical_or(out_of_upper_limit, out_of_lower_limit)


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

