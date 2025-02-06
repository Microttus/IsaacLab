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
    return torch.any(asset.data.body_link_lin_vel_w[:] > max_speed)