from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
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
    # 5) Return negative distance as a penalty
    return torch.sum(dist)
