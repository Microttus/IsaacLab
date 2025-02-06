from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_cylinder_pos(
        env: ManagerBasedRLEnv,
        center_cfg: SceneEntityCfg,
        pipe_cfg: SceneEntityCfg,
        pin_cfg: SceneEntityCfg
):

    # Center reset
    #center_state = center_cfg.data.default_root_state.clone()
    #center_state[:, :3] += scene.env_origins
    #center.write_root_state_to_sim(center_state)

    return False