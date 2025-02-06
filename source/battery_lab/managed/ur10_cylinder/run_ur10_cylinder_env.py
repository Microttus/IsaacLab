# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the manager-based UR10 + box environment.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run the UR10 + cylinder manager-based environment.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

# The manager-based RL environment base
from isaaclab.envs import ManagerBasedRLEnv
# Our environment config
from my_ur10_cylinder_env_cfg import MyUR10CylinderEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = MyUR10CylinderEnvCfg()
    # override the default from 16 to user-specified
    env_cfg.scene.num_envs = args_cli.num_envs

    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulation loop
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset every 300 steps
            if step_count % 300 == 0:
                step_count = 0
                env.reset()
                print("[INFO]: Reset environment.")

            # sample random actions
            # shape: (num_envs, action_dim) if needed
            actions = torch.randn_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(actions)

            # print example of an observation
            print("[Env 0]: UR10 joint pos:", obs["policy"][0][0].item())

            step_count += 1

    # close
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
