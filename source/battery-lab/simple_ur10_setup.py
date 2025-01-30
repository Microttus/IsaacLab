# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script that demonstrates how to use the interactive scene interface to set up a scene
with multiple UR10 robots, plus ground plane, dome light, and a custom CAD rigid body (with gravity enabled),
in the IsaacLab/OmniIsaac framework.

Usage from IsaacLab:

.. code-block:: bash

    # For example, launch with 3 environments
    ./isaaclab.sh -p path/to/my_ur10_scene_script.py --num_envs 3
"""

import argparse

from omni.isaac.lab.app import AppLauncher

# Build the CLI parser
parser = argparse.ArgumentParser(description="Example script for an InteractiveScene with UR10 robots + custom CAD.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# Append the default AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)

# Parse CLI
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# IsaacLab imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import UR10_CFG  # If you have a pre-defined UR10 config. Otherwise define your own.

from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidPrimCfg, RigidBodyPropertiesCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

@configclass
class UR10SceneCfg(InteractiveSceneCfg):
    """
    Configuration for a scene that places:
      - UR10 robots
      - Ground plane
      - Dome light
      - A custom CAD object as a rigid body
    """
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # UR10 articulation
    robot: ArticulationCfg = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10")

    # Add a custom CAD as a rigid body with gravity enabled
    # Replace the usd_path below with your actual path.
    my_custom_cad: RigidPrimCfg = RigidPrimCfg(
        prim_path="{ENV_REGEX_NS}/MyCAD",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery-lab/Four_box_ass.usd",
            activate_contact_sensors=False
        ),
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,  # gravity is enabled
            max_depenetration_velocity=5.0,
        )
    )


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """
    Runs a simple simulation loop that periodically resets all UR10s and applies random joint efforts.
    The custom CAD is present as well, subject to gravity. We do nothing special with it here.
    """
    # Access the UR10 group using the config name: "robot"
    robot = scene["robot"]

    # Access the custom CAD rigid body if needed:
    my_cad = scene["my_custom_cad"]

    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        # Reset every 500 steps
        if step_count % 500 == 0:
            step_count = 0

            # Root state reset for UR10
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins  # offset by environment origins
            robot.write_root_link_pose_to_sim(root_state[:, :7])
            robot.write_root_com_velocity_to_sim(root_state[:, 7:])

            # Joint state reset for UR10
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.2  # random small offsets
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # (Optional) reset the custom CAD root state if you want to place it at a known location
            cad_root_state = my_cad.data.default_root_state.clone()
            cad_root_state[:, :3] += scene.env_origins  # place one in each environment
            my_cad.write_root_state_to_sim(cad_root_state)

            # Clear buffers
            scene.reset()
            print("[INFO]: Resetting UR10 and custom CAD state...")

        # Apply random efforts to UR10
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)

        # Write data to simulation
        scene.write_data_to_sim()

        # Step simulation
        sim.step()
        step_count += 1

        # Update buffers
        scene.update(sim_dt)


def main():
    # Create the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Set a camera view so we can see the scene
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 1.0])

    # Build an InteractiveScene with UR10, custom CAD, ground plane, and dome light.
    scene_cfg = UR10SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Ready to play
    sim.reset()
    print("[INFO]: Scene setup complete. Starting simulation...")

    # Run our simulation loop
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
