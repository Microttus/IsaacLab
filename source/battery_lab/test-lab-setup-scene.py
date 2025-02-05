"""
Author: Martin Økter
Mail: martino@uia.no
University of Agder

This is the first test setup for the simulation training
of disassembly using UR10 and a test box
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description='Test lab setup for disassembly using UR10 and a test box')
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")

# Parse the arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the onmiverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""And rest? Everything!"""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre defined configs
##
from omni.isaac.lab_assets import UR10_CFG

@configclass
class DisassemblyTestEnv(InteractiveSceneCfg):
    """
    Configuration class for the disassembly test environment.
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
        #init_state=AssetBaseCfg.InitStateCfg(pos=(0.0, 0.0, -1.05))
    )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot arm
    robot_arm = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_simulatior(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot_arm"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    #Simulation
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # Reset counter
            count = 0

            # Robot reset
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_link_pose_to_sim(root_state[:, :7])
            robot.write_root_com_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # Clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # Apply action
        robot.set_joint_effort_target(efforts)
        # Write to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

def main():
    """Mani function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = DisassemblyTestEnv(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    print("[INFO]: Starting simulation...")
    run_simulatior(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()