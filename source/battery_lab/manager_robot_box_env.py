# manager_robot_box_env.py

from __future__ import annotations
import argparse
from gettext import translation

from isaaclab.app import AppLauncher

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

# IsaacLab/OmniIsaac imports
from isaaclab_assets import UR10_CFG

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, AssetBaseCfg, RigidObject, \
    AssetBase
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class RobotBoxManagerCfg(ManagerBasedRLEnvCfg):
    """
    Manager-based environment config that spawns:
      - UR10 robot
      - 5 sub-objects from a single .usd
      - ground plane, dome light
    """

    # Overridden from ManagerRLEnvCfg
    episode_length_s: float = 10.0   # e.g. 10-second episodes
    decimation: int = 2
    action_space: int = 6
    observation_space: int = 24
    num_envs: int = 16

    def __post_init__(self):
        # apply user override
        self.scene.num_envs = self.num_envs

    # Simulation parameters
    sim: SimulationCfg = SimulationCfg(
        dt=1/60.0,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        )
    )

    # Scene config
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,              # number of env instances
        env_spacing=3.0,          # spacing between each environment
        replicate_physics=True
    )

    # Robot config (placeholder)
    '''
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/UR10",
        spawn=UsdFileCfg(
            usd_path="omniverse://my_server/Robots/UR10/ur10_instanceable.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        # actuators, etc. if needed
    )
    '''
    robot: ArticulationCfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/UR10")

    # We'll define 5 sub-objects as RigidObjects for demonstration
    # Suppose your .usd includes 5 distinct sub-prim paths
    # like "/BoxAssembly/Part_0", ..., "/BoxAssembly/Part_4"
    # We can define a list or dict of them.

    # Example of individually referencing each sub-prim:
    box_parts = [
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_0",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/battery_lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_1",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/battery_lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_2",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/battery_lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_3",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/battery_lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_4",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/battery_lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),

    ]

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/groundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    )


class RobotBoxManagerEnv(ManagerBasedRLEnv):
    cfg: RobotBoxManagerCfg

    def __init__(self, cfg: RobotBoxManagerCfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def _setup_scene(self):
        """
        Called once per environment instance by the manager. We'll create
        the UR10, the 5 sub-objects, ground plane, and dome light.
        """
        super()._setup_scene()

        # 1) Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # 2) Boxes (5 sub-objects)
        self._boxes = []
        for i, part_cfg in enumerate(self.cfg.box_parts):
            box = RigidObject(part_cfg)
            self.scene.rigid_objects[f"box_part_{i}"] = box
            self._boxes.append(box)

        # 3) Ground
        self._ground = AssetBase(self.cfg.ground)
        self.scene.prims["ground"] = self._ground

        # 4) Dome light
        self._light = AssetBase(self.cfg.dome_light)
        self.scene.prims["dome_light"] = self._light

        # 5) Clone envs if needed
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])

    def post_reset(self):
        """
        Called after all environment setup is done (per environment).
        We'll do a random reset or place the boxes/robot as needed.
        """
        super().post_reset()

        # e.g., randomize or set initial states
        # Robot: we can write joint states or root transforms
        for env_idx in self.scene.env_indices:
            # optional: random pose or default
            joint_pos = self._robot.data.default_joint_pos[env_idx].clone()
            joint_vel = self._robot.data.default_joint_vel[env_idx].clone()
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=[env_idx])

            # Place each box
            for box in self._boxes:
                box_state = box.data.default_root_state[env_idx].clone()
                # put them somewhere
                box_state[:3] += torch.tensor([0.5, 0.0, 0.3], device=box_state.device)
                box.write_root_state_to_sim(
                    box_state.unsqueeze(0), env_ids=[env_idx]
                )

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Called each step before physics. Typically parse 'actions' for the robot.
        """
        # parse actions if we do RL on UR10, etc.
        pass

    def _get_observations(self) -> dict:
        """
        Return observation dict for each environment.
        Typically gather robot joint states, box states, etc.
        """
        # example placeholder
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel
        # maybe box positions if you want them in the obs
        box_positions = [box.data.root_pos_w for box in self._boxes]

        # build your obs
        # e.g. combine them, clamp, etc.
        return {"policy": dof_pos}

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute your reward, e.g. distance to a target, or something involving the box objects.
        """
        # placeholder: 0
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self):
        """
        Return (terminated, truncated) boolean arrays per environment
        """
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated


def main():
    cfg = RobotBoxManagerCfg(num_envs=8)
    env = RobotBoxManagerEnv(cfg)
    for i in range(5):
        env.reset()
        for step in range(1000):
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            obs, reward, done, info = env.step(actions)
            if done.any():
                break

if __name__ == "__main__":
    main()