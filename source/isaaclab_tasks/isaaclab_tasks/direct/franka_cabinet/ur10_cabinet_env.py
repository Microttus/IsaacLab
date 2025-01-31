# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

# OmniIsaac / IsaacLab imports
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class UR10BlockEnvCfg(DirectRLEnvCfg):
    """
    Minimal environment config to demonstrate a UR10 picking a block.
    We'll keep it simpler than the complex cabinet example.
    """
    # Basic RL config
    episode_length_s = 5.0  # short episodes
    decimation = 2
    action_space = 6  # e.g. 6 DOFs for UR10?
    observation_space = 18  # example dimension
    state_space = 0

    # Simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene config: how many environments, spacing, etc.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)

    # Robot config (UR10). Here, you can adapt to your actual UR10 asset.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    '''robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/UR10",
        spawn=sim_utils.UsdFileCfg(
            # Provide a real UR10 USD if needed, or replace with a simpler arm.
            # Below is just a placeholder for demonstration.
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UR10/ur10_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            # Optionally define default joint positions. This is placeholder.
            joint_pos={
                "ur10_joint1": 0.0,
                "ur10_joint2": -1.57,
                "ur10_joint3": 0.0,
                "ur10_joint4": -1.57,
                "ur10_joint5": 0.0,
                "ur10_joint6": 0.0,
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["ur10_joint[1-3]"],
                effort_limit=150.0,
                velocity_limit=2.0,
                stiffness=40.0,
                damping=4.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["ur10_joint[4-6]"],
                effort_limit=100.0,
                velocity_limit=2.0,
                stiffness=40.0,
                damping=4.0,
            ),
        },
    )'''

    # Block config
    block = ArticulationCfg(
        prim_path="/World/envs/env_.*/Block",
        spawn=sim_utils.UsdFileCfg(
            # We can specify a box or any block USD in the nucleus.
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/BasicShapes/BoxPrimitive.usd",
            activate_contact_sensors=False,
        ),
        # We'll treat the block as an articulation for demonstration, though
        # it could also be a rigid body or something simpler.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.02),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={},
        ),
    )

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Misc.
    action_scale = 1.0
    dof_velocity_scale = 0.1

    # Some reward scales
    dist_reward_scale = 2.0
    action_penalty_scale = 0.01


class UR10BlockEnv(DirectRLEnv):
    """
    A simplified environment to place a UR10 and a block in the scene.
    Demonstrates how to structure the environment so it can be launched.
    """
    cfg: UR10BlockEnvCfg

    def __init__(self, cfg: UR10BlockEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # placeholders for dof limits, etc.
        self.robot_dof_lower_limits = torch.empty(0, device=self.device)
        self.robot_dof_upper_limits = torch.empty(0, device=self.device)
        # We'll store our target dof positions for the UR10.
        self.robot_dof_targets = torch.empty(0, device=self.device)

    def _setup_scene(self):
        """
        Called automatically by the parent class.
        We'll instantiate the UR10, the block, and the ground.
        """
        # 1. Create the UR10 articulation.
        self._robot = Articulation(self.cfg.robot)
        # 2. Create the block articulation.
        self._block = Articulation(self.cfg.block)
        # 3. Add them to the scene.
        self.scene.articulations["ur10"] = self._robot
        self.scene.articulations["block"] = self._block

        # 4. Create the plane.
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 5. Clone, filter collisions, etc.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 6. Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def post_reset(self):
        """
        Called once after scene setup is done.
        We'll read DOF limits from the robot.
        """
        super().post_reset()

        # read the DOF limits from the UR10
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        # init dof targets
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + (self.dt * self.actions * self.cfg.action_scale)
        # clamp to safe joint range
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (terminated, truncated) boolean tensors.
        We'll do a simple time-based termination.
        """
        terminated = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """
        We'll do a simple distance to block reward, plus an action penalty.
        """
        # refresh intermediate values if needed.
        self._compute_intermediate_values()

        # Suppose we track the end-effector link. We'll pick the last link as the end-effector.
        eef_pos = self._robot.data.body_link_pos_w[:, -1]
        block_pos = self._block.data.root_pos_w
        dist = torch.norm(eef_pos - block_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist)
        dist_reward = dist_reward * self.cfg.dist_reward_scale

        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions ** 2, dim=-1)
        rewards = dist_reward - action_penalty
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # randomize the robot DOF state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.2,
            0.2,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        # set the targets and the sim state
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # randomize block location
        block_pos = self._block.data.default_root_pos[env_ids]
        block_pos[:, 0] += torch.rand_like(block_pos[:, 0]) * 0.2 - 0.1  # x
        block_pos[:, 1] += torch.rand_like(block_pos[:, 1]) * 0.2 - 0.1  # y
        block_pos[:, 2] = 0.025  # small offset
        block_rot = self._block.data.default_root_quat[env_ids]

        self._block.set_root_state(block_pos, block_rot, env_ids=env_ids)

        # refresh
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # We'll build an observation of [dof_pos, dof_vel, eef->block vector].
        dof_pos = self._robot.data.joint_pos
        # scale the dof pos to [-1, 1]
        dof_pos_scaled = 2.0 * (dof_pos - self.robot_dof_lower_limits) / (
            self.robot_dof_upper_limits - self.robot_dof_lower_limits
        ) - 1.0

        dof_vel = self._robot.data.joint_vel * self.cfg.dof_velocity_scale

        eef_pos = self._robot.data.body_link_pos_w[:, -1]
        block_pos = self._block.data.root_pos_w
        to_block = block_pos - eef_pos

        obs = torch.cat((dof_pos_scaled, dof_vel, to_block), dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        # not used in this simple example, but you can store intermediate transforms here
        pass
