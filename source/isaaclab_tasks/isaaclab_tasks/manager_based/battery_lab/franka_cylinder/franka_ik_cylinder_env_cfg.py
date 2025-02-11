# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from urllib.parse import uses_relative

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.markers import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sim import ArticulationRootPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

#Robot
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# MDP
import isaaclab_tasks.manager_based.battery_lab.franka_cylinder.mdp as mdp


@configclass
class FrankaCylinderSceneCfg(InteractiveSceneCfg):
    """Configuration for a Franka + box assembly scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Franka
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Franka"
    )
    # Listens to the required transforms
    ee_frame: FrameTransformerCfg = MISSING

    # Add a custom CAD
    center: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Center",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/center.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
    pipe: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/pipe.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000000.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
    pin: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pin",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/pin.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )


#
# MDP: Actions, Observations, Events, Rewards, Terminations
#

@configclass
class ActionCfg:
    """Action configuration with implemented inverse kinematics."""
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", uses_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        cylinder_pos = ObsTerm(func=mdp.cylinder_position_in_world_frame)
        #pin_pos = ObsTerm(func=mdp.pin_pos_in_env_frame)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self) -> None:
            # same style as the cartpole
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()  # single group named policy


@configclass
class EventCfg:
    """Configuration for events: resets, etc."""

    #reset
    reset_stage = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={}
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Alive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-4.0)
    # (3) Joint velocity penalty
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"), # May can specify joints!
            "soft_ratio": 10.0,
        }
    )
    # (4) End effector near objects
    pin_effector_penalty = RewTerm(
        func=mdp.object_frame_distance_lin_reward,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("pin"),
        }
    )
    # (5) Pin removed from cylinder
    pin_removed_reward = RewTerm(
        func=mdp.object_2_distance_lin_reward,
        weight=1.0,
        params={
            "main_object_cfg": SceneEntityCfg("pin"),
            "ref_object_cfg": SceneEntityCfg("pipe"),
        }
    )
    # (6) Object pos compared to target
    pin_penalty = RewTerm(
        func=mdp.object_target_distance_lin_reward,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("pin"),
            "target": (0.5, 2.0, 0.0),
        }
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Object outside env
    pin_out = DoneTerm(
        func=mdp.object_outside_bounds,
        params={
            "asset_cfg": SceneEntityCfg("pin"),
            "bounds": (-3.0, 3.0),
        }
    )



#
# Environment configuration
#


@configclass
class FrankaCylinderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UR10 + box manager-based environment."""

    print(f"NVIDIA root dir : {ISAACLAB_NUCLEUS_DIR}/NVIDIA")

    # scene settings
    scene: FrankaCylinderSceneCfg = FrankaCylinderSceneCfg(num_envs=64, env_spacing=2.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # viewer
        self.viewer.eye = (8.0, 0.0, 5.0)
        # sim
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


        """
        Tool frame configurations
        """
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Franka/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Franka/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Franka/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Franka/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
