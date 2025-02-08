# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_assets import FRANKA_PANDA_CFG

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
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
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
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
    pipe: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/pipe.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
    pin: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pin",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/pin.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )


#
# MDP: Actions, Observations, Events, Rewards, Terminations
#


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=100.0
    )
    gripper_effort = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
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
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self) -> None:
            # same style as the cartpole
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_center = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("center"),
            }
        )
        grasp_pipe = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("pipe"),
            }
        )
        grasp_pin = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("pin"),
            }
        )

        def __post_init__(self) -> None:
            # same style as the cartpole
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()  # single group named policy
    subtask: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Configuration for events: resets, etc."""

    #reset
    reset_stage = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={}
    )
    '''
    reset_ur10 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint"],
            "pos_range": (-0.2, 0.2),
            "vel_range": (-0.2, 0.2),
        },
    )

    reset_box = EventTerm(
        func=mdp.reset_cylinder_pos,
        mode="reset",
        params={
            "center_cfg": SceneEntityCfg("center"),
            "pipe_cfg": SceneEntityCfg("pipe"),
            "pin_cfg": SceneEntityCfg("pin"),
        },
    )
    '''


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Alive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Joint velocity penalty
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"), #TODO: Can specify joints!
            "soft_ratio": 10.0,
        }
    )
    # (4) End effector near objects #TODO: Add this
    center_effector_penalty = RewTerm(
        func=mdp.object_distance_error_reward,
        weight=-0.05,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("center"),
        }
    )
    pipe_effector_penalty = RewTerm(
        func=mdp.object_distance_error_reward,
        weight=-0.05,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("pipe"),
        }
    )
    pin_effector_penalty = RewTerm(
        func=mdp.object_distance_error_reward,
        weight=-0.05,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("pin"),
        }
    )
    # (5) Objects in correct pos    #TODO: Add this
        # success = RewTerm(func=my_mdp.task_success, weight=5.0)
    # (4) Object pos compared to target
    center_penalty = RewTerm(
        func=mdp.position_error_reward,
        weight=-1.0,
        params={
            "object_cfg": SceneEntityCfg("center"),
            "target_pos": (0.5, 0.5, 0.0),
        }
    )
    pin_penalty = RewTerm(
        func=mdp.position_error_reward,
        weight=-1.5,
        params={
            "object_cfg": SceneEntityCfg("pin"),
            "target_pos": (0.5, -0.5, 0.0),
        }
    )
    pipe_penalty = RewTerm(
        func=mdp.position_error_reward,
        weight=-2.0,
        params={
            "object_cfg": SceneEntityCfg("pipe"),
            "target_pos": (1.0, 0.0, 0.0),
        }
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Add box velocity etc #TODO: Add something sensible
        # box_fell = DoneTerm(func=my_mdp.box_fell_off_table)
    # (3) If objects has high speed
    """
    center_speed = DoneTerm(
        func=mdp.object_speed_over_max,
        params={
            "max_speed": 30.0,
            "asset_cfg": SceneEntityCfg("center")},
       time_out=True,
    )
    """


#
# Environment configuration
#


@configclass
class FrankaCylinderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UR10 + box manager-based environment."""

    print(f"NVIDIA root dir : {ISAACLAB_NUCLEUS_DIR}/NVIDIA")

    # scene settings
    scene: FrankaCylinderSceneCfg = FrankaCylinderSceneCfg(num_envs=16, env_spacing=2.0)
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
