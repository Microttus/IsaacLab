# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import ArticulationRootPropertiesCfg
from isaaclab.utils import configclass

#UR10
from isaaclab_assets import UR10_CFG

# MDP
import isaaclab_tasks.manager_based.battery_lab.ur10_cylinder.mdp as mdp


@configclass
class MyUR10CylinderSceneCfg(InteractiveSceneCfg):
    """Configuration for a UR10 + box assembly scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )

    # UR10
    robot: ArticulationCfg = UR10_CFG.replace(
        prim_path="/World/envs/env_.*/UR10"
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Add a custom CAD
    center: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Center",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/center.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.1)),
    )
    pipe: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/pipe.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.1)),
    )
    pin: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pin",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rhino/IsaacLab/source/battery_lab/managed/ur10_box/pin.usd",
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.1)),
    )


#
# MDP: Actions, Observations, Events, Rewards, Terminations
#


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint",
                     "shoulder_lift_joint",
                     "elbow_joint",
                     "wrist_1_joint",
                     "wrist_2_joint",
                     "wrist_3_joint"],
        scale=100.0
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

        def __post_init__(self) -> None:
            # same style as the cartpole
            self.enable_corruption = False
            self.concatenate_terms = True

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
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"), #TODO: Can specify joints!
            "soft_ratio": 2.0,
        }
    )
    # (4) End effector near objects #TODO: Add this
        # end_effector_distance = RewTerm(func=my_mdp.end_effector_box_dist, weight=-1.0)
    # (5) Objects in correct pos    #TODO: Add this
        # success = RewTerm(func=my_mdp.task_success, weight=5.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Add box velocity etc #TODO: Add something sensible
        # box_fell = DoneTerm(func=my_mdp.box_fell_off_table)


#
# Environment configuration
#


@configclass
class MyUR10CylinderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UR10 + box manager-based environment."""

    # scene settings
    scene: MyUR10CylinderSceneCfg = MyUR10CylinderSceneCfg(num_envs=16, env_spacing=2.0)
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
        self.episode_length_s = 5.0
        # viewer
        self.viewer.eye = (8.0, 0.0, 5.0)
        # sim
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
