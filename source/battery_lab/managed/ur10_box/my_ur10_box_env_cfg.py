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
from isaaclab.utils import configclass

#UR10
from isaaclab_assets import UR10_CFG


import source.battery_lab.managed.ur10_box.mdp as mdp


@configclass
class MyUR10BoxSceneCfg(InteractiveSceneCfg):
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

    # box assembly (single asset with multiple sub-objects inside)
    box_parts = [
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_0",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/rhino/IsaacLab/source/battery-lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_1",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/rhino/IsaacLab/source/battery-lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_2",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/rhino/IsaacLab/source/battery-lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_3",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/rhino/IsaacLab/source/battery-lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/BoxAssembly/Part_4",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/rhino/IsaacLab/source/battery-lab/Four_box_ass_2.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        ),
    ]


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

    #TODO: Add more events
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
        func=mdp.reset_box_positions,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("my_box"),  # or sub-objects
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # e.g. keep the end-effector near the box, or keep box stable
    end_effector_distance = RewTerm(func=my_mdp.end_effector_box_dist, weight=-1.0)
    # small penalty for joint velocities
    joint_vel_penalty = RewTerm(func=my_mdp.ur10_joint_vel_penalty, weight=-0.01)
    # bonus for something
    success = RewTerm(func=my_mdp.task_success, weight=5.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=my_mdp.time_out, time_out=True)
    # e.g. if the box falls off the table or something
    box_fell = DoneTerm(func=my_mdp.box_fell_off_table)


#
# Environment configuration
#


@configclass
class MyUR10BoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UR10 + box manager-based environment."""

    # scene settings
    scene: MyUR10BoxSceneCfg = MyUR10BoxSceneCfg(num_envs=16, env_spacing=2.0)
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
