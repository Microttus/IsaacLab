"""
Author: Martin Økter
Mail: martino@uia.no
University of Agder

This is the first test environment for the simulation training
of disassembly using UR10 and a test box
"""

import math
import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
import omni.isaac.lab.envs import MaagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

import omni.lab_tasks.manager_based.clasic

##
# Pre-defined configs
##

from omni.isaac.lab_assets import UR10_CFG

##
# Scene definition
##

@configclass
class DisassemblyTestEnv(object):
    """
    Configuration class for the disassembly test environment.
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot arm
    robot: ArticulationCfg = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")