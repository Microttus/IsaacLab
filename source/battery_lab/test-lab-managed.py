"""
Author: Martin Økter
Mail: martino@uia.no
University of Agder

This is the first test setup for the simulation training
of disassembly using UR10 and a test box in managed
style
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description='Test lab setup for disassembly using UR10 and a test box managed')
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")

# Parse the arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the onmiverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""And rest? Everything!"""

import torch

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

