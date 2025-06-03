"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages
import gymnasium as gym

from . import agents
from .dextrah_kuka_allegro_env import DextrahKukaAllegroEnv
from .dextrah_kuka_allegro_env_cfg import DextrahKukaAllegroEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Dextrah-Kuka-Allegro",
    #entry_point="isaaclab_tasks.direct.shadow_hand:ShadowHandEnv",
    entry_point="dextrah_lab.tasks.dextrah_kuka_allegro.dextrah_kuka_allegro_env:DextrahKukaAllegroEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DextrahKukaAllegroEnvCfg,
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_scratch_cnn_aux.yaml",
        #"rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ShadowHandPPORunnerCfg,
    },
)

# The blacklist is used to prevent importing configs from sub-packages
#_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
#import_packages(__name__, _BLACKLIST_PKGS)
