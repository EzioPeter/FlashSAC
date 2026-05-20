"""29-DoF Unitree G1 task aligned with humanoid_rl_gym MuJoCo deployment."""

import gymnasium as gym

gym.register(
    id="FlashSAC-Isaac-G1-HRLG-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1HRLGFlatEnvCfg",
    },
)

