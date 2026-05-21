"""HRLG-compatible MuJoCo Playground G1 task."""

from flash_rl.envs.mujoco_playground_tasks.g1_hrlg.joystick_hrlg import (
    G1JoystickFlatTerrainHRLG,
    default_config,
)
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg.randomize import (
    event_cfg_domain_randomize,
)

__all__ = [
    "G1JoystickFlatTerrainHRLG",
    "default_config",
    "event_cfg_domain_randomize",
]
