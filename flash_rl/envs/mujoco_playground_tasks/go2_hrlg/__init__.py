"""HRLG-compatible MuJoCo Playground Go2 task."""

from flash_rl.envs.mujoco_playground_tasks.go2_hrlg.joystick_hrlg import (
    Go2JoystickFlatTerrainHRLG,
    default_config,
)
from flash_rl.envs.mujoco_playground_tasks.go2_hrlg.randomize import (
    event_cfg_domain_randomize,
)

__all__ = [
    "Go2JoystickFlatTerrainHRLG",
    "default_config",
    "event_cfg_domain_randomize",
]

