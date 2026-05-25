"""Constants for the HRLG-compatible Unitree Go2 MuJoCo Playground task."""

from __future__ import annotations

ACTION_SIZE = 12
POLICY_OBS_SIZE = 45
OBS_SIZE = POLICY_OBS_SIZE

ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = (3.0, 2.0, 0.5)

DEFAULT_BASE_HEIGHT = 0.445
DEFAULT_ANGLES = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)

# Policy/qpos/qvel order, matching humanoid_rl_gym Go2 deployment config.
POLICY_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

# MuJoCo actuator order in unitree_go2/go2_mjx.xml.
ACTUATOR_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)

POLICY_TO_ACTUATOR = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
ACTUATOR_TO_POLICY = tuple(POLICY_TO_ACTUATOR.index(i) for i in range(ACTION_SIZE))

KP = (20.0,) * ACTION_SIZE
KD = (0.5,) * ACTION_SIZE

FEET_GEOMS = ("FL", "FR", "RL", "RR")
FEET_SITES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FEET_BODY_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
ROOT_BODY = "base"
IMU_SITE = "imu"

