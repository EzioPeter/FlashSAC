"""Constants matching humanoid_rl_gym's G1 MuJoCo deploy interface."""

ENV_NAME = "G1JoystickFlatTerrainHRLG"

POLICY_OBS_SIZE = 98
ACTION_SIZE = 29
ACTION_SCALE = 0.25
GAIT_PHASE_CYCLE = 0.64

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = (2.0, 2.0, 0.25)

DEFAULT_BASE_HEIGHT = 0.793

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_ANGLES = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25,
    0.0,
    0.97,
    0.15,
    0.0,
    0.0,
    0.0,
    -0.25,
    0.0,
    0.97,
    -0.15,
    0.0,
    0.0,
)

WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

WAIST_STIFFNESS = (
    40.1792,
    28.5012,
    28.5012,
)

WAIST_DAMPING = (
    2.5579,
    1.8144,
    1.8144,
)
