"""Interface constants shared by Isaac Lab training and MuJoCo deployment.

These values mirror:
`/home/xjy/writing/CORL_2026/code/humanoid_rl_gym/deploy/deploy_mujoco/configs/g1.yaml`.
"""

G1_HRLG_JOINT_NAMES = [
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
]

G1_HRLG_DEFAULT_JOINT_POS = [
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
]

G1_HRLG_DEFAULT_JOINT_POS_DICT = dict(zip(G1_HRLG_JOINT_NAMES, G1_HRLG_DEFAULT_JOINT_POS))

G1_HRLG_NUM_OBS = 98
G1_HRLG_NUM_ACTIONS = 29
G1_HRLG_ACTION_SCALE = 0.25
G1_HRLG_ANG_VEL_SCALE = 0.25
G1_HRLG_DOF_POS_SCALE = 1.0
G1_HRLG_DOF_VEL_SCALE = 0.05
G1_HRLG_CMD_SCALE = (2.0, 2.0, 0.25)
G1_HRLG_GAIT_PHASE_CYCLE_S = 0.64

