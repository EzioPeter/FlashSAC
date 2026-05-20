"""Isaac Lab flat-terrain G1 task matching humanoid_rl_gym's MuJoCo G1 policy IO."""

from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

from . import mdp as hrlg_mdp
from .interface import (
    G1_HRLG_ACTION_SCALE,
    G1_HRLG_ANG_VEL_SCALE,
    G1_HRLG_CMD_SCALE,
    G1_HRLG_DEFAULT_JOINT_POS_DICT,
    G1_HRLG_DOF_POS_SCALE,
    G1_HRLG_DOF_VEL_SCALE,
    G1_HRLG_GAIT_PHASE_CYCLE_S,
    G1_HRLG_JOINT_NAMES,
)


def _joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=G1_HRLG_JOINT_NAMES, preserve_order=True)


@configclass
class G1HRLGActionsCfg(ActionsCfg):
    """29 actions in humanoid_rl_gym joint order."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_HRLG_JOINT_NAMES,
        scale=G1_HRLG_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class G1HRLGObservationsCfg(ObservationsCfg):
    """98-D policy observation matching deploy_mujoco/deploy_g1.py."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=hrlg_mdp.scaled_base_ang_vel,
            params={"scale": G1_HRLG_ANG_VEL_SCALE},
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=hrlg_mdp.scaled_generated_commands,
            params={"command_name": "base_velocity", "scale": G1_HRLG_CMD_SCALE},
        )
        joint_pos = ObsTerm(
            func=hrlg_mdp.scaled_joint_pos_rel,
            params={"asset_cfg": _joint_cfg(), "scale": G1_HRLG_DOF_POS_SCALE},
        )
        joint_vel = ObsTerm(
            func=hrlg_mdp.scaled_joint_vel_rel,
            params={"asset_cfg": _joint_cfg(), "scale": G1_HRLG_DOF_VEL_SCALE},
        )
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(
            func=hrlg_mdp.gait_phase,
            params={"cycle_time": G1_HRLG_GAIT_PHASE_CYCLE_S},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1HRLGRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"])},
    )


@configclass
class G1HRLGTerminationsCfg(TerminationsCfg):
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class G1HRLGFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: G1HRLGObservationsCfg = G1HRLGObservationsCfg()
    actions: G1HRLGActionsCfg = G1HRLGActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: G1HRLGRewardsCfg = G1HRLGRewardsCfg()
    terminations: G1HRLGTerminationsCfg = G1HRLGTerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        robot_cfg = G1_29DOF_CFG.copy().replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.spawn.activate_contact_sensors = True
        robot_cfg.init_state.joint_pos = G1_HRLG_DEFAULT_JOINT_POS_DICT.copy()
        robot_cfg.init_state.joint_vel = {".*": 0.0}
        self.scene.robot = robot_cfg

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = None

        self.rewards.undesired_contacts = None

        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.physics_material.params["asset_cfg"].body_names = ".*"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
