"""MuJoCo Playground Go2 flat task with humanoid_rl_gym-compatible policy I/O."""

from __future__ import annotations

from typing import Any, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.mjx_env import State

from . import constants as go2
from . import randomize as go2_randomize


def get_assets() -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    path = mjx_env.MENAGERIE_PATH / "unitree_go2"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_scale=go2.ACTION_SCALE,
        soft_joint_pos_limit_factor=0.95,
        event_domain_randomization=False,
        termination_base_height=0.18,
        termination_bad_orientation_angle=0.8,
        domain_randomization_config=config_dict.create(
            motor_zero_offset_enable=True,
            motor_zero_offset_range=[-0.035, 0.035],
            action_delay_enable=True,
            action_delay_step_range=[0, 1],
            push_enable=True,
            push_interval_range=[3.0, 8.0],
            push_linear_velocity_range=[-0.4, 0.4],
            push_vertical_velocity_range=[-0.05, 0.05],
            push_roll_pitch_velocity_range=[-0.2, 0.2],
            push_yaw_velocity_range=[-0.6, 0.6],
        ),
        command_config=config_dict.create(
            initial_step=0,
            a=[0.5, 0.5, 1.0],
            b=[1.0, 1.0, 1.0],
            zero_command_prob=0.10,
            resample_time_s=5.0,
        ),
        push_config=config_dict.create(
            enable=False,
            interval_range=[3.0, 8.0],
            linear_velocity_range=[-0.4, 0.4],
            angular_velocity_range=[-0.6, 0.6],
        ),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
        reward_config=config_dict.create(
            only_positive_rewards=False,
            scales=config_dict.create(
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.2,
                lin_vel_z=-5.0,
                ang_vel_xy=-0.05,
                dof_acc=-3e-7,
                dof_power=-1e-4,
                torques=-1e-4,
                base_height=-20.0,
                action_rate=-0.005,
                action_smoothness=-0.01,
                collision=-1.0,
                dof_pos_limits=-2.0,
                feet_regulation=-0.05,
                hip_to_default=-0.05,
                similar_to_default=-0.05,
                feet_air_time=1.0,
                termination=-0.0,
            ),
            tracking_sigma=0.25,
            base_height_target=0.38,
            feet_air_time_threshold=0.5,
        ),
    )


class Go2JoystickFlatTerrainHRLG(mjx_env.MjxEnv):
    """Flat-terrain Unitree Go2 joystick task using HRLG deploy obs/action."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)
        xml_path = mjx_env.MENAGERIE_PATH / "unitree_go2" / "scene_mjx.xml"
        self._xml_path = xml_path.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(epath.Path(self._xml_path).read_text(), assets=get_assets())
        self._mj_model.opt.timestep = self._config.sim_dt
        self._policy_to_actuator_np = np.asarray(go2.POLICY_TO_ACTUATOR, dtype=np.int32)
        self._actuator_to_policy_np = np.asarray(go2.ACTUATOR_TO_POLICY, dtype=np.int32)
        self._policy_to_actuator = jp.asarray(go2.POLICY_TO_ACTUATOR, dtype=jp.int32)
        self._actuator_to_policy = jp.asarray(go2.ACTUATOR_TO_POLICY, dtype=jp.int32)
        self._default_pose = jp.asarray(go2.DEFAULT_ANGLES, dtype=jp.float32)
        self._cmd_scale = jp.asarray(go2.CMD_SCALE, dtype=jp.float32)
        self._apply_real_world_pd_gains()
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.asarray(self._mj_model.qpos0, dtype=jp.float32)
        self._init_q = self._init_q.at[2].set(go2.DEFAULT_BASE_HEIGHT)
        self._init_q = self._init_q.at[7:].set(self._default_pose)

        lowers, uppers = self._mj_model.jnt_range[1:].T
        self._soft_lowers = jp.asarray(lowers * self._config.soft_joint_pos_limit_factor, dtype=jp.float32)
        self._soft_uppers = jp.asarray(uppers * self._config.soft_joint_pos_limit_factor, dtype=jp.float32)

        self._base_body_id = self._mj_model.body(go2.ROOT_BODY).id
        self._base_mass = float(self._mj_model.body_subtreemass[self._base_body_id])
        self._imu_site_id = self._mj_model.site(go2.IMU_SITE).id
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.asarray([self._mj_model.geom(name).id for name in go2.FEET_GEOMS], dtype=np.int32)
        self._feet_site_id = np.asarray([self._mj_model.site(name).id for name in go2.FEET_SITES], dtype=np.int32)
        self._feet_body_id = jp.asarray([self._mj_model.body(name).id for name in go2.FEET_BODY_NAMES], dtype=jp.int32)
        foot_geom_ids = set(int(geom_id) for geom_id in self._feet_geom_id)
        base_geom_ids: list[int] = []
        collision_geom_ids: list[int] = []
        for geom_id in range(self._mj_model.ngeom):
            if geom_id == self._floor_geom_id or geom_id in foot_geom_ids:
                continue
            body_name = mujoco.mj_id2name(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self._mj_model.geom_bodyid[geom_id]),
            )
            is_contact_geom = self._mj_model.geom_contype[geom_id] != 0 or self._mj_model.geom_conaffinity[geom_id] != 0
            if is_contact_geom and body_name == go2.ROOT_BODY:
                base_geom_ids.append(geom_id)
            if is_contact_geom and any(token in body_name for token in ("_thigh", "_calf")):
                collision_geom_ids.append(geom_id)
        self._base_geom_id = np.asarray(base_geom_ids, dtype=np.int32)
        self._collision_geom_id = np.asarray(collision_geom_ids, dtype=np.int32)
        self._cmd_a = jp.asarray(self._config.command_config.a, dtype=jp.float32)
        self._cmd_b = jp.asarray(self._config.command_config.b, dtype=jp.float32)
        self._validate_go2_joint_order()

    def _apply_real_world_pd_gains(self) -> None:
        kp_policy = np.asarray(go2.KP, dtype=np.float64)
        kd_policy = np.asarray(go2.KD, dtype=np.float64)
        kp_actuator = kp_policy[self._actuator_to_policy_np]
        kd_actuator = kd_policy[self._actuator_to_policy_np]
        self._mj_model.actuator_gainprm[:, 0] = kp_actuator
        self._mj_model.actuator_biasprm[:, 1] = -kp_actuator
        self._mj_model.actuator_biasprm[:, 2] = -kd_actuator
        self._mj_model.dof_damping[6 : 6 + go2.ACTION_SIZE] = kd_policy

    def _validate_go2_joint_order(self) -> None:
        if self._mj_model.nu != go2.ACTION_SIZE:
            raise ValueError(f"Expected {go2.ACTION_SIZE} Go2 actuators, got {self._mj_model.nu}.")
        for index, joint_name in enumerate(go2.POLICY_JOINT_NAMES):
            joint = self._mj_model.joint(joint_name)
            qpos_index = int(joint.qposadr[0]) - 7
            dof_index = int(joint.dofadr[0]) - 6
            if qpos_index != index or dof_index != index:
                raise ValueError(
                    f"Joint order mismatch for {joint_name}: "
                    f"qpos_index={qpos_index}, dof_index={dof_index}, expected={index}."
                )
        for actuator_id, actuator_name in enumerate(go2.ACTUATOR_NAMES):
            actual = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if actual != actuator_name:
                raise ValueError(f"Actuator order mismatch at {actuator_id}: {actual} != {actuator_name}.")
            joint_id = int(self._mj_model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            expected_policy_index = go2.ACTUATOR_TO_POLICY[actuator_id]
            expected_joint = go2.POLICY_JOINT_NAMES[expected_policy_index]
            if joint_name != expected_joint:
                raise ValueError(
                    f"Actuator {actuator_name} controls {joint_name}, expected {expected_joint}."
                )

    def reset(self, rng: jax.Array) -> State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-jp.pi, maxval=jp.pi)
        quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

        rng, key = jax.random.split(rng)
        joint_noise = jax.random.uniform(key, (go2.ACTION_SIZE,), minval=-0.05, maxval=0.05)
        qpos = qpos.at[7:].set(self._default_pose + joint_noise)

        rng, key = jax.random.split(rng)
        qvel = qvel.at[6:].set(jax.random.uniform(key, (go2.ACTION_SIZE,), minval=-1.0, maxval=1.0))

        ctrl = self._default_pose[self._policy_to_actuator]
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

        initial_step = jp.asarray(self._config.command_config.initial_step, dtype=jp.int32)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng, jp.zeros(3), initial_step)
        rng, key = jax.random.split(rng)
        steps_until_next_cmd = jp.round(
            self._config.command_config.resample_time_s / self.dt
        ).astype(jp.int32)
        steps_until_next_cmd = jp.maximum(steps_until_next_cmd, jp.array(1, dtype=jp.int32))

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=go2_randomize.PUSH_INTERVAL_RANGE_S[0],
            maxval=go2_randomize.PUSH_INTERVAL_RANGE_S[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        if (
            self._config.event_domain_randomization
            and self._config.domain_randomization_config.motor_zero_offset_enable
        ):
            rng, zero_offset_rng, action_delay_rng = jax.random.split(rng, 3)
            zero_offset_range = self._config.domain_randomization_config.motor_zero_offset_range
            motor_zero_offset = jax.random.uniform(
                zero_offset_rng,
                shape=(self.mjx_model.nu,),
                minval=zero_offset_range[0],
                maxval=zero_offset_range[1],
            )
        else:
            rng, action_delay_rng = jax.random.split(rng)
            motor_zero_offset = jp.zeros(self.mjx_model.nu)

        if self._config.event_domain_randomization and self._config.domain_randomization_config.action_delay_enable:
            delay_range = self._config.domain_randomization_config.action_delay_step_range
            action_delay_steps = jax.random.randint(
                action_delay_rng,
                shape=(),
                minval=delay_range[0],
                maxval=delay_range[1] + 1,
                dtype=jp.int32,
            )
        else:
            action_delay_steps = jp.array(0, dtype=jp.int32)

        info = {
            "rng": rng,
            "step": initial_step,
            "policy_step": jp.array(0, dtype=jp.int32),
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "applied_action": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_pose,
            "motor_zero_offset": motor_zero_offset,
            "action_delay_steps": action_delay_steps,
            "feet_air_time": jp.zeros(4),
            "feet_contact_time": jp.zeros(4),
            "last_air_time": jp.zeros(4),
            "last_contact_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "last_feet_pos": data.site_xpos[self._feet_site_id],
            "last_joint_vel": data.qvel[6:],
            "push": jp.zeros(6),
            "push_step": jp.array(0, dtype=jp.int32),
            "push_interval_steps": push_interval_steps,
        }
        metrics = {f"reward/{key}": jp.zeros(()) for key in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.asarray([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, contact)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        state = self._maybe_apply_push(state)

        action_delay_steps = state.info.get("action_delay_steps", jp.array(0, dtype=jp.int32))
        applied_action = jp.where(action_delay_steps > 0, state.info["last_act"], action)
        motor_zero_offset = state.info.get("motor_zero_offset", jp.zeros(self.mjx_model.nu))
        motor_targets_policy = self._default_pose + motor_zero_offset + applied_action * self._config.action_scale
        motor_targets_actuator = motor_targets_policy[self._policy_to_actuator]
        data = mjx_env.step(self.mjx_model, state.data, motor_targets_actuator, self.n_substeps)

        contact = jp.asarray([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) & contact_filt
        first_air = (~contact) & state.info["last_contact"]
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = (feet_pos - state.info["last_feet_pos"]) / self.dt
        foot_z = feet_pos[:, 2]
        feet_air_time = state.info["feet_air_time"] + self.dt
        feet_contact_time = state.info["feet_contact_time"] + self.dt
        last_air_time = jp.where(first_contact, feet_air_time, state.info["last_air_time"])
        last_contact_time = jp.where(first_air, feet_contact_time, state.info["last_contact_time"])
        swing_peak = jp.maximum(state.info["swing_peak"], foot_z)
        reward_info = dict(state.info)
        reward_info["feet_air_time"] = feet_air_time
        reward_info["feet_contact_time"] = feet_contact_time
        reward_info["last_air_time"] = last_air_time
        reward_info["last_contact_time"] = last_contact_time
        reward_info["swing_peak"] = swing_peak

        done = self._get_termination(data)
        rewards = self._get_reward(data, action, reward_info, done, first_contact, contact, feet_vel)
        rewards = {
            key: value * self._reward_scale(key, reward_info["step"]) for key, value in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt
        reward = jp.where(self._config.reward_config.only_positive_rewards, jp.clip(reward, min=0.0), reward)

        state.info["step"] += 1
        state.info["policy_step"] = jp.where(done, 0, state.info["policy_step"] + 1)
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["applied_action"] = applied_action
        state.info["motor_targets"] = motor_targets_policy
        state.info["feet_air_time"] = feet_air_time * ~contact
        state.info["feet_contact_time"] = feet_contact_time * contact
        state.info["last_air_time"] = last_air_time
        state.info["last_contact_time"] = last_contact_time
        state.info["last_contact"] = contact
        state.info["swing_peak"] = swing_peak * ~contact
        state.info["last_feet_pos"] = feet_pos
        state.info["last_joint_vel"] = data.qvel[6:]
        state.info["rng"], cmd_rng, time_rng = jax.random.split(state.info["rng"], 3)
        should_resample = state.info["steps_until_next_cmd"] <= 0
        state.info["command"] = jp.where(
            should_resample,
            self.sample_command(cmd_rng, state.info["command"], state.info["step"]),
            state.info["command"],
        )
        del time_rng
        new_cmd_steps = jp.round(self._config.command_config.resample_time_s / self.dt).astype(jp.int32)
        state.info["steps_until_next_cmd"] = jp.where(
            done | should_resample,
            jp.maximum(new_cmd_steps, jp.array(1, dtype=jp.int32)),
            state.info["steps_until_next_cmd"] - 1,
        )
        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        state = state.replace(data=data, reward=reward, done=done.astype(reward.dtype))
        obs = self._get_obs(data, state.info, contact)
        return state.replace(obs=obs)

    def _maybe_apply_push(self, state: State) -> State:
        if self._config.event_domain_randomization:
            dr_cfg = self._config.domain_randomization_config
            linear = dr_cfg.push_linear_velocity_range
            vertical = dr_cfg.push_vertical_velocity_range
            roll_pitch = dr_cfg.push_roll_pitch_velocity_range
            yaw = dr_cfg.push_yaw_velocity_range
            min_vel = jp.asarray(
                [linear[0], linear[0], vertical[0], roll_pitch[0], roll_pitch[0], yaw[0]],
                dtype=jp.float32,
            )
            max_vel = jp.asarray(
                [linear[1], linear[1], vertical[1], roll_pitch[1], roll_pitch[1], yaw[1]],
                dtype=jp.float32,
            )
            interval_range = dr_cfg.push_interval_range
            enabled = bool(dr_cfg.push_enable)
        else:
            linear = self._config.push_config.linear_velocity_range
            angular = self._config.push_config.angular_velocity_range
            min_vel = jp.asarray([linear[0], linear[0], 0.0, 0.0, 0.0, angular[0]], dtype=jp.float32)
            max_vel = jp.asarray([linear[1], linear[1], 0.0, 0.0, 0.0, angular[1]], dtype=jp.float32)
            interval_range = self._config.push_config.interval_range
            enabled = bool(self._config.push_config.enable)

        state.info["rng"], push_rng, interval_rng = jax.random.split(state.info["rng"], 3)
        push_step = state.info["push_step"] + 1
        should_push = (jp.mod(push_step, state.info["push_interval_steps"]) == 0) & enabled
        push_velocity = jax.random.uniform(push_rng, shape=(6,), minval=min_vel, maxval=max_vel)
        qvel = state.data.qvel.at[:6].set(state.data.qvel[:6] + push_velocity * should_push)
        data = state.data.replace(qvel=qvel)
        new_interval = jax.random.uniform(
            interval_rng,
            minval=interval_range[0],
            maxval=interval_range[1],
        )
        state.info["push_interval_steps"] = jp.where(
            should_push,
            jp.round(new_interval / self.dt).astype(jp.int32),
            state.info["push_interval_steps"],
        )
        state.info["push_step"] = jp.where(should_push, 0, push_step)
        state.info["push"] = push_velocity * should_push
        return state.replace(data=data)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array):
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = gyro + (
            (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = gravity + (
            (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = joint_angles + (
            (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = joint_vel + (
            (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        state = jp.hstack(
            [
                noisy_gyro * go2.ANG_VEL_SCALE,
                noisy_gravity,
                info["command"] * self._cmd_scale,
                (noisy_joint_angles - self._default_pose) * go2.DOF_POS_SCALE,
                noisy_joint_vel * go2.DOF_VEL_SCALE,
                info["last_act"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        linvel = self.get_local_linvel(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = self.get_feet_linvel(data).ravel()
        actuator_force_policy = data.actuator_force[self._policy_to_actuator]
        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                linvel,
                global_angvel,
                joint_angles - self._default_pose,
                joint_vel,
                data.qpos[2:3],
                actuator_force_policy,
                contact,
                feet_vel,
                info["feet_air_time"],
                info["push"],
                info["motor_zero_offset"],
            ]
        )
        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
        feet_vel: jax.Array,
    ) -> dict[str, jax.Array]:
        actuator_force_policy = data.actuator_force[self._policy_to_actuator]
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data)),
            "tracking_ang_vel": self._reward_tracking_ang_vel(info["command"], self.get_gyro(data)),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "dof_acc": self._cost_dof_acc(data.qvel[6:], info["last_joint_vel"]),
            "dof_power": self._cost_dof_power(data.qvel[6:], actuator_force_policy),
            "torques": self._cost_torques(actuator_force_policy),
            "base_height": self._cost_base_height(data),
            "action_rate": self._cost_action_rate(action, info["last_act"]),
            "action_smoothness": self._cost_action_smoothness(action, info["last_act"], info["last_last_act"]),
            "collision": self._cost_collision(data),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "feet_regulation": self._cost_feet_regulation(data, feet_vel),
            "hip_to_default": self._cost_hip_to_default(data.qpos[7:]),
            "similar_to_default": self._cost_similar_to_default(data.qpos[7:]),
            "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"]),
            "termination": done,
        }

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        gravity = self.get_gravity(data)
        up_z = -gravity[2]
        bad_orientation = up_z < jp.cos(self._config.termination_bad_orientation_angle)
        low_height = data.qpos[2] < self._config.termination_base_height
        base_contact = self._has_base_contact(data)
        nonfinite = ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        return bad_orientation | low_height | base_contact | nonfinite

    def _reward_tracking_lin_vel(self, commands: jax.Array, local_vel: jax.Array) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands: jax.Array, ang_vel: jax.Array) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torques))

    def _cost_dof_power(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_base_height(self, data: mjx.Data) -> jax.Array:
        return jp.square(data.qpos[2] - self._config.reward_config.base_height_target)

    def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(act - last_act))

    def _cost_action_smoothness(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(act - 2.0 * last_act + last_last_act))

    def _cost_dof_acc(self, qvel: jax.Array, last_qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square((qvel - last_qvel) / self.dt))

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_collision(self, data: mjx.Data) -> jax.Array:
        collisions = jp.asarray(
            [geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._collision_geom_id]
        )
        return jp.sum(collisions)

    def _cost_feet_regulation(self, data: mjx.Data, feet_vel: jax.Array) -> jax.Array:
        foot_z = data.site_xpos[self._feet_site_id, 2]
        base_height = jp.maximum(data.qpos[2], 1e-3)
        feet_height = jp.clip(foot_z, min=0.0)
        feet_xy_speed_sq = jp.sum(jp.square(feet_vel[:, :2]), axis=-1)
        denom = jp.maximum(0.025 * self._config.reward_config.base_height_target, 1e-6)
        return jp.sum(feet_xy_speed_sq * jp.exp(-feet_height / denom)) * (base_height > 0.0)

    def _cost_hip_to_default(self, qpos: jax.Array) -> jax.Array:
        hip_indices = jp.asarray([0, 3, 6, 9], dtype=jp.int32)
        return jp.sum(jp.abs(qpos[hip_indices] - self._default_pose[hip_indices]))

    def _cost_similar_to_default(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qpos - self._default_pose))

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        threshold = self._config.reward_config.feet_air_time_threshold
        rew_air_time = jp.sum((air_time - threshold) * first_contact)
        return rew_air_time * (jp.linalg.norm(commands) > 0.01)

    def _reward_scale(self, name: str, step: jax.Array) -> jax.Array:
        del step
        scale = jp.asarray(self._config.reward_config.scales[name], dtype=jp.float32)
        return scale

    def _has_base_contact(self, data: mjx.Data) -> jax.Array:
        collisions = jp.asarray(
            [geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._base_geom_id]
        )
        return jp.any(collisions)

    def sample_command(self, rng: jax.Array, current_command: jax.Array, step: jax.Array) -> jax.Array:
        del current_command, step
        rng, cmd_rng, mask_rng, zero_rng = jax.random.split(rng, 4)
        cmd_a = self._cmd_a
        cmd = jax.random.uniform(cmd_rng, shape=(3,), minval=-cmd_a, maxval=cmd_a)
        axis_mask = jax.random.bernoulli(mask_rng, self._cmd_b, shape=(3,))
        zero_all = jax.random.bernoulli(zero_rng, self._config.command_config.zero_command_prob)
        return jp.where(zero_all, jp.zeros(3), cmd * axis_mask)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.asarray([0.0, 0.0, -1.0], dtype=jp.float32)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "global_linvel")

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "global_angvel")

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ self.get_global_linvel(data)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "accelerometer")

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "gyro")

    def get_feet_linvel(self, data: mjx.Data) -> jax.Array:
        return data.cvel[self._feet_body_id, 3:]

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
