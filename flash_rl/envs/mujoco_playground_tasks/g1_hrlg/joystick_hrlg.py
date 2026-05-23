"""MuJoCo Playground G1 task with humanoid_rl_gym-compatible policy I/O."""

from __future__ import annotations

from typing import Any, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.mjx_env import State

from . import constants as hrlg
from . import randomize as hrlg_randomize


def default_config():
    """Return the original MJP G1 config with HRLG deploy interface values."""
    cfg = g1_joystick.default_config()
    cfg.action_scale = hrlg.ACTION_SCALE
    cfg.hrlg_gait_phase_cycle = hrlg.GAIT_PHASE_CYCLE
    cfg.event_domain_randomization = False
    return cfg


class G1JoystickFlatTerrainHRLG(g1_joystick.Joystick):
    """Flat-terrain G1 joystick task whose actor obs/action match HRLG deploy."""

    def __init__(
        self,
        config=None,
        config_overrides: Optional[dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task="flat_terrain",
            config=config or default_config(),
            config_overrides=config_overrides,
        )

    def _post_init(self) -> None:
        super()._post_init()
        self._default_pose = jp.array(hrlg.DEFAULT_ANGLES, dtype=jp.float32)
        self._init_q = self._init_q.at[2].set(hrlg.DEFAULT_BASE_HEIGHT)
        self._init_q = self._init_q.at[7:].set(self._default_pose)
        self._base_mjx_qpos0 = jp.array(self.mjx_model.qpos0)
        self._cmd_scale = jp.array(hrlg.CMD_SCALE, dtype=jp.float32)
        self._apply_real_world_pd_gains()
        self._validate_hrlg_joint_order()

    def _joint_default_pose(self) -> jax.Array:
        return self._default_pose + (self.mjx_model.qpos0[7:] - self._base_mjx_qpos0[7:])

    def _apply_real_world_pd_gains(self) -> None:
        joint_names = hrlg.LEG_JOINT_NAMES + hrlg.WAIST_JOINT_NAMES + hrlg.ARM_JOINT_NAMES
        stiffness = np.asarray(
            hrlg.LEG_STIFFNESS + hrlg.WAIST_STIFFNESS + hrlg.ARM_STIFFNESS,
            dtype=np.float64,
        )
        damping = np.asarray(
            hrlg.LEG_DAMPING + hrlg.WAIST_DAMPING + hrlg.ARM_DAMPING,
            dtype=np.float64,
        )
        if stiffness.shape != (len(joint_names),) or damping.shape != (len(joint_names),):
            raise ValueError(
                f"Expected {len(joint_names)} real-world PD gains, got "
                f"stiffness={stiffness.shape}, damping={damping.shape}."
            )

        for joint_name, kp, kd in zip(joint_names, stiffness, damping):
            joint = self._mj_model.joint(joint_name)
            actuator_id = hrlg.JOINT_NAMES.index(joint_name)
            dof_id = int(joint.dofadr[0])
            self._mj_model.actuator_gainprm[actuator_id, 0] = kp
            self._mj_model.actuator_biasprm[actuator_id, 1] = -kp
            self._mj_model.dof_damping[dof_id] = kd

        self._mjx_model = mjx.put_model(self._mj_model)

    def _validate_hrlg_joint_order(self) -> None:
        if self.mjx_model.nu != hrlg.ACTION_SIZE:
            raise ValueError(f"Expected {hrlg.ACTION_SIZE} actuators, got {self.mjx_model.nu}.")

        for index, joint_name in enumerate(hrlg.JOINT_NAMES):
            joint = self._mj_model.joint(joint_name)
            qpos_index = int(joint.qposadr[0]) - 7
            dof_index = int(joint.dofadr[0]) - 6
            if qpos_index != index or dof_index != index:
                raise ValueError(
                    f"Joint order mismatch for {joint_name}: "
                    f"qpos_index={qpos_index}, dof_index={dof_index}, expected={index}."
                )
            actuator_joint_id = int(self._mj_model.actuator_trnid[index, 0])
            if actuator_joint_id != joint.id:
                raise ValueError(
                    f"Actuator order mismatch at index {index}: "
                    f"actuator joint id={actuator_joint_id}, expected {joint.id} ({joint_name})."
                )

    def reset(self, rng: jax.Array) -> State:
        joint_default_pose = self._joint_default_pose()

        qpos = self._init_q.at[7:].set(joint_default_pose)
        qvel = jp.zeros(self.mjx_model.nv)

        # Match MuJoCo Playground G1 reset randomization.
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5))

        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0, jp.pi])

        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        rng, event_push_rng = jax.random.split(rng)
        event_push_interval = jax.random.uniform(
            event_push_rng,
            minval=hrlg_randomize.PUSH_INTERVAL_RANGE_S[0],
            maxval=hrlg_randomize.PUSH_INTERVAL_RANGE_S[1],
        )
        event_push_interval_steps = jp.round(event_push_interval / self.dt).astype(jp.int32)

        if self._config.event_domain_randomization:
            rng, zero_offset_rng, action_delay_rng = jax.random.split(rng, 3)
            motor_zero_offset = jax.random.uniform(
                zero_offset_rng,
                shape=(self.mjx_model.nu,),
                minval=hrlg_randomize.MOTOR_ZERO_OFFSET_RANGE[0],
                maxval=hrlg_randomize.MOTOR_ZERO_OFFSET_RANGE[1],
            )
            action_delay_steps = jax.random.randint(
                action_delay_rng,
                shape=(),
                minval=hrlg_randomize.ACTION_DELAY_STEP_RANGE[0],
                maxval=hrlg_randomize.ACTION_DELAY_STEP_RANGE[1] + 1,
                dtype=jp.int32,
            )
        else:
            motor_zero_offset = jp.zeros(self.mjx_model.nu)
            action_delay_steps = jp.array(0, dtype=jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "applied_action": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "motor_zero_offset": motor_zero_offset,
            "action_delay_steps": action_delay_steps,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "phase_dt": phase_dt,
            "phase": phase,
            "push": jp.zeros(6),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "event_push_step": 0,
            "event_push_interval_steps": event_push_interval_steps,
            "joint_default_pose": joint_default_pose,
            "policy_step": jp.array(0, dtype=jp.int32),
        }

        metrics = {}
        for key in self._config.reward_config.scales.keys():
            metrics[f"reward/{key}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        state = mjx_env.State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        previous_policy_step = state.info.get("policy_step", state.info["step"])
        if self._config.event_domain_randomization:
            state.info["rng"], push_rng, interval_rng = jax.random.split(state.info["rng"], 3)
            event_push_step = state.info["event_push_step"] + 1
            should_push = jp.mod(event_push_step, state.info["event_push_interval_steps"]) == 0
            push_velocity = jax.random.uniform(
                push_rng,
                shape=(6,),
                minval=jp.array(hrlg_randomize.PUSH_VELOCITY_MIN, dtype=jp.float32),
                maxval=jp.array(hrlg_randomize.PUSH_VELOCITY_MAX, dtype=jp.float32),
            )
            qvel = state.data.qvel.at[:6].set(state.data.qvel[:6] + push_velocity * should_push)
            data = state.data.replace(qvel=qvel)
            state = state.replace(data=data)
            new_interval = jax.random.uniform(
                interval_rng,
                minval=hrlg_randomize.PUSH_INTERVAL_RANGE_S[0],
                maxval=hrlg_randomize.PUSH_INTERVAL_RANGE_S[1],
            )
            state.info["event_push_interval_steps"] = jp.where(
                should_push,
                jp.round(new_interval / self.dt).astype(jp.int32),
                state.info["event_push_interval_steps"],
            )
            state.info["event_push_step"] = jp.where(should_push, 0, event_push_step)
            push = push_velocity * should_push
        else:
            state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)
            push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
            push_magnitude = jax.random.uniform(
                push2_rng,
                minval=self._config.push_config.magnitude_range[0],
                maxval=self._config.push_config.magnitude_range[1],
            )
            push_xy = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
            push_xy *= jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
            push_xy *= self._config.push_config.enable
            qvel = state.data.qvel.at[:2].set(push_xy * push_magnitude + state.data.qvel[:2])
            data = state.data.replace(qvel=qvel)
            state = state.replace(data=data)
            push = jp.hstack([push_xy * push_magnitude, jp.zeros(4)])

        joint_default_pose = state.info.get("joint_default_pose", self._default_pose)
        action_delay_steps = state.info.get("action_delay_steps", jp.array(0, dtype=jp.int32))
        applied_action = jp.where(action_delay_steps > 0, state.info["last_act"], action)
        motor_zero_offset = state.info.get("motor_zero_offset", jp.zeros(self.mjx_model.nu))
        motor_targets = joint_default_pose + motor_zero_offset + applied_action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["applied_action"] = applied_action
        state.info["motor_targets"] = motor_targets

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, state.metrics, done, first_contact, contact)
        rewards = {key: value * self._config.reward_config.scales[key] for key, value in rewards.items()}
        reward = sum(rewards.values()) * self.dt

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, reward=reward, done=done)
        state.info["policy_step"] = jp.where(state.done, 0, previous_policy_step + 1)

        contact = jp.array(
            [geoms_colliding(state.data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id]
        )
        obs = self._get_obs(state.data, state.info, contact)
        return state.replace(obs=obs)

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        contact: jax.Array,
    ):
        gyro = self.get_gyro(data, "pelvis")
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = gyro + (
            (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0.0, 0.0, -1.0])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = gravity + (
            (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        joint_default_pose = info.get("joint_default_pose", self._default_pose)
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

        policy_step = jp.asarray(info.get("policy_step", info["step"]), dtype=jp.float32)
        gait_phase = policy_step * self.dt / hrlg.GAIT_PHASE_CYCLE
        gait_angle = 2.0 * jp.pi * gait_phase

        state = jp.hstack(
            [
                noisy_gyro * hrlg.ANG_VEL_SCALE,
                noisy_gravity,
                info["command"] * self._cmd_scale,
                (noisy_joint_angles - joint_default_pose) * hrlg.DOF_POS_SCALE,
                noisy_joint_vel * hrlg.DOF_VEL_SCALE,
                info["last_act"],
                jp.array([jp.sin(gait_angle), jp.cos(gait_angle)]),
            ]
        )

        accelerometer = self.get_accelerometer(data, "pelvis")
        linvel = self.get_local_linvel(data, "pelvis")
        global_angvel = self.get_global_angvel(data, "pelvis")
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                linvel,
                global_angvel,
                joint_angles - joint_default_pose,
                joint_vel,
                root_height,
                data.actuator_force,
                contact,
                feet_vel,
                info["feet_air_time"],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }
