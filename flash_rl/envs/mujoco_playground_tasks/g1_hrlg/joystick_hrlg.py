"""MuJoCo Playground G1 task with humanoid_rl_gym-compatible policy I/O."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.mjx_env import State

from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import constants as hrlg


def default_config():
    """Return the original MJP G1 config with HRLG deploy interface values."""
    cfg = g1_joystick.default_config()
    cfg.action_scale = hrlg.ACTION_SCALE
    cfg.hrlg_gait_phase_cycle = hrlg.GAIT_PHASE_CYCLE
    return cfg


class G1JoystickFlatTerrainHRLG(g1_joystick.Joystick):
    """Flat-terrain G1 joystick task whose actor obs/action match HRLG deploy."""

    def __init__(
        self,
        config=None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
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
        self._cmd_scale = jp.array(hrlg.CMD_SCALE, dtype=jp.float32)
        self._validate_hrlg_joint_order()

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
        state = super().reset(rng)
        state.info["policy_step"] = jp.array(0, dtype=jp.int32)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        previous_policy_step = state.info.get("policy_step", state.info["step"])
        state = super().step(state, action)
        state.info["policy_step"] = jp.where(state.done, 0, previous_policy_step + 1)

        contact = jp.array([
            geoms_colliding(state.data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
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

        state = jp.hstack([
            noisy_gyro * hrlg.ANG_VEL_SCALE,
            noisy_gravity,
            info["command"] * self._cmd_scale,
            (noisy_joint_angles - self._default_pose) * hrlg.DOF_POS_SCALE,
            noisy_joint_vel * hrlg.DOF_VEL_SCALE,
            info["last_act"],
            jp.array([jp.sin(gait_angle), jp.cos(gait_angle)]),
        ])

        accelerometer = self.get_accelerometer(data, "pelvis")
        linvel = self.get_local_linvel(data, "pelvis")
        global_angvel = self.get_global_angvel(data, "pelvis")
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack([
            state,
            gyro,
            accelerometer,
            gravity,
            linvel,
            global_angvel,
            joint_angles - self._default_pose,
            joint_vel,
            root_height,
            data.actuator_force,
            contact,
            feet_vel,
            info["feet_air_time"],
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }
