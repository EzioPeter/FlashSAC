"""Domain randomization for the HRLG-compatible Go2 MuJoCo Playground task."""

from __future__ import annotations

import os

import jax
import jax.numpy as jp
from mujoco import mjx

NUM_MATERIAL_BUCKETS = 64
FLOOR_GEOM_ID = 0
BASE_BODY_ID = 1

STATIC_FRICTION_RANGE = (0.2, 1.25)
DYNAMIC_FRICTION_RANGE = (0.2, 1.0)
JOINT_DEFAULT_POS_RANGE = (-0.01, 0.01)
BASE_MASS_ADDED_RANGE = (-1.0, 1.0)
LINK_MASS_MULTIPLIER_RANGE = (0.9, 1.1)
BASE_COM_RANGE = ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))
PD_STIFFNESS_MULTIPLIER_RANGE = (0.9, 1.1)
PD_DAMPING_MULTIPLIER_RANGE = (0.9, 1.1)
MOTOR_STRENGTH_RANGE = (0.8, 1.2)
ARMATURE_MULTIPLIER_RANGE = (1.0, 1.05)
FRICTIONLOSS_MULTIPLIER_RANGE = (0.9, 1.1)

MOTOR_ZERO_OFFSET_RANGE = (-0.035, 0.035)
ACTION_DELAY_STEP_RANGE = (0, 1)

PUSH_INTERVAL_RANGE_S = (3.0, 8.0)
PUSH_VELOCITY_MIN = (-0.4, -0.4, -0.05, -0.2, -0.2, -0.6)
PUSH_VELOCITY_MAX = (0.4, 0.4, 0.05, 0.2, 0.2, 0.6)


def _range_from_env(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if not value:
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be two comma-separated floats, got: {value}")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound, got: {value}")
    return low, high


def _base_com_range_from_env() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    value = os.environ.get("GO2_DR_BASE_COM_RANGE")
    if not value:
        return BASE_COM_RANGE
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 2:
        low, high = parts
        return ((low, high), (low, high), (low, high))
    if len(parts) == 6:
        return ((parts[0], parts[1]), (parts[2], parts[3]), (parts[4], parts[5]))
    raise ValueError(
        "GO2_DR_BASE_COM_RANGE must be either low,high or "
        f"x_low,x_high,y_low,y_high,z_low,z_high, got: {value}"
    )


def _bucket_uniform(rng: jax.Array, shape: tuple[int, ...], value_range: tuple[float, float]) -> jax.Array:
    bucket = jax.random.randint(rng, shape, minval=0, maxval=NUM_MATERIAL_BUCKETS)
    bucket_center = (bucket.astype(jp.float32) + 0.5) / float(NUM_MATERIAL_BUCKETS)
    return value_range[0] + bucket_center * (value_range[1] - value_range[0])


def event_cfg_domain_randomize(model: mjx.Model, rng: jax.Array) -> tuple[mjx.Model, mjx.Model]:
    """Randomize model fields with ranges close to humanoid_rl_gym defaults."""

    static_friction_range = _range_from_env("GO2_DR_STATIC_FRICTION_RANGE", STATIC_FRICTION_RANGE)
    dynamic_friction_range = _range_from_env("GO2_DR_DYNAMIC_FRICTION_RANGE", DYNAMIC_FRICTION_RANGE)
    joint_default_pos_range = _range_from_env("GO2_DR_JOINT_DEFAULT_POS_RANGE", JOINT_DEFAULT_POS_RANGE)
    base_mass_added_range = _range_from_env("GO2_DR_BASE_MASS_ADDED_RANGE", BASE_MASS_ADDED_RANGE)
    link_mass_multiplier_range = _range_from_env("GO2_DR_LINK_MASS_MULTIPLIER_RANGE", LINK_MASS_MULTIPLIER_RANGE)
    base_com_range = _base_com_range_from_env()
    pd_stiffness_multiplier_range = _range_from_env(
        "GO2_DR_PD_STIFFNESS_MULTIPLIER_RANGE", PD_STIFFNESS_MULTIPLIER_RANGE
    )
    pd_damping_multiplier_range = _range_from_env("GO2_DR_PD_DAMPING_MULTIPLIER_RANGE", PD_DAMPING_MULTIPLIER_RANGE)
    motor_strength_range = _range_from_env("GO2_DR_MOTOR_STRENGTH_RANGE", MOTOR_STRENGTH_RANGE)
    armature_multiplier_range = _range_from_env("GO2_DR_ARMATURE_MULTIPLIER_RANGE", ARMATURE_MULTIPLIER_RANGE)
    frictionloss_multiplier_range = _range_from_env(
        "GO2_DR_FRICTIONLOSS_MULTIPLIER_RANGE", FRICTIONLOSS_MULTIPLIER_RANGE
    )

    @jax.vmap
    def rand_dynamics(one_rng: jax.Array):
        one_rng, key = jax.random.split(one_rng)
        floor_friction = _bucket_uniform(key, (), static_friction_range)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(floor_friction)

        robot_geom_mask = model.geom_bodyid > 0
        one_rng, key = jax.random.split(one_rng)
        static_friction = _bucket_uniform(key, (model.ngeom,), static_friction_range)
        one_rng, key = jax.random.split(one_rng)
        dynamic_friction = _bucket_uniform(key, (model.ngeom,), dynamic_friction_range)
        geom_friction = geom_friction.at[:, 0].set(
            jp.where(robot_geom_mask, static_friction, geom_friction[:, 0])
        )
        geom_friction = geom_friction.at[:, 1].set(
            jp.where(robot_geom_mask, dynamic_friction, geom_friction[:, 1])
        )

        one_rng, key = jax.random.split(one_rng)
        qpos0 = model.qpos0.at[7:].set(
            model.qpos0[7:]
            + jax.random.uniform(
                key,
                shape=(model.nu,),
                minval=joint_default_pos_range[0],
                maxval=joint_default_pos_range[1],
            )
        )

        one_rng, key = jax.random.split(one_rng)
        base_com_offset = jax.random.uniform(
            key,
            shape=(3,),
            minval=jp.array([r[0] for r in base_com_range], dtype=jp.float32),
            maxval=jp.array([r[1] for r in base_com_range], dtype=jp.float32),
        )
        body_ipos = model.body_ipos.at[BASE_BODY_ID].set(model.body_ipos[BASE_BODY_ID] + base_com_offset)

        one_rng, key = jax.random.split(one_rng)
        base_mass_delta = jax.random.uniform(
            key,
            (),
            minval=base_mass_added_range[0],
            maxval=base_mass_added_range[1],
        )
        body_mass = model.body_mass.at[BASE_BODY_ID].set(
            jp.maximum(1e-3, model.body_mass[BASE_BODY_ID] + base_mass_delta)
        )

        one_rng, key = jax.random.split(one_rng)
        link_mass_multiplier = jax.random.uniform(
            key,
            shape=(model.nbody,),
            minval=link_mass_multiplier_range[0],
            maxval=link_mass_multiplier_range[1],
        )
        link_mass_multiplier = jp.where(jp.arange(model.nbody) > BASE_BODY_ID, link_mass_multiplier, 1.0)
        body_mass = body_mass * link_mass_multiplier

        one_rng, key = jax.random.split(one_rng)
        kp_multiplier = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=pd_stiffness_multiplier_range[0],
            maxval=pd_stiffness_multiplier_range[1],
        )
        one_rng, key = jax.random.split(one_rng)
        kd_multiplier = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=pd_damping_multiplier_range[0],
            maxval=pd_damping_multiplier_range[1],
        )
        one_rng, key = jax.random.split(one_rng)
        motor_strength = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=motor_strength_range[0],
            maxval=motor_strength_range[1],
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
            model.actuator_gainprm[:, 0] * kp_multiplier * motor_strength
        )
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
            model.actuator_biasprm[:, 1] * kp_multiplier * motor_strength
        )
        actuator_biasprm = actuator_biasprm.at[:, 2].set(
            model.actuator_biasprm[:, 2] * kd_multiplier * motor_strength
        )
        dof_damping = model.dof_damping.at[6 : 6 + model.nu].set(
            model.dof_damping[6 : 6 + model.nu] * kd_multiplier * motor_strength
        )

        one_rng, key = jax.random.split(one_rng)
        dof_frictionloss = model.dof_frictionloss.at[6 : 6 + model.nu].set(
            model.dof_frictionloss[6 : 6 + model.nu]
            * jax.random.uniform(
                key,
                shape=(model.nu,),
                minval=frictionloss_multiplier_range[0],
                maxval=frictionloss_multiplier_range[1],
            )
        )
        one_rng, key = jax.random.split(one_rng)
        dof_armature = model.dof_armature.at[6 : 6 + model.nu].set(
            model.dof_armature[6 : 6 + model.nu]
            * jax.random.uniform(
                key,
                shape=(model.nu,),
                minval=armature_multiplier_range[0],
                maxval=armature_multiplier_range[1],
            )
        )

        return (
            geom_friction,
            qpos0,
            body_ipos,
            body_mass,
            actuator_gainprm,
            actuator_biasprm,
            dof_damping,
            dof_frictionloss,
            dof_armature,
        )

    (
        geom_friction,
        qpos0,
        body_ipos,
        body_mass,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
        dof_frictionloss,
        dof_armature,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "qpos0": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
            "dof_damping": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
        }
    )
    model = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "qpos0": qpos0,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "dof_damping": dof_damping,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
        }
    )
    return model, in_axes
