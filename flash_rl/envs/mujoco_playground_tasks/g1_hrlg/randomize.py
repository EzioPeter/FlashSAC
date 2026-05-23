"""EventCfg-style domain randomization for the HRLG G1 MuJoCo Playground task."""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

NUM_MATERIAL_BUCKETS = 64
BASE_BODY_ID = 1
TORSO_BODY_ID = 16

STATIC_FRICTION_RANGE = (0.3, 1.6)
DYNAMIC_FRICTION_RANGE = (0.3, 1.2)
RESTITUTION_RANGE = (0.0, 0.5)
JOINT_DEFAULT_POS_RANGE = (-0.01, 0.01)
BASE_MASS_ADDED_RANGE = (-1.0, 1.0)
LINK_MASS_MULTIPLIER_RANGE = (0.9, 1.1)
PD_STIFFNESS_MULTIPLIER_RANGE = (0.9, 1.1)
PD_DAMPING_MULTIPLIER_RANGE = (0.9, 1.1)
MOTOR_STRENGTH_RANGE = (0.8, 1.2)
MOTOR_ZERO_OFFSET_RANGE = (-0.035, 0.035)
ACTION_DELAY_STEP_RANGE = (0, 1)
TORSO_COM_RANGE = (
    (-0.025, 0.025),
    (-0.05, 0.05),
    (-0.05, 0.05),
)

PUSH_INTERVAL_RANGE_S = (1.0, 3.0)
PUSH_VELOCITY_MIN = (-0.5, -0.5, -0.2, -0.52, -0.52, -0.78)
PUSH_VELOCITY_MAX = (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)


def _bucket_uniform(rng: jax.Array, shape: tuple[int, ...], value_range: tuple[float, float]) -> jax.Array:
    bucket = jax.random.randint(rng, shape, minval=0, maxval=NUM_MATERIAL_BUCKETS)
    bucket_center = (bucket.astype(jp.float32) + 0.5) / float(NUM_MATERIAL_BUCKETS)
    return value_range[0] + bucket_center * (value_range[1] - value_range[0])


def event_cfg_domain_randomize(model: mjx.Model, rng: jax.Array) -> tuple[mjx.Model, mjx.Model]:
    """Randomize MJX model fields using the IsaacLab tracking EventCfg ranges.

    MuJoCo does not expose separate PhysX-style static/dynamic friction or
    restitution materials. We map static friction to MuJoCo sliding friction and
    use the dynamic-friction sample for the second explicit pair-friction slot.
    Restitution is intentionally left unmapped because the closest MuJoCo knobs
    are solver parameters rather than a material restitution field.
    """

    @jax.vmap
    def rand_dynamics(one_rng: jax.Array):
        one_rng, key = jax.random.split(one_rng)
        static_friction = _bucket_uniform(key, (model.ngeom,), STATIC_FRICTION_RANGE)
        one_rng, key = jax.random.split(one_rng)
        dynamic_friction = _bucket_uniform(key, (model.ngeom,), DYNAMIC_FRICTION_RANGE)

        robot_geom_mask = model.geom_bodyid > 0
        geom_friction = model.geom_friction
        geom_friction = geom_friction.at[:, 0].set(jp.where(robot_geom_mask, static_friction, geom_friction[:, 0]))
        geom_friction = geom_friction.at[:, 1].set(jp.where(robot_geom_mask, dynamic_friction, geom_friction[:, 1]))

        one_rng, key = jax.random.split(one_rng)
        pair_static = _bucket_uniform(key, (model.npair,), STATIC_FRICTION_RANGE)
        one_rng, key = jax.random.split(one_rng)
        pair_dynamic = _bucket_uniform(key, (model.npair,), DYNAMIC_FRICTION_RANGE)
        pair_friction = model.pair_friction
        pair_friction = pair_friction.at[:, 0].set(pair_static)
        pair_friction = pair_friction.at[:, 1].set(pair_dynamic)

        one_rng, key = jax.random.split(one_rng)
        joint_offset = jax.random.uniform(
            key,
            shape=(29,),
            minval=JOINT_DEFAULT_POS_RANGE[0],
            maxval=JOINT_DEFAULT_POS_RANGE[1],
        )
        qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + joint_offset)

        one_rng, key = jax.random.split(one_rng)
        torso_com_offset = jax.random.uniform(
            key,
            shape=(3,),
            minval=jp.array([r[0] for r in TORSO_COM_RANGE], dtype=jp.float32),
            maxval=jp.array([r[1] for r in TORSO_COM_RANGE], dtype=jp.float32),
        )
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + torso_com_offset)

        one_rng, key = jax.random.split(one_rng)
        base_mass_delta = jax.random.uniform(
            key,
            (),
            minval=BASE_MASS_ADDED_RANGE[0],
            maxval=BASE_MASS_ADDED_RANGE[1],
        )
        body_mass = model.body_mass.at[BASE_BODY_ID].set(
            jp.maximum(1e-3, model.body_mass[BASE_BODY_ID] + base_mass_delta)
        )

        one_rng, key = jax.random.split(one_rng)
        link_mass_multiplier = jax.random.uniform(
            key,
            shape=(model.nbody,),
            minval=LINK_MASS_MULTIPLIER_RANGE[0],
            maxval=LINK_MASS_MULTIPLIER_RANGE[1],
        )
        link_mass_multiplier = jp.where(jp.arange(model.nbody) > BASE_BODY_ID, link_mass_multiplier, 1.0)
        body_mass = body_mass * link_mass_multiplier

        one_rng, key = jax.random.split(one_rng)
        kp_multiplier = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=PD_STIFFNESS_MULTIPLIER_RANGE[0],
            maxval=PD_STIFFNESS_MULTIPLIER_RANGE[1],
        )
        one_rng, key = jax.random.split(one_rng)
        kd_multiplier = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=PD_DAMPING_MULTIPLIER_RANGE[0],
            maxval=PD_DAMPING_MULTIPLIER_RANGE[1],
        )
        one_rng, key = jax.random.split(one_rng)
        motor_strength = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=MOTOR_STRENGTH_RANGE[0],
            maxval=MOTOR_STRENGTH_RANGE[1],
        )

        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
            model.actuator_gainprm[:, 0] * kp_multiplier * motor_strength
        )
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
            model.actuator_biasprm[:, 1] * kp_multiplier * motor_strength
        )
        dof_damping = model.dof_damping.at[6 : 6 + model.nu].set(
            model.dof_damping[6 : 6 + model.nu] * kd_multiplier * motor_strength
        )

        return (
            geom_friction,
            pair_friction,
            qpos0,
            body_ipos,
            body_mass,
            actuator_gainprm,
            actuator_biasprm,
            dof_damping,
        )

    (
        geom_friction,
        pair_friction,
        qpos0,
        body_ipos,
        body_mass,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "pair_friction": 0,
            "qpos0": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
            "dof_damping": 0,
        }
    )
    model = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "pair_friction": pair_friction,
            "qpos0": qpos0,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "dof_damping": dof_damping,
        }
    )
    return model, in_axes
