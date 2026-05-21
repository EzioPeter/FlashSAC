"""EventCfg-style domain randomization for the HRLG G1 MuJoCo Playground task."""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

NUM_MATERIAL_BUCKETS = 64
TORSO_BODY_ID = 16

STATIC_FRICTION_RANGE = (0.3, 1.6)
DYNAMIC_FRICTION_RANGE = (0.3, 1.2)
RESTITUTION_RANGE = (0.0, 0.5)
JOINT_DEFAULT_POS_RANGE = (-0.01, 0.01)
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
        geom_friction = geom_friction.at[:, 0].set(
            jp.where(robot_geom_mask, static_friction, geom_friction[:, 0])
        )
        geom_friction = geom_friction.at[:, 1].set(
            jp.where(robot_geom_mask, dynamic_friction, geom_friction[:, 1])
        )

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

        return geom_friction, pair_friction, qpos0, body_ipos

    geom_friction, pair_friction, qpos0, body_ipos = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "pair_friction": 0,
        "qpos0": 0,
        "body_ipos": 0,
    })
    model = model.tree_replace({
        "geom_friction": geom_friction,
        "pair_friction": pair_friction,
        "qpos0": qpos0,
        "body_ipos": body_ipos,
    })
    return model, in_axes
