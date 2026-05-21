"""Sanity checks for the HRLG-compatible MuJoCo Playground G1 task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HRLG_G1_YAML = Path("/root/AGES/code/humanoid_rl_gym/deploy/deploy_mujoco/configs/g1.yaml")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.envs.mujoco_playground import make_mujoco_playground_env  # noqa: E402
from flash_rl.envs.mujoco_playground_tasks.g1_hrlg import (  # noqa: E402
    G1JoystickFlatTerrainHRLG,
    constants as hrlg,
    default_config,
)


def _load_hrlg_config() -> dict:
    with HRLG_G1_YAML.open("r") as f:
        return yaml.safe_load(f)


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
        raise RuntimeError(f"Could not resolve joint id {joint_id}.")
    return name


def _actuator_joint_order(model: mujoco.MjModel) -> list[str]:
    return [
        _joint_name(model, int(model.actuator_trnid[actuator_id, 0]))
        for actuator_id in range(model.nu)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slow-step",
        action="store_true",
        help="Also run one full MJX step. This can take minutes the first time because XLA compiles the model.",
    )
    args = parser.parse_args()

    deploy_cfg = _load_hrlg_config()
    env = G1JoystickFlatTerrainHRLG(config=default_config())

    obs_size = env.observation_size
    assert isinstance(obs_size, dict), f"Expected asymmetric observation dict, got {obs_size!r}"
    actor_obs_dim = obs_size["state"][0]
    critic_obs_dim = obs_size["privileged_state"][0]
    action_dim = env.action_size
    actuator_joint_order = _actuator_joint_order(env.mj_model)

    assert hrlg.ENV_NAME == "G1JoystickFlatTerrainHRLG"
    assert actor_obs_dim == deploy_cfg["num_obs"] == hrlg.POLICY_OBS_SIZE
    assert critic_obs_dim > actor_obs_dim
    assert action_dim == deploy_cfg["num_actions"] == hrlg.ACTION_SIZE
    assert actuator_joint_order == deploy_cfg["model_joint_names"] == list(hrlg.JOINT_NAMES)
    assert np.isclose(env._config.action_scale, deploy_cfg["action_scale"])
    assert np.allclose(np.array(hrlg.DEFAULT_ANGLES), np.array(deploy_cfg["default_angles"]))
    assert np.allclose(np.array(hrlg.CMD_SCALE), np.array(deploy_cfg["cmd_scale"]))
    assert np.isclose(hrlg.ANG_VEL_SCALE, deploy_cfg["ang_vel_scale"])
    assert np.isclose(hrlg.DOF_VEL_SCALE, deploy_cfg["dof_vel_scale"])
    assert np.isclose(hrlg.GAIT_PHASE_CYCLE, deploy_cfg["gait_phase"])

    rng = jax.random.PRNGKey(0)
    raw_state = env.reset(rng)
    raw_policy_obs = np.array(raw_state.obs["state"])
    raw_priv_obs = np.array(raw_state.obs["privileged_state"])
    assert raw_policy_obs.shape == (hrlg.POLICY_OBS_SIZE,)
    assert raw_priv_obs.shape == (critic_obs_dim,)
    assert np.all(np.isfinite(raw_policy_obs))
    assert np.all(np.isfinite(raw_priv_obs))

    action = jp.zeros((hrlg.ACTION_SIZE,), dtype=jp.float32)
    step_shape = jax.eval_shape(env.step, raw_state, action)
    assert step_shape.obs["state"].shape == (hrlg.POLICY_OBS_SIZE,)
    assert step_shape.obs["privileged_state"].shape == (critic_obs_dim,)
    assert step_shape.reward.shape == ()
    assert step_shape.done.shape == ()

    slow_step_status = "skipped"
    if args.slow_step:
        raw_next_state = env.step(raw_state, action)
        raw_next_policy_obs = np.array(raw_next_state.obs["state"])
        raw_next_priv_obs = np.array(raw_next_state.obs["privileged_state"])
        raw_reward = np.array(raw_next_state.reward)
        assert raw_next_policy_obs.shape == (hrlg.POLICY_OBS_SIZE,)
        assert raw_next_priv_obs.shape == (critic_obs_dim,)
        assert np.all(np.isfinite(raw_next_policy_obs))
        assert np.all(np.isfinite(raw_next_priv_obs))
        assert np.all(np.isfinite(raw_reward))
        slow_step_status = "ok"

    vec_env = make_mujoco_playground_env(
        env_name=hrlg.ENV_NAME,
        seed=0,
        num_envs=1,
        max_episode_steps=1000,
        use_domain_randomization=False,
        use_push_randomization=False,
    )
    assert vec_env.single_observation_space.shape == (critic_obs_dim,)
    assert vec_env.single_action_space.shape == (hrlg.ACTION_SIZE,)
    assert vec_env.obs_size[-1] == hrlg.POLICY_OBS_SIZE

    print(f"env_name={hrlg.ENV_NAME}")
    print(f"actor_obs_dim={actor_obs_dim}")
    print(f"critic_obs_dim={critic_obs_dim}")
    print(f"action_dim={action_dim}")
    print(f"joint_order_ok={actuator_joint_order == deploy_cfg['model_joint_names']}")
    print(f"action_scale={env._config.action_scale}")
    print("reset_nan_check=ok")
    print("step_shape_check=ok")
    print(f"slow_step_nan_check={slow_step_status}")


if __name__ == "__main__":
    main()
