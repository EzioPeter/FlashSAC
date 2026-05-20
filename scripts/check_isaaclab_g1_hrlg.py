"""Smoke-check the FlashSAC Isaac Lab G1 humanoid_rl_gym interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.envs.isaaclab import make_isaaclab_env  # noqa: E402
from flash_rl.envs.isaaclab_tasks.g1_hrlg.interface import (  # noqa: E402
    G1_HRLG_DEFAULT_JOINT_POS,
    G1_HRLG_JOINT_NAMES,
    G1_HRLG_NUM_ACTIONS,
    G1_HRLG_NUM_OBS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_isaaclab_env(
        env_name="FlashSAC-Isaac-G1-HRLG-Flat-v0",
        num_envs=args.num_envs,
        seed=args.seed,
    )
    obs, infos = env.reset(random_start_init=False)

    raw_env = env.envs.unwrapped
    action_term = raw_env.action_manager.get_term("joint_pos")
    resolved_joint_names = list(action_term._joint_names)
    robot = raw_env.scene["robot"]
    joint_ids = action_term._joint_ids
    resolved_default = robot.data.default_joint_pos[0, joint_ids].detach().cpu().numpy()

    print(f"obs.shape={obs.shape}")
    print(f"action_space.shape={env.single_action_space.shape}")
    print(f"actor_observation_size={infos['actor_observation_size']}")
    print(f"resolved_joint_names={resolved_joint_names}")
    print(f"resolved_default={np.round(resolved_default, 5).tolist()}")

    assert tuple(obs.shape) == (args.num_envs, G1_HRLG_NUM_OBS), obs.shape
    assert tuple(env.single_action_space.shape) == (G1_HRLG_NUM_ACTIONS,), env.single_action_space.shape
    assert tuple(infos["actor_observation_size"]) == (G1_HRLG_NUM_OBS,), infos["actor_observation_size"]
    assert resolved_joint_names == G1_HRLG_JOINT_NAMES, resolved_joint_names
    np.testing.assert_allclose(
        resolved_default,
        np.array(G1_HRLG_DEFAULT_JOINT_POS, dtype=np.float32),
        rtol=0.0,
        atol=1.0e-5,
    )

    actions = env.action_space.sample()
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)
    print(f"next_obs.shape={next_obs.shape}")
    print(f"reward.shape={np.asarray(rewards).shape}")
    print(f"terminations.shape={np.asarray(terminations).shape}")
    print(f"truncations.shape={np.asarray(truncations).shape}")
    print(f"final_obs.shape={np.asarray(step_infos['final_obs']).shape}")
    assert tuple(next_obs.shape) == (args.num_envs, G1_HRLG_NUM_OBS), next_obs.shape

    # Isaac Sim permits only one SimulationApp in a process; close is intentionally no-op in the wrapper.
    _ = raw_env  # keep a local reference alive until all checks complete.


if __name__ == "__main__":
    main()
