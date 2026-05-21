#!/usr/bin/env python3
"""Evaluate the trained MuJoCo Playground G1 HRLG policy and export an mp4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.mujoco_playground import make_mujoco_playground_env
from flash_rl.evaluation import evaluate, record_video


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "models/mjp_g1_hrlg/flat_50m_gpu2/G1JoystickFlatTerrainHRLG/seed0-0521-115545/step48829"
)


def _write_mp4(video: np.ndarray, path: Path, fps: int) -> None:
    """Write a (B, T, C, H, W) uint8 video tensor to mp4 using the first batch item."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video[0].transpose(0, 2, 3, 1)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _load_actor_for_eval(agent: object, checkpoint: Path) -> None:
    actor = getattr(agent, "_actor")
    ckpt = torch.load(checkpoint / "actor.pt", map_location=next(actor.network.parameters()).device)
    state_dict = ckpt["network_state_dict"]
    current_state = actor.network.state_dict()
    if state_dict and next(iter(state_dict)).startswith("_orig_mod.") and not next(iter(current_state)).startswith(
        "_orig_mod."
    ):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    actor.network.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--video-length", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "videos")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    hydra.initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs"))
    cfg = hydra.compose(
        config_name="flashSAC_base",
        overrides=[
            "env=mujoco_playground_g1_hrlg",
            "agent=flashSAC",
            "agent.asymmetric_observation=true",
            "agent.buffer_max_length=1",
            "agent.buffer_min_length=1",
            "agent.buffer_device_type=cpu",
            "agent.use_compile=false",
            "agent.load_optimizer=false",
            "agent.load_reward_normalizer=false",
            "num_eval_envs=50",
            "num_record_envs=1",
            f"num_eval_episodes={args.num_eval_episodes}",
            "num_record_episodes=1",
            "gamma=0.97",
            "n_step=1",
        ],
    )
    OmegaConf.resolve(cfg)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    eval_env = make_mujoco_playground_env(
        env_name=cfg.env.env_name,
        seed=cfg.seed,
        num_envs=cfg.num_eval_envs,
        max_episode_steps=cfg.env.max_episode_steps,
        use_domain_randomization=False,
        use_push_randomization=False,
        height=args.height,
        width=args.width,
    )
    record_env = make_mujoco_playground_env(
        env_name=cfg.env.env_name,
        seed=cfg.seed + 10_000,
        num_envs=1,
        max_episode_steps=cfg.env.max_episode_steps,
        use_domain_randomization=False,
        use_push_randomization=False,
        height=args.height,
        width=args.width,
    )

    _, env_info = eval_env.reset()
    agent = create_agent(
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    _load_actor_for_eval(agent, checkpoint)

    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(
        agent,
        record_env,
        num_episodes=1,
        env_type=cfg.env.env_type,
        video_length=args.video_length,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"mjp_g1_hrlg_{checkpoint.name}_{stamp}"
    video_path = args.output_dir / f"{stem}.mp4"
    metrics_path = args.output_dir / f"{stem}_metrics.json"

    _write_mp4(video_info["video"], video_path, fps=args.fps)
    metrics_path.write_text(json.dumps(eval_info, indent=2, sort_keys=True) + "\n")

    print("eval_info=" + json.dumps(eval_info, sort_keys=True))
    print(f"video_path={video_path}")
    print(f"metrics_path={metrics_path}")

    eval_env.close()
    record_env.close()


if __name__ == "__main__":
    main()
