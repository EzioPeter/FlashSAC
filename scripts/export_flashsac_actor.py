"""Export a FlashSAC actor checkpoint to a deterministic TorchScript policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402


class DeterministicFlashSACActor(nn.Module):
    def __init__(self, actor: FlashSACActor):
        super().__init__()
        self.actor = actor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor.get_mean_and_std(observations.float(), training=False)
        return torch.tanh(mean)


def _resolve_actor_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_dir():
        return checkpoint_path / "actor.pt"
    return checkpoint_path


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in state_dict):
        return {key.removeprefix(prefix): value for key, value in state_dict.items()}
    return state_dict


def _infer_actor_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    input_dim = int(state_dict["embedder.norm.weight"].shape[0])
    hidden_dim = int(state_dict["embedder.w.w.weight"].shape[0])
    action_dim = int(state_dict["predictor.mean_bias"].shape[0])
    block_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("encoder.") and key.split(".")[1].isdigit()
    }
    return len(block_ids), input_dim, hidden_dim, action_dim


def export_actor(checkpoint_path: Path, output_path: Path, device: str) -> None:
    actor_path = _resolve_actor_path(checkpoint_path)
    ckpt = torch.load(actor_path, map_location=device)
    state_dict = _strip_compile_prefix(ckpt["network_state_dict"])
    num_blocks, input_dim, hidden_dim, action_dim = _infer_actor_dims(state_dict)

    actor = FlashSACActor(
        num_blocks=num_blocks,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    ).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    actor.requires_grad_(False)

    wrapper = DeterministicFlashSACActor(actor).eval()
    example_obs = torch.zeros(1, input_dim, device=device)
    traced = torch.jit.trace(wrapper, example_obs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))

    loaded = torch.jit.load(str(output_path), map_location=device)
    output = loaded(example_obs)
    if tuple(output.shape) != (1, action_dim):
        raise RuntimeError(f"Unexpected exported output shape: {tuple(output.shape)} != {(1, action_dim)}")

    print(
        f"Exported {actor_path} -> {output_path} "
        f"(obs_dim={input_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to actor.pt or a checkpoint dir.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path for the TorchScript policy.pt.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device used during export.")
    args = parser.parse_args()

    export_actor(args.checkpoint_path, args.output_path, args.device)


if __name__ == "__main__":
    main()
