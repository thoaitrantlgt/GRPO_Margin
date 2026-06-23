from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from .config import MarginConfig
from .margin import apply_boundary_margin

SUPPORTED_TRL_VERSION = "1.6.0"


def require_supported_trl() -> None:
    try:
        installed = version("trl")
    except PackageNotFoundError as error:
        raise RuntimeError("TRL is not installed; install bm-grpo[train]") from error
    if installed != SUPPORTED_TRL_VERSION:
        raise RuntimeError(
            f"BoundaryMarginGRPOTrainer supports trl=={SUPPORTED_TRL_VERSION}; found trl=={installed}. "
            "The trainer relies on private reward/generation hooks."
        )


try:
    from trl import GRPOTrainer
except ImportError:  # Allows pure margin/data unit tests without training extras.
    GRPOTrainer = object  # type: ignore[assignment,misc]


class BoundaryMarginGRPOTrainer(GRPOTrainer):  # type: ignore[misc]
    """TRL GRPOTrainer that only reweights already-computed advantages."""

    def __init__(self, *args: Any, margin_config: MarginConfig | None = None, **kwargs: Any) -> None:
        require_supported_trl()
        self.margin_config = margin_config or MarginConfig(enabled=False)
        self._bm_rewards_per_func: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def _calculate_rewards(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        rewards = super()._calculate_rewards(*args, **kwargs)
        self._bm_rewards_per_func = rewards.detach().clone()
        return rewards

    def _generate_and_score_completions(self, inputs: Any) -> dict[str, Any]:
        output = super()._generate_and_score_completions(inputs)
        if not self.margin_config.enabled:
            self._bm_rewards_per_func = None
            return output
        if self._bm_rewards_per_func is None:
            raise RuntimeError("TRL did not call _calculate_rewards before producing advantages")
        if "advantages" not in output:
            raise RuntimeError("TRL generation output no longer contains 'advantages'")

        base_advantages = output["advantages"]
        gathered_rewards = self._bm_rewards_per_func
        reward_weights = self.reward_weights.detach().to(gathered_rewards.device)
        num_generations = self.num_generations if self.model.training else self.num_generations_eval

        if len(gathered_rewards) == len(base_advantages):
            result = apply_boundary_margin(
                base_advantages,
                gathered_rewards,
                reward_weights,
                num_generations,
                self.margin_config,
            )
            output["advantages"] = result.advantages
        else:
            # Rewards are gathered globally by TRL while advantages are local.
            weight_result = apply_boundary_margin(
                torch.ones(len(gathered_rewards), device=gathered_rewards.device, dtype=base_advantages.dtype),
                gathered_rewards,
                reward_weights,
                num_generations,
                self.margin_config,
            )
            local_count = len(base_advantages)
            start = self.accelerator.process_index * local_count
            stop = start + local_count
            local_weights = weight_result.weights[start:stop].to(base_advantages.device)
            if len(local_weights) != local_count:
                raise RuntimeError("Unable to align gathered margin weights with local advantages")
            weighted = base_advantages * local_weights
            if self.margin_config.advantage_clip is not None:
                weighted = torch.clamp(
                    weighted,
                    -self.margin_config.advantage_clip,
                    self.margin_config.advantage_clip,
                )
            output["advantages"] = weighted
            weight_result.metrics["margin/effective_advantage_norm"] = float(weighted.norm().item())
            result = weight_result

        mode = "train" if self.model.training else "eval"
        for name, value in result.metrics.items():
            self._metrics[mode][name].append(value)
        self._bm_rewards_per_func = None
        return output
