from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import MarginConfig


@dataclass(slots=True)
class MarginResult:
    advantages: torch.Tensor
    weights: torch.Tensor
    metrics: dict[str, float]


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@torch.no_grad()
def apply_boundary_margin(
    base_advantages: torch.Tensor,
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor | list[float],
    group_size: int,
    config: MarginConfig,
) -> MarginResult:
    """Reweight GRPO advantages without changing reward or loss computation.

    Rows must be ordered as contiguous completion groups, which is the ordering
    produced by TRL's GRPO trainer after reward gathering.
    """

    if base_advantages.ndim != 1:
        raise ValueError("base_advantages must be a 1-D tensor")
    if rewards_per_func.ndim != 2:
        raise ValueError("rewards_per_func must have shape (num_completions, num_reward_funcs)")
    if len(base_advantages) != len(rewards_per_func):
        raise ValueError("advantages and rewards must have the same number of rows")
    if group_size < 2 or len(base_advantages) % group_size != 0:
        raise ValueError("number of rows must be divisible by group_size >= 2")
    if not 0 <= config.accuracy_reward_index < rewards_per_func.shape[1]:
        raise ValueError("accuracy_reward_index is outside rewards_per_func")

    weights_tensor = torch.as_tensor(
        reward_weights,
        dtype=rewards_per_func.dtype,
        device=rewards_per_func.device,
    )
    if weights_tensor.numel() != rewards_per_func.shape[1]:
        raise ValueError("reward_weights length must match rewards_per_func columns")

    if not config.enabled:
        return MarginResult(
            advantages=base_advantages,
            weights=torch.ones_like(base_advantages),
            metrics={},
        )

    scorable = ~torch.isnan(rewards_per_func).all(dim=1)
    total_rewards = torch.nansum(rewards_per_func * weights_tensor.unsqueeze(0), dim=1)
    accuracy = rewards_per_func[:, config.accuracy_reward_index]
    output_weights = torch.ones_like(base_advantages)

    pass_rates: list[float] = []
    gates: list[float] = []
    top1_top2_values: list[float] = []
    top1_mean_values: list[float] = []
    top1_median_values: list[float] = []
    boundary_distances: list[float] = []
    fallback_count = 0
    zero_variance_count = 0
    hopeless_count = 0
    mid_count = 0
    easy_count = 0

    for start in range(0, len(base_advantages), group_size):
        stop = start + group_size
        group_accuracy = accuracy[start:stop]
        group_rewards = total_rewards[start:stop]
        valid_accuracy = ~torch.isnan(group_accuracy)
        valid_rewards = scorable[start:stop] & valid_accuracy

        if int(valid_accuracy.sum().item()) < config.min_valid_rewards:
            fallback_count += 1
            continue

        valid_accuracy_values = group_accuracy[valid_accuracy]
        pass_rate = float(valid_accuracy_values.mean().item())
        pass_rates.append(pass_rate)
        if pass_rate < 0.25:
            hopeless_count += 1
        elif pass_rate > 0.75:
            easy_count += 1
        else:
            mid_count += 1

        valid_group_rewards = group_rewards[valid_rewards]
        if valid_group_rewards.numel() < 2:
            fallback_count += 1
            continue

        sorted_rewards = torch.sort(valid_group_rewards, descending=True).values
        top1 = sorted_rewards[0]
        top1_top2 = float((top1 - sorted_rewards[1]).item())
        top1_mean = float((top1 - valid_group_rewards.mean()).item())
        top1_median = float((top1 - valid_group_rewards.median()).item())
        top1_top2_values.append(top1_top2)
        top1_mean_values.append(top1_mean)
        top1_median_values.append(top1_median)

        dispersion = valid_group_rewards.max() - valid_group_rewards.min()
        if torch.isclose(dispersion, torch.zeros_like(dispersion)):
            zero_variance_count += 1
            continue

        if config.gate_type == "correct_rate":
            gate = 4.0 * pass_rate * (1.0 - pass_rate)
        elif config.gate_type == "top1_top2":
            gap = torch.tensor(top1_top2, device=group_rewards.device, dtype=group_rewards.dtype)
            gate = float(torch.sigmoid((gap - config.top_gap_beta) / config.top_gap_tau).item())
        else:  # defensive guard for direct construction without RunConfig.validate()
            raise ValueError(f"Unsupported gate_type: {config.gate_type}")
        if not config.use_group_gate:
            gate = 1.0
        gates.append(gate)

        boundary = 0.5 * (valid_group_rewards.max() + valid_group_rewards.min())
        distances = torch.abs(group_rewards - boundary)
        proximity = torch.exp(-distances / config.boundary_bandwidth)
        proximity = torch.where(valid_rewards, proximity, torch.zeros_like(proximity))
        boundary_distances.extend(distances[valid_rewards].tolist())
        if not config.use_boundary_proximity:
            proximity = torch.zeros_like(proximity)

        group_weights = (
            config.gate_floor
            + (1.0 - config.gate_floor) * gate
            + config.alpha * gate * proximity
        )
        # Unscorable rows remain neutral; TRL has already set their advantage to zero.
        group_weights = torch.where(valid_accuracy, group_weights, torch.ones_like(group_weights))
        output_weights[start:stop] = group_weights

    weighted_advantages = base_advantages * output_weights.detach()
    if config.advantage_clip is not None:
        weighted_advantages = torch.clamp(
            weighted_advantages,
            min=-config.advantage_clip,
            max=config.advantage_clip,
        )

    num_groups = len(base_advantages) // group_size
    metrics = {
        "margin/pass_rate": _safe_mean(pass_rates),
        "margin/group_gate": _safe_mean(gates),
        "margin/top1_top2": _safe_mean(top1_top2_values),
        "margin/top1_mean": _safe_mean(top1_mean_values),
        "margin/top1_median": _safe_mean(top1_median_values),
        "margin/boundary_distance_mean": _safe_mean(boundary_distances),
        "margin/weight_mean": float(output_weights.mean().item()),
        "margin/weight_min": float(output_weights.min().item()),
        "margin/weight_max": float(output_weights.max().item()),
        "margin/effective_advantage_norm": float(weighted_advantages.norm().item()),
        "margin/group_bucket/hopeless": hopeless_count / num_groups,
        "margin/group_bucket/mid": mid_count / num_groups,
        "margin/group_bucket/easy": easy_count / num_groups,
        "margin/fallback_rate": fallback_count / num_groups,
        "margin/zero_variance_rate": zero_variance_count / num_groups,
    }
    return MarginResult(
        advantages=weighted_advantages,
        weights=output_weights.detach(),
        metrics=metrics,
    )
