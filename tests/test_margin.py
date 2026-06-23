import pytest
import torch

from bm_grpo.config import MarginConfig
from bm_grpo.margin import apply_boundary_margin


@pytest.mark.parametrize(
    ("correct", "expected"),
    [(0, 0.0), (2, 0.75), (4, 1.0), (6, 0.75), (8, 0.0)],
)
def test_correct_rate_gate_for_group_eight(correct: int, expected: float) -> None:
    accuracy = torch.tensor([1.0] * correct + [0.0] * (8 - correct))
    rewards = torch.stack([accuracy, torch.zeros(8)], dim=1)
    result = apply_boundary_margin(
        torch.ones(8),
        rewards,
        [1.0, 0.1],
        group_size=8,
        config=MarginConfig(use_boundary_proximity=False),
    )
    if correct in {0, 8}:
        # Equal total rewards use the explicit plain-GRPO zero-variance fallback.
        assert result.metrics["margin/zero_variance_rate"] == 1.0
        assert torch.equal(result.weights, torch.ones(8))
    else:
        assert result.metrics["margin/group_gate"] == pytest.approx(expected)


def test_disabled_margin_returns_same_tensor() -> None:
    advantages = torch.randn(8)
    rewards = torch.zeros((8, 2))
    result = apply_boundary_margin(
        advantages,
        rewards,
        [1.0, 0.1],
        8,
        MarginConfig(enabled=False),
    )
    assert result.advantages is advantages
    assert torch.equal(result.weights, torch.ones_like(advantages))


def test_invalid_rewards_fall_back_and_detach_weights() -> None:
    advantages = torch.ones(8, requires_grad=True)
    rewards = torch.full((8, 2), torch.nan)
    rewards[0, 0] = 1.0
    result = apply_boundary_margin(advantages, rewards, [1.0, 0.1], 8, MarginConfig())
    assert result.metrics["margin/fallback_rate"] == 1.0
    assert not result.weights.requires_grad
    assert torch.equal(result.weights, torch.ones(8))


def test_advantage_clipping() -> None:
    advantages = torch.tensor([100.0, -100.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    accuracy = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    rewards = torch.stack([accuracy, torch.arange(8) % 2], dim=1)
    result = apply_boundary_margin(
        advantages,
        rewards,
        [1.0, 0.1],
        8,
        MarginConfig(advantage_clip=5.0),
    )
    assert result.advantages.max() <= 5
    assert result.advantages.min() >= -5

