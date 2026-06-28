from pathlib import Path

import pytest

from bm_grpo.compare import (
    build_comparison,
    comparison_markdown,
    load_pair_config,
    _eval_checkpoint,
    _needs_more_training,
    validate_controlled_pair,
)
from bm_grpo.config import load_run_config

ROOT = Path(__file__).parents[1]


def test_smoke_pair_is_controlled_and_uses_group_eight() -> None:
    pair = load_pair_config(ROOT / "configs/compare/smoke.yaml")
    baseline = load_run_config(pair.baseline_config)
    method = load_run_config(pair.method_config)
    validate_controlled_pair(baseline, method)
    assert baseline.trainer.num_generations == method.trainer.num_generations == 8
    assert baseline.margin.enabled is False
    assert method.margin.enabled is True


def test_all_paired_profiles_are_controlled() -> None:
    for name in ("smoke", "gsm8k", "paper_seed42", "paper_qwen3_4b_seed42"):
        pair = load_pair_config(ROOT / f"configs/compare/{name}.yaml")
        validate_controlled_pair(load_run_config(pair.baseline_config), load_run_config(pair.method_config))


def test_controlled_pair_rejects_different_rollout_budget() -> None:
    pair = load_pair_config(ROOT / "configs/compare/smoke.yaml")
    baseline = load_run_config(pair.baseline_config)
    method = load_run_config(pair.method_config)
    method.trainer.max_steps += 1
    with pytest.raises(ValueError, match="trainer"):
        validate_controlled_pair(baseline, method)


def test_compare_uses_latest_checkpoint_for_resume_and_eval(tmp_path) -> None:
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-300").mkdir()
    assert _needs_more_training(tmp_path, 400) is True
    assert _needs_more_training(tmp_path, 300) is False
    assert _eval_checkpoint(tmp_path, "final_adapter").name == "checkpoint-300"


def test_comparison_builds_metric_deltas_and_markdown() -> None:
    comparison = build_comparison(
        {"train_loss": 1.0, "train_runtime": 100.0},
        {"train_loss": 0.8, "train_runtime": 110.0},
        {"gsm8k/pass1": {"pass_at_k": 0.5, "format_rate": 0.9}},
        {"gsm8k/pass1": {"pass_at_k": 0.6, "format_rate": 0.95}},
    )
    assert comparison["training"]["train_loss"]["delta"] == pytest.approx(-0.2)
    assert comparison["evaluation"]["gsm8k/pass1"]["pass_at_k"]["delta"] == pytest.approx(0.1)
    report = comparison_markdown("test", comparison)
    assert "GRPO vs Boundary-Margin" in report
    assert "gsm8k/pass1" in report
