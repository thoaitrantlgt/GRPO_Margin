from pathlib import Path

from bm_grpo.experiments import build_runs

ROOT = Path(__file__).parents[1]


def test_main_matrix_expands_two_variants_and_three_seeds(tmp_path: Path) -> None:
    matrix = ROOT / "configs/experiments/main.yaml"
    runs = build_runs(matrix, materialize=False)
    assert len(runs) == 6
    assert {name for name, _, _ in runs} == {
        "paper_grpo_seed13",
        "paper_grpo_seed42",
        "paper_grpo_seed2026",
        "paper_boundary_seed13",
        "paper_boundary_seed42",
        "paper_boundary_seed2026",
    }


def test_qwen3_ablation_matrix_expands_expected_variants() -> None:
    matrix = ROOT / "configs/experiments/ablations_qwen3_4b.yaml"
    runs = build_runs(matrix, materialize=False)
    assert len(runs) == 7
    assert {name for name, _, _ in runs} == {
        "paper_qwen3_4b_standard_grpo_seed42",
        "paper_qwen3_4b_boundary_margin_seed42",
        "paper_qwen3_4b_gate_only_seed42",
        "paper_qwen3_4b_boundary_only_seed42",
        "paper_qwen3_4b_no_format_reward_seed42",
        "paper_qwen3_4b_top1_top2_seed42",
        "paper_qwen3_4b_no_advantage_clip_seed42",
    }
