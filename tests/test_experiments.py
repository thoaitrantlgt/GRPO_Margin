from pathlib import Path
import subprocess
import sys

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


def test_qwen3_ablation_eval_only_dry_run_skips_training() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bm_grpo.experiments",
            "--matrix",
            str(ROOT / "configs/experiments/ablations_qwen3_4b.yaml"),
            "--eval-only",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "EVAL-ONLY: skipping all training commands." in completed.stdout
    assert "bm_grpo.train" not in completed.stdout
    assert "bm_grpo.evaluate" in completed.stdout


def test_eval_only_skips_missing_checkpoints(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    base = tmp_path / "base.yaml"
    eval_config = tmp_path / "eval.yaml"
    base.write_text(
        """
experiment:
  name: base
  output_dir: outputs/base
trainer:
  max_steps: 1
margin:
  enabled: true
""",
        encoding="utf-8",
    )
    eval_config.write_text("seed: 42\nmodel: {}\ndatasets: {}\ngeneration: {}\n", encoding="utf-8")
    matrix.write_text(
        """
base_config: base.yaml
generated_dir: generated
accelerate_config: configs/accelerate/single_gpu.yaml
evaluation_config: eval.yaml
seeds: [42]
variants:
  - name: missing
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bm_grpo.experiments",
            "--matrix",
            str(matrix),
            "--eval-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "SKIP EVAL missing_seed42: No checkpoint found" in completed.stdout
