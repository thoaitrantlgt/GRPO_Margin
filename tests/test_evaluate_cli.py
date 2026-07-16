from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_evaluate_cli_has_base_only_flag() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bm_grpo.evaluate", "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "--base-only" in completed.stdout


def test_evaluate_cli_requires_checkpoint_without_base_only() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bm_grpo.evaluate", "--config", "configs/eval/paper_qwen3_4b.yaml"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode != 0
    assert "--checkpoint is required unless --base-only is set" in completed.stderr
