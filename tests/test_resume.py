from pathlib import Path

from bm_grpo.config import load_run_config
from bm_grpo.train import find_latest_checkpoint, resolve_resume_checkpoint

ROOT = Path(__file__).parents[1]


def test_find_latest_checkpoint_picks_highest_step(tmp_path) -> None:
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-250").mkdir()
    assert find_latest_checkpoint(tmp_path).name == "checkpoint-500"


def test_resolve_resume_checkpoint_allows_config_mismatch(tmp_path) -> None:
    config = load_run_config(ROOT / "configs/train/smoke_grpo.yaml")
    config.experiment.output_dir = str(tmp_path)
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "resolved_config.yaml").write_text(
        (ROOT / "configs/train/smoke_boundary.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    resume_from = resolve_resume_checkpoint(config)
    assert resume_from is not None
    assert resume_from.endswith("checkpoint-100")


def test_resolve_resume_checkpoint_returns_latest_when_config_matches(tmp_path) -> None:
    config = load_run_config(ROOT / "configs/train/smoke_grpo.yaml")
    config.experiment.output_dir = str(tmp_path)
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "resolved_config.yaml").write_text(
        (ROOT / "configs/train/smoke_grpo.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    resume_from = resolve_resume_checkpoint(config)
    assert resume_from is not None
    assert resume_from.endswith("checkpoint-500")
