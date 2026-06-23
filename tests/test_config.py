from pathlib import Path

import pytest

from bm_grpo.config import load_run_config

ROOT = Path(__file__).parents[1]


def test_smoke_config_uses_group_size_eight() -> None:
    config = load_run_config(ROOT / "configs/train/smoke_boundary.yaml")
    assert config.trainer.num_generations == 8
    assert config.trainer.gradient_accumulation_steps == 8


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("trainer:\n  mystery: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_run_config(config_path)
