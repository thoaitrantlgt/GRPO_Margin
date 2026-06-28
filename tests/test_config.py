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


def test_qwen3_profile_uses_instruct_model_without_vllm() -> None:
    config = load_run_config(ROOT / "configs/train/paper_qwen3_4b_boundary_seed42.yaml")
    assert config.model.name_or_path == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert config.trainer.use_vllm is False
    assert config.trainer.vllm_mode == "colocate"
    assert config.trainer.vllm_gpu_memory_utilization == pytest.approx(0.55)
    assert config.trainer.vllm_max_model_length == 3072
    assert config.trainer.chat_template_kwargs == {"enable_thinking": False}
    assert config.trainer.temperature == pytest.approx(0.7)
    assert config.trainer.top_k == 50
    assert config.trainer.min_p == pytest.approx(0.02)
    assert config.trainer.repetition_penalty == pytest.approx(1.08)
