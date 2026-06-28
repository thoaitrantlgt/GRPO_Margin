from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "boundary_margin"
    seed: int = 42
    output_dir: str = "outputs/boundary_margin"


@dataclass(slots=True)
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = MODEL_REVISION
    dtype: str = "bfloat16"
    use_peft: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_bnb_nested_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(slots=True)
class DataConfig:
    train_path: str = "data/processed/smoke/train.parquet"
    validation_path: str | None = "data/processed/smoke/validation.parquet"


@dataclass(slots=True)
class RewardsConfig:
    weights: list[float] = field(default_factory=lambda: [1.0, 0.1])
    accuracy_reward_index: int = 0
    require_single_box: bool = True


@dataclass(slots=True)
class TrainerConfig:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 384
    max_steps: int = 20
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    optim: str = "paged_adamw_8bit"
    loss_type: str = "grpo"
    scale_rewards: str = "group"
    beta: float = 0.0
    epsilon: float = 0.2
    temperature: float = 0.9
    top_p: float = 0.95
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.5
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_model_impl: str = "vllm"
    vllm_gpu_memory_utilization: float = 0.45
    vllm_max_model_length: int | None = None
    vllm_tensor_parallel_size: int = 1
    vllm_enable_sleep_mode: bool = True
    vllm_server_base_url: str | None = None
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    vllm_group_port: int = 51216
    logging_steps: int = 1
    save_steps: int = 100
    eval_strategy: str = "no"
    report_to: str | list[str] = "tensorboard"
    log_completions: bool = True
    num_completions_to_print: int | None = 4
    log_unique_prompts: bool = False
    save_total_limit: int = 2


@dataclass(slots=True)
class MarginConfig:
    enabled: bool = True
    gate_type: str = "correct_rate"
    gate_floor: float = 0.25
    alpha: float = 0.5
    boundary_bandwidth: float = 0.25
    advantage_clip: float | None = 5.0
    min_valid_rewards: int = 2
    accuracy_reward_index: int = 0
    use_group_gate: bool = True
    use_boundary_proximity: bool = True
    top_gap_beta: float = 0.1
    top_gap_tau: float = 0.1


@dataclass(slots=True)
class TrackingConfig:
    save_environment: bool = True
    save_resolved_config: bool = True


@dataclass(slots=True)
class RunConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> None:
        effective_batch_size = (
            self.trainer.per_device_train_batch_size * self.trainer.gradient_accumulation_steps
        )
        if effective_batch_size % self.trainer.num_generations != 0:
            raise ValueError(
                "per_device_train_batch_size * gradient_accumulation_steps must be divisible by num_generations"
            )
        if self.trainer.num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if self.margin.boundary_bandwidth <= 0:
            raise ValueError("margin.boundary_bandwidth must be > 0")
        if not 0 <= self.margin.gate_floor <= 1:
            raise ValueError("margin.gate_floor must be in [0, 1]")
        if self.margin.alpha < 0:
            raise ValueError("margin.alpha must be >= 0")
        if self.margin.min_valid_rewards < 2:
            raise ValueError("margin.min_valid_rewards must be >= 2")
        if self.margin.top_gap_tau <= 0:
            raise ValueError("margin.top_gap_tau must be > 0")
        if self.margin.gate_type not in {"correct_rate", "top1_top2"}:
            raise ValueError("margin.gate_type must be 'correct_rate' or 'top1_top2'")
        if len(self.rewards.weights) < 1:
            raise ValueError("rewards.weights cannot be empty")
        if self.margin.accuracy_reward_index != self.rewards.accuracy_reward_index:
            raise ValueError("margin and rewards accuracy_reward_index must match")
        if not 0 <= self.margin.accuracy_reward_index < len(self.rewards.weights):
            raise ValueError("accuracy_reward_index is outside rewards.weights")
        if self.trainer.loss_type != "grpo":
            raise ValueError("v1 requires trainer.loss_type='grpo' for a controlled baseline")
        if self.trainer.scale_rewards != "group":
            raise ValueError("v1 requires trainer.scale_rewards='group'")
        if self.trainer.vllm_mode not in {"colocate", "server"}:
            raise ValueError("trainer.vllm_mode must be 'colocate' or 'server'")
        if self.trainer.vllm_model_impl not in {"vllm", "transformers"}:
            raise ValueError("trainer.vllm_model_impl must be 'vllm' or 'transformers'")
        if not 0 < self.trainer.vllm_gpu_memory_utilization <= 1:
            raise ValueError("trainer.vllm_gpu_memory_utilization must be in (0, 1]")
        if self.trainer.vllm_tensor_parallel_size < 1:
            raise ValueError("trainer.vllm_tensor_parallel_size must be >= 1")
        if self.trainer.vllm_server_port <= 0:
            raise ValueError("trainer.vllm_server_port must be > 0")
        if self.trainer.vllm_group_port <= 0:
            raise ValueError("trainer.vllm_group_port must be > 0")
        if self.trainer.vllm_server_timeout <= 0:
            raise ValueError("trainer.vllm_server_timeout must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: dict[str, Any] | None, section: str) -> T:
    values = values or {}
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {', '.join(unknown)}")
    return cls(**values)


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Run config must be a YAML mapping")
    allowed = {item.name for item in dataclasses.fields(RunConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level config sections: {', '.join(unknown)}")
    config = RunConfig(
        experiment=_strict_dataclass(ExperimentConfig, raw.get("experiment"), "experiment"),
        model=_strict_dataclass(ModelConfig, raw.get("model"), "model"),
        data=_strict_dataclass(DataConfig, raw.get("data"), "data"),
        rewards=_strict_dataclass(RewardsConfig, raw.get("rewards"), "rewards"),
        trainer=_strict_dataclass(TrainerConfig, raw.get("trainer"), "trainer"),
        margin=_strict_dataclass(MarginConfig, raw.get("margin"), "margin"),
        tracking=_strict_dataclass(TrackingConfig, raw.get("tracking"), "tracking"),
    )
    config.validate()
    return config


def dump_run_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
