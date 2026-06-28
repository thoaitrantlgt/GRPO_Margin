from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig, dump_run_config, load_run_config
from .rewards import accuracy_reward, boxed_format_reward
from .runtime import write_environment
from .trainer import BoundaryMarginGRPOTrainer


@dataclass(frozen=True, slots=True)
class RuntimePrecision:
    model_dtype: torch.dtype
    bf16: bool
    fp16: bool
    tf32: bool


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    base = Path(output_dir)
    if not base.exists():
        return None
    checkpoints = [path for path in base.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))
    return checkpoints[-1]


def _load_saved_config(path: Path) -> RunConfig | None:
    if not path.exists():
        return None
    return load_run_config(path)


def _require_existing_path(path_value: str | None, label: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing {label}: {path}. Run `python -m bm_grpo.data.prepare --config configs/data/gsm8k.yaml` "
        "or point the train config at an existing processed dataset."
    )


def resolve_resume_checkpoint(config: RunConfig, explicit_resume: str | None = None) -> str | None:
    if explicit_resume:
        print(f"Resuming from explicit checkpoint: {explicit_resume}")
        return explicit_resume

    output_dir = Path(config.experiment.output_dir)
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        print(f"No checkpoint found in {output_dir}; starting from scratch.")
        return None

    saved_config = _load_saved_config(output_dir / "resolved_config.yaml")
    if saved_config is not None and saved_config.to_dict() != config.to_dict():
        warnings.warn(
            "Found an existing checkpoint and the saved resolved config differs from the current config. "
            "Resuming anyway from the latest checkpoint as requested.",
            stacklevel=2,
        )
    print(f"Auto-resuming from latest checkpoint: {latest}")
    return str(latest)


def resolve_runtime_precision(config: RunConfig) -> RuntimePrecision:
    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen 1.5B QLoRA profile requires a CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    ampere_or_newer = capability[0] >= 8
    bf16_supported = bool(torch.cuda.is_bf16_supported())

    use_bf16 = config.trainer.bf16 and bf16_supported
    use_fp16 = config.trainer.fp16 or (config.trainer.bf16 and not bf16_supported)
    use_tf32 = config.trainer.tf32 and ampere_or_newer

    if config.trainer.bf16 and not bf16_supported:
        warnings.warn(
            f"GPU capability {capability} does not support BF16; falling back to FP16.",
            stacklevel=2,
        )
    if config.trainer.tf32 and not ampere_or_newer:
        warnings.warn(
            f"GPU capability {capability} does not support TF32; disabling TF32.",
            stacklevel=2,
        )
    if use_bf16 and use_fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return RuntimePrecision(model_dtype=model_dtype, bf16=use_bf16, fp16=use_fp16, tf32=use_tf32)


def _filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _require_vllm_if_needed(config: RunConfig) -> None:
    if not config.trainer.use_vllm or config.trainer.vllm_mode != "colocate":
        return
    if importlib.util.find_spec("vllm") is not None:
        return
    raise RuntimeError(
        "trainer.use_vllm=true with vllm_mode='colocate' requires vLLM. "
        "Install it on the Linux training server with `pip install -e '.[train,test,vllm]'`."
    )


def _build_training_components(config: RunConfig):
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig
    except ImportError as error:
        raise RuntimeError("Training requires bm-grpo[train]") from error

    _require_vllm_if_needed(config)
    precision = resolve_runtime_precision(config)
    dtype = precision.model_dtype
    quantization = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=config.model.use_bnb_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _require_existing_path(config.data.train_path, "training dataset")
    _require_existing_path(config.data.validation_path, "validation dataset")

    train_dataset = load_dataset("parquet", data_files=config.data.train_path, split="train")
    peft_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    grpo_kwargs = {
        "output_dir": config.experiment.output_dir,
        "seed": config.experiment.seed,
        "data_seed": config.experiment.seed,
        "per_device_train_batch_size": config.trainer.per_device_train_batch_size,
        "gradient_accumulation_steps": config.trainer.gradient_accumulation_steps,
        "num_generations": config.trainer.num_generations,
        "max_prompt_length": config.trainer.max_prompt_length,
        "max_completion_length": config.trainer.max_completion_length,
        "max_steps": config.trainer.max_steps,
        "learning_rate": config.trainer.learning_rate,
        "lr_scheduler_type": config.trainer.lr_scheduler_type,
        "warmup_steps": max(0, round(config.trainer.max_steps * config.trainer.warmup_ratio)),
        "optim": config.trainer.optim,
        "loss_type": config.trainer.loss_type,
        "scale_rewards": config.trainer.scale_rewards,
        "beta": config.trainer.beta,
        "epsilon": config.trainer.epsilon,
        "temperature": config.trainer.temperature,
        "top_p": config.trainer.top_p,
        "bf16": precision.bf16,
        "fp16": precision.fp16,
        "tf32": precision.tf32,
        "gradient_checkpointing": config.trainer.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": config.trainer.max_grad_norm,
        "use_vllm": config.trainer.use_vllm,
        "vllm_mode": config.trainer.vllm_mode,
        "vllm_model_impl": config.trainer.vllm_model_impl,
        "vllm_gpu_memory_utilization": config.trainer.vllm_gpu_memory_utilization,
        "vllm_max_model_len": config.trainer.vllm_max_model_length,
        "vllm_max_model_length": config.trainer.vllm_max_model_length,
        "vllm_tensor_parallel_size": config.trainer.vllm_tensor_parallel_size,
        "vllm_enable_sleep_mode": config.trainer.vllm_enable_sleep_mode,
        "vllm_server_base_url": config.trainer.vllm_server_base_url,
        "vllm_server_host": config.trainer.vllm_server_host,
        "vllm_server_port": config.trainer.vllm_server_port,
        "vllm_server_timeout": config.trainer.vllm_server_timeout,
        "vllm_group_port": config.trainer.vllm_group_port,
        "logging_steps": config.trainer.logging_steps,
        "save_steps": config.trainer.save_steps,
        "eval_strategy": config.trainer.eval_strategy,
        "report_to": config.trainer.report_to,
        "log_completions": config.trainer.log_completions,
        "num_completions_to_print": config.trainer.num_completions_to_print,
        "log_unique_prompts": config.trainer.log_unique_prompts,
        "save_total_limit": config.trainer.save_total_limit,
        "reward_weights": config.rewards.weights,
        "remove_unused_columns": False,
    }
    args = GRPOConfig(**_filter_kwargs(GRPOConfig, grpo_kwargs))
    trainer = BoundaryMarginGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward, boxed_format_reward],
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        margin_config=config.margin,
    )
    return trainer


def run_training(config: RunConfig, resume_from: str | None = None) -> dict[str, Any]:
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from = resolve_resume_checkpoint(config, resume_from)
    if config.tracking.save_resolved_config:
        dump_run_config(config, output_dir / "resolved_config.yaml")
    if config.tracking.save_environment:
        write_environment(output_dir / "environment.json")
    trainer = _build_training_components(config)
    result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir / "final_adapter"))
    metrics = dict(result.metrics)
    metrics["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Boundary-Margin GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_run_config(args.config)
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.output_dir is not None:
        config.experiment.output_dir = args.output_dir
    if args.max_steps is not None:
        config.trainer.max_steps = args.max_steps
    config.validate()
    metrics = run_training(config, args.resume_from)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
