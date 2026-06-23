from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig, dump_run_config, load_run_config
from .rewards import accuracy_reward, boxed_format_reward
from .runtime import write_environment
from .trainer import BoundaryMarginGRPOTrainer


def _build_training_components(config: RunConfig):
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig
    except ImportError as error:
        raise RuntimeError("Training requires bm-grpo[train]") from error

    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen 1.5B QLoRA profile requires a CUDA GPU")
    dtype = torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("parquet", data_files=config.data.train_path, split="train")
    peft_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    args = GRPOConfig(
        output_dir=config.experiment.output_dir,
        seed=config.experiment.seed,
        data_seed=config.experiment.seed,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        num_generations=config.trainer.num_generations,
        max_completion_length=config.trainer.max_completion_length,
        max_steps=config.trainer.max_steps,
        learning_rate=config.trainer.learning_rate,
        lr_scheduler_type=config.trainer.lr_scheduler_type,
        warmup_ratio=config.trainer.warmup_ratio,
        optim=config.trainer.optim,
        loss_type=config.trainer.loss_type,
        scale_rewards=config.trainer.scale_rewards,
        beta=config.trainer.beta,
        epsilon=config.trainer.epsilon,
        temperature=config.trainer.temperature,
        top_p=config.trainer.top_p,
        bf16=config.trainer.bf16,
        tf32=config.trainer.tf32,
        gradient_checkpointing=config.trainer.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=config.trainer.max_grad_norm,
        use_vllm=config.trainer.use_vllm,
        logging_steps=config.trainer.logging_steps,
        save_steps=config.trainer.save_steps,
        eval_strategy=config.trainer.eval_strategy,
        report_to=config.trainer.report_to,
        log_completions=config.trainer.log_completions,
        save_total_limit=config.trainer.save_total_limit,
        reward_weights=config.rewards.weights,
        remove_unused_columns=False,
    )
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
