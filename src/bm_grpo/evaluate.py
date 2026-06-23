from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml

from .rewards import accuracy_reward, boxed_format_reward, extract_boxed


def _bootstrap_interval(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def _batched(values: list[Any], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("Evaluation requires bm-grpo[train]") from error

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    random.seed(int(raw["seed"]))
    torch.manual_seed(int(raw["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(raw["seed"]))
    model_config = raw["model"]
    generation = raw["generation"]
    output_dir = Path(output_dir_override or raw["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        revision=model_config["revision"],
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"], revision=model_config["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_metrics: dict[str, Any] = {}
    all_completions: list[dict[str, Any]] = []
    for dataset_name, path in raw["datasets"].items():
        dataset = load_dataset("parquet", data_files=path, split="train")
        rows = [dict(row) for row in dataset]
        for mode_name, mode in generation.items():
            correctness: list[float] = []
            format_scores: list[float] = []
            parse_scores: list[float] = []
            scores_by_example: dict[str, list[float]] = {}
            for batch in _batched(rows, int(mode["batch_size"])):
                prompts = [
                    tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                    for row in batch
                ]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                do_sample = bool(mode["temperature"] > 0)
                with torch.inference_mode():
                    generation_kwargs = {
                        "do_sample": do_sample,
                        "num_return_sequences": int(mode["num_return_sequences"]),
                        "max_new_tokens": int(mode["max_new_tokens"]),
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if do_sample:
                        generation_kwargs["temperature"] = float(mode["temperature"])
                        generation_kwargs["top_p"] = float(mode.get("top_p", 1.0))
                    generated = model.generate(**inputs, **generation_kwargs)
                prompt_length = inputs["input_ids"].shape[1]
                texts = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
                expanded_rows = [row for row in batch for _ in range(int(mode["num_return_sequences"]))]
                accuracy = accuracy_reward(
                    texts,
                    [row["reference_answer"] for row in expanded_rows],
                    [row["answer_type"] for row in expanded_rows],
                )
                formats = boxed_format_reward(texts)
                for row, text, acc, fmt in zip(expanded_rows, texts, accuracy, formats, strict=True):
                    score = 0.0 if acc is None else float(acc)
                    correctness.append(score)
                    format_scores.append(fmt)
                    parse_scores.append(float(len(extract_boxed(text)) == 1))
                    scores_by_example.setdefault(row["example_id"], []).append(score)
                    all_completions.append(
                        {
                            "dataset": dataset_name,
                            "mode": mode_name,
                            "example_id": row["example_id"],
                            "completion": text,
                            "correct": score,
                            "format_valid": fmt,
                        }
                    )
            pass_scores = [float(any(score > 0 for score in scores)) for scores in scores_by_example.values()]
            low, high = _bootstrap_interval(pass_scores, int(raw["seed"]))
            all_metrics[f"{dataset_name}/{mode_name}"] = {
                "pass_at_k": mean(pass_scores) if pass_scores else 0.0,
                "completion_accuracy": mean(correctness) if correctness else 0.0,
                "format_rate": mean(format_scores) if format_scores else 0.0,
                "parse_rate": mean(parse_scores) if parse_scores else 0.0,
                "pass_at_k_ci95": [low, high],
                "num_completions": len(correctness),
                "num_examples": len(pass_scores),
            }
    pq.write_table(pa.Table.from_pylist(all_completions), output_dir / "completions.parquet")
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Boundary-Margin GRPO adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.checkpoint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
