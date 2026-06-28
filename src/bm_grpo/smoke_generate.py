from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import load_run_config


def smoke_generate(config_path: str | Path, prompt: str) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("Smoke generation requires bm-grpo[train]") from error

    config = load_run_config(config_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=config.model.use_bnb_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=False,
    )
    messages = [
        {"role": "system", "content": "Solve the problem. End with exactly one final answer in \\boxed{...}."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **config.trainer.chat_template_kwargs,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=min(config.trainer.max_completion_length, 256),
            do_sample=True,
            temperature=config.trainer.temperature,
            top_p=config.trainer.top_p,
            top_k=config.trainer.top_k,
            repetition_penalty=config.trainer.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=False)
    return {"prompt": text, "completion": completion}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test model generation with the train config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="What is 2+2?")
    args = parser.parse_args()
    print(json.dumps(smoke_generate(args.config, args.prompt), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
