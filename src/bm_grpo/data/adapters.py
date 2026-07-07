from __future__ import annotations

from typing import Any

from .normalize import answer_type_for, normalize_answer, normalize_problem, problem_fingerprint

SYSTEM_PROMPT = (
    "Solve the problem carefully. You may show your reasoning, but finish with "
    "exactly one final answer in the form \\boxed{...}."
)

MATHARENA_AIME_SOURCES = {
    "MathArena/aime_2025": "aime25",
    "MathArena/aime_2026": "aime26",
}


def _deepmath_problem(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    raise ValueError("DeepMath prompt has no user content")


def adapt_example(dataset_id: str, example: dict[str, Any]) -> dict[str, Any]:
    if dataset_id == "openai/gsm8k":
        problem = str(example["question"])
        solution = str(example["answer"])
        if "####" not in solution:
            raise ValueError("GSM8K answer does not contain ####")
        reference_answer = solution.rsplit("####", 1)[1].strip()
        source = "gsm8k"
    elif dataset_id == "trl-lib/DeepMath-103K":
        problem = _deepmath_problem(example["prompt"])
        solution = str(example["solution"])
        reference_answer = solution.strip()
        source = "deepmath_103k"
    elif dataset_id == "HuggingFaceH4/MATH-500":
        problem = str(example["problem"])
        solution = str(example["solution"])
        reference_answer = str(example["answer"])
        source = "math500"
    elif dataset_id == "HuggingFaceH4/aime_2024":
        problem = str(example["problem"])
        solution = str(example["solution"])
        reference_answer = str(example["answer"])
        source = "aime24"
    elif dataset_id in MATHARENA_AIME_SOURCES:
        problem = str(example["problem"])
        solution = str(example.get("solution", ""))
        reference_answer = str(example["answer"])
        source = MATHARENA_AIME_SOURCES[dataset_id]
    else:
        raise ValueError(f"No adapter registered for dataset: {dataset_id}")

    problem = normalize_problem(problem)
    solution = solution.strip()
    reference_answer = normalize_answer(reference_answer)
    if not problem or not reference_answer:
        raise ValueError("problem and reference_answer must be non-empty")
    return {
        "example_id": problem_fingerprint(source, problem),
        "source": source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "problem": problem,
        "solution": solution,
        "reference_answer": reference_answer,
        "answer_type": answer_type_for(reference_answer),
        "prompt_tokens": 0,
    }
