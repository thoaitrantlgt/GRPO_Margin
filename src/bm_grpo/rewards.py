from __future__ import annotations

import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

CATEGORICAL_ANSWERS = {"yes", "no", "true", "false"}


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        return "".join(completion_text(item) for item in completion)
    return str(completion)


def extract_boxed(text: str) -> list[tuple[str, int, int]]:
    """Return balanced ``\\boxed{...}`` expressions as (content, start, end)."""

    results: list[tuple[str, int, int]] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        depth = 1
        index = start + len(marker)
        content_start = index
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append((text[content_start : index - 1].strip(), start, index))
            cursor = index
        else:
            cursor = start + len(marker)
    return results


def _normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", value)


def _normalize_numeric(value: str) -> Decimal | None:
    cleaned = value.strip().replace(",", "")
    cleaned = cleaned.removeprefix("$").removesuffix("$").strip()
    cleaned = cleaned.rstrip(".%")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _math_verify(prediction: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        gold_parsed = parse(gold)
        prediction_parsed = parse(prediction)
        if not gold_parsed:
            return None
        if not prediction_parsed:
            return False
        return bool(verify(gold_parsed, prediction_parsed))
    except Exception:
        return None


def verify_answer(prediction: str, gold: str, answer_type: str) -> bool | None:
    if answer_type == "categorical":
        normalized_gold = _normalize_text(gold)
        if normalized_gold not in CATEGORICAL_ANSWERS:
            return None
        return _normalize_text(prediction) == normalized_gold
    if answer_type == "numeric":
        gold_value = _normalize_numeric(gold)
        prediction_value = _normalize_numeric(prediction)
        if gold_value is None:
            return None
        return prediction_value == gold_value
    if answer_type == "math":
        verified = _math_verify(prediction, gold)
        if verified is not None:
            return verified
        # The fallback is deliberately conservative and exists for local tests
        # where math-verify is not installed. Paper runs install math-verify.
        return _normalize_text(prediction) == _normalize_text(gold)
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def reference_is_valid(gold: str, answer_type: str) -> bool:
    if answer_type == "categorical":
        return _normalize_text(gold) in CATEGORICAL_ANSWERS
    if answer_type == "numeric":
        return _normalize_numeric(gold) is not None
    if answer_type == "math":
        try:
            from math_verify import parse
        except ImportError as error:
            raise RuntimeError("Math reference validation requires the 'math-verify' package") from error
        try:
            return bool(parse(gold))
        except Exception:
            return False
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def accuracy_reward(
    completions: list[Any],
    reference_answer: list[str],
    answer_type: list[str],
    log_extra: Callable[[str, list[Any]], None] | None = None,
    log_metric: Callable[[str, float], None] | None = None,
    **_: Any,
) -> list[float | None]:
    scores: list[float | None] = []
    parsed_answers: list[str | None] = []
    verifier_valid: list[float] = []
    for completion, gold, kind in zip(completions, reference_answer, answer_type, strict=True):
        boxes = extract_boxed(completion_text(completion))
        prediction = boxes[-1][0] if len(boxes) == 1 and boxes[-1][0] else None
        parsed_answers.append(prediction)
        if prediction is None:
            scores.append(0.0)
            verifier_valid.append(1.0)
            continue
        verified = verify_answer(prediction, gold, kind)
        scores.append(None if verified is None else float(verified))
        verifier_valid.append(float(verified is not None))
    if log_extra is not None:
        log_extra("parsed_answer", parsed_answers)
    if log_metric is not None:
        parse_values = [float(value is not None) for value in parsed_answers]
        log_metric("verifier/parse_rate", sum(parse_values) / len(parse_values) if parse_values else 0.0)
        log_metric("verifier/valid_rate", sum(verifier_valid) / len(verifier_valid) if verifier_valid else 0.0)
    return scores


def boxed_format_reward(
    completions: list[Any],
    log_extra: Callable[[str, list[Any]], None] | None = None,
    **_: Any,
) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        text = completion_text(completion)
        boxes = extract_boxed(text)
        valid = len(boxes) == 1 and bool(boxes[0][0])
        if valid:
            suffix = text[boxes[0][2] :].strip()
            # Accept common display-math wrappers after the final box, e.g.
            # ``\[ \boxed{2} \]`` or ``$$\boxed{2}$$``. Any natural-language
            # content after the box still invalidates the format reward.
            valid = bool(re.fullmatch(r"(?:(?:\\\])|(?:\\\))|(?:\$\$?)|[\s.!?,;:])*", suffix))
        scores.append(float(valid))
    if log_extra is not None:
        log_extra("format_valid", scores)
    return scores
