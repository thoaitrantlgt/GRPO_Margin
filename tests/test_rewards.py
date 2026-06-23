import sys

import pytest

from bm_grpo.rewards import (
    accuracy_reward,
    boxed_format_reward,
    extract_boxed,
    reference_is_valid,
    verify_answer,
)


def test_extract_boxed_handles_nested_latex() -> None:
    boxes = extract_boxed(r"Reasoning. Final: \boxed{\frac{63}{400}}.")
    assert boxes[0][0] == r"\frac{63}{400}"


def test_numeric_and_categorical_verification() -> None:
    assert verify_answer("1,024", "1024", "numeric") is True
    assert verify_answer("Yes", "yes", "categorical") is True
    assert verify_answer("no", "yes", "categorical") is False
    assert reference_is_valid("1,024", "numeric") is True
    assert reference_is_valid("maybe", "categorical") is False


@pytest.mark.skipif(sys.platform == "win32", reason="math-verify worker teardown is unreliable on Windows")
def test_math_verify_equivalence() -> None:
    assert verify_answer(r"\frac{1}{2}", r"\frac{1}{2}", "math") is True


def test_accuracy_requires_single_box() -> None:
    metrics = {}
    scores = accuracy_reward(
        [r"Answer \boxed{42}", r"\boxed{41} then \boxed{42}", "42"],
        ["42", "42", "42"],
        ["numeric", "numeric", "numeric"],
        log_metric=lambda name, value: metrics.__setitem__(name, value),
    )
    assert scores == [1.0, 0.0, 0.0]
    assert metrics["verifier/parse_rate"] == pytest.approx(1 / 3)


def test_format_reward_rejects_content_after_box() -> None:
    scores = boxed_format_reward(
        [r"Work. \boxed{42}.", r"\boxed{}", r"\boxed{42} but maybe 43", r"\boxed{1}\boxed{2}"]
    )
    assert scores == [1.0, 0.0, 0.0, 0.0]


def test_format_reward_accepts_display_math_wrappers() -> None:
    completions = [
        r"\[ \boxed{2} \]",
        r"$$\boxed{2}$$",
        r"\(\boxed{2}\)",
        r"\[\boxed{2}\].",
    ]
    assert boxed_format_reward(completions) == [1.0, 1.0, 1.0, 1.0]
    assert accuracy_reward(completions, ["2"] * 4, ["numeric"] * 4) == [1.0] * 4
