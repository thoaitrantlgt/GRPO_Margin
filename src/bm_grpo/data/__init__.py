"""Dataset preparation utilities for Boundary-Margin GRPO."""

from .adapters import SYSTEM_PROMPT, adapt_example
from .normalize import answer_type_for, normalize_problem, problem_fingerprint

__all__ = [
    "SYSTEM_PROMPT",
    "adapt_example",
    "answer_type_for",
    "normalize_problem",
    "problem_fingerprint",
]

