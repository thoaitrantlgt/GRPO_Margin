from __future__ import annotations

import hashlib
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from bm_grpo.rewards import CATEGORICAL_ANSWERS


def normalize_problem(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    return "\n".join(lines).strip()


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", value)


def answer_type_for(value: str) -> str:
    normalized = normalize_answer(value)
    if normalized.lower() in CATEGORICAL_ANSWERS:
        return "categorical"
    numeric = normalized.replace(",", "").removeprefix("$").removesuffix("$").strip()
    try:
        Decimal(numeric)
        return "numeric"
    except InvalidOperation:
        return "math"


def problem_fingerprint(source: str, normalized_problem: str) -> str:
    payload = f"{source}\n{normalized_problem}".encode()
    return hashlib.sha256(payload).hexdigest()


def content_fingerprint(normalized_problem: str) -> str:
    return hashlib.sha256(normalized_problem.encode("utf-8")).hexdigest()
