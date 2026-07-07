import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from bm_grpo.data.adapters import SYSTEM_PROMPT, adapt_example
from bm_grpo.data.audit import audit_manifest
from bm_grpo.data.normalize import answer_type_for, normalize_problem
from bm_grpo.data.prepare import _filter_near_duplicates


def test_gsm8k_adapter_extracts_hash_answer() -> None:
    row = adapt_example(
        "openai/gsm8k",
        {"question": "What is 40 + 2?", "answer": "Add them.\n#### 42"},
    )
    assert row["reference_answer"] == "42"
    assert row["answer_type"] == "numeric"
    assert row["prompt"][1]["content"] == "What is 40 + 2?"


def test_matharena_aime_adapter_uses_problem_and_answer() -> None:
    row = adapt_example(
        "MathArena/aime_2025",
        {"problem": "Find $2+2$.", "answer": 4, "problem_idx": 1},
    )
    assert row["source"] == "aime25"
    assert row["reference_answer"] == "4"
    assert row["answer_type"] == "numeric"
    assert row["solution"] == ""


def test_math_and_categorical_types() -> None:
    assert answer_type_for("Yes") == "categorical"
    assert answer_type_for("1,024") == "numeric"
    assert answer_type_for(r"\frac{1}{2}") == "math"


def test_normalization_is_stable() -> None:
    assert normalize_problem(" A  +   B\r\n\r\n C ") == "A + B\n\nC"


def test_near_duplicate_filter_removes_protected_problem() -> None:
    protected = [{"example_id": "eval", "problem": "Compute the value of 40 plus 2."}]
    train = [
        {"example_id": "same", "problem": "Compute the value of 40 plus 2."},
        {"example_id": "other", "problem": "Find the area of a circle with radius seven."},
    ]
    kept, dropped = _filter_near_duplicates(train, protected, threshold=0.95, num_perm=64)
    assert dropped == 1
    assert [row["example_id"] for row in kept] == ["other"]


def test_audit_accepts_valid_materialized_dataset(tmp_path) -> None:
    rows = [
        {
            "example_id": "one",
            "source": "synthetic",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is 1 + 1?"},
            ],
            "problem": "What is 1 + 1?",
            "solution": "2",
            "reference_answer": "2",
            "answer_type": "numeric",
            "prompt_tokens": 16,
        }
    ]
    data_path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data_path)
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = {
        "profile": "synthetic",
        "config": {"max_prompt_tokens": 512},
        "files": {"train": {"path": data_path.name, "rows": 1, "sha256": checksum}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert audit_manifest(manifest_path)["status"] == "ok"
