from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bm_grpo.rewards import reference_is_valid

from .adapters import adapt_example
from .normalize import content_fingerprint


@dataclass(slots=True)
class SourceSpec:
    name: str
    dataset: str
    revision: str
    split: str
    role: str
    config: str | None = None


@dataclass(slots=True)
class PrepareConfig:
    profile: str
    output_dir: str
    sources: list[SourceSpec]
    seed: int = 42
    model_tokenizer: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    max_prompt_tokens: int = 512
    validation_reserve: int = 0
    train_limit: int | None = None
    validation_limit: int | None = None
    near_duplicate_threshold: float = 0.95
    near_duplicate_num_perm: int = 64
    deduplicate_against_eval: bool = True
    materialize_roles: list[str] = field(default_factory=lambda: ["train", "validation", "test"])


def load_prepare_config(path: str | Path) -> PrepareConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Data config must be a mapping")
    source_values = raw.pop("sources", None)
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("sources must be a non-empty list")
    allowed = {field.name for field in PrepareConfig.__dataclass_fields__.values()} - {"sources"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown data config keys: {', '.join(unknown)}")
    sources = [SourceSpec(**value) for value in source_values]
    config = PrepareConfig(sources=sources, **raw)
    if config.profile not in {"smoke", "gsm8k", "paper"}:
        raise ValueError("profile must be smoke, gsm8k, or paper")
    if not 0 < config.near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    return config


def _stable_rank(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minhash(problem: str, num_perm: int):
    try:
        from datasketch import MinHash
    except ImportError as error:
        raise RuntimeError("Near-duplicate filtering requires the 'datasketch' package") from error
    value = f"  {problem.lower()}  "
    shingles = {value[index : index + 5] for index in range(max(1, len(value) - 4))}
    sketch = MinHash(num_perm=num_perm)
    for shingle in shingles:
        sketch.update(shingle.encode("utf-8"))
    return sketch


def _filter_near_duplicates(
    train_rows: list[dict[str, Any]],
    protected_rows: list[dict[str, Any]],
    threshold: float,
    num_perm: int,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from datasketch import MinHashLSH
    except ImportError as error:
        raise RuntimeError("Near-duplicate filtering requires the 'datasketch' package") from error
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for index, row in enumerate(protected_rows):
        lsh.insert(f"protected:{index}", _minhash(row["problem"], num_perm))
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in train_rows:
        sketch = _minhash(row["problem"], num_perm)
        if lsh.query(sketch):
            dropped += 1
            continue
        lsh.insert(f"train:{row['example_id']}", sketch)
        kept.append(row)
    return kept, dropped


def _adapt_and_filter(
    dataset, spec: SourceSpec, tokenizer, max_tokens: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"raw": len(dataset), "adapter_error": 0, "invalid_gold": 0, "too_long": 0}
    for example in dataset:
        try:
            row = adapt_example(spec.dataset, dict(example))
        except (KeyError, TypeError, ValueError):
            counts["adapter_error"] += 1
            continue
        if not reference_is_valid(row["reference_answer"], row["answer_type"]):
            counts["invalid_gold"] += 1
            continue
        row["prompt_tokens"] = len(
            tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        )
        if row["prompt_tokens"] > max_tokens:
            counts["too_long"] += 1
            continue
        row["content_hash"] = content_fingerprint(row["problem"])
        rows.append(row)
    counts["valid"] = len(rows)
    return rows, counts


def _deduplicate_exact(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row["content_hash"] in seen:
            continue
        seen.add(row["content_hash"])
        kept.append(row)
    return kept, len(rows) - len(kept)


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "content_hash"}


def prepare(config: PrepareConfig) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Dataset preparation requires bm-grpo[train]") from error

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_tokenizer,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    role_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for spec in config.sources:
        dataset = load_dataset(
            spec.dataset,
            name=spec.config,
            split=spec.split,
            revision=spec.revision,
        )
        rows, counts = _adapt_and_filter(dataset, spec, tokenizer, config.max_prompt_tokens)
        rows, exact_dropped = _deduplicate_exact(rows)
        counts["exact_duplicate"] = exact_dropped
        source_counts[spec.name] = counts
        role_rows.setdefault(spec.role, []).extend(rows)

    protected_roles = [role for role in role_rows if role != "train"]
    protected_rows = [row for role in protected_roles for row in role_rows[role]]
    train_rows = role_rows.get("train", [])
    protected_hashes = {row["content_hash"] for row in protected_rows}
    before_leakage = len(train_rows)
    if config.deduplicate_against_eval:
        train_rows = [row for row in train_rows if row["content_hash"] not in protected_hashes]
        exact_leakage_dropped = before_leakage - len(train_rows)
        train_rows, near_dropped = _filter_near_duplicates(
            train_rows,
            protected_rows,
            config.near_duplicate_threshold,
            config.near_duplicate_num_perm,
        )
    else:
        exact_leakage_dropped = 0
        near_dropped = 0

    train_rows.sort(key=lambda row: _stable_rank(row["example_id"], config.seed))
    if config.validation_reserve:
        reserved = train_rows[: config.validation_reserve]
        train_rows = train_rows[config.validation_reserve :]
        role_rows["validation"] = reserved
    if config.train_limit is not None:
        train_rows = train_rows[: config.train_limit]
    role_rows["train"] = train_rows
    if config.validation_limit is not None and "validation" in role_rows:
        role_rows["validation"].sort(key=lambda row: _stable_rank(row["example_id"], config.seed))
        role_rows["validation"] = role_rows["validation"][: config.validation_limit]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    token_lengths: list[int] = []
    for role, rows in role_rows.items():
        if role not in config.materialize_roles:
            continue
        clean_rows = [_strip_internal(row) for row in rows]
        if not clean_rows:
            continue
        path = output_dir / f"{role}.parquet"
        pq.write_table(pa.Table.from_pylist(clean_rows), path)
        token_lengths.extend(row["prompt_tokens"] for row in clean_rows)
        files[role] = {"path": path.name, "rows": len(clean_rows), "sha256": _sha256_file(path)}

    sorted_lengths = sorted(token_lengths)
    token_stats = {}
    if sorted_lengths:
        token_stats = {
            "min": sorted_lengths[0],
            "median": sorted_lengths[len(sorted_lengths) // 2],
            "p95": sorted_lengths[int(0.95 * (len(sorted_lengths) - 1))],
            "max": sorted_lengths[-1],
        }
    manifest = {
        "profile": config.profile,
        "seed": config.seed,
        "config": {**asdict(config), "sources": [asdict(spec) for spec in config.sources]},
        "sources": source_counts,
        "filtering": {
            "exact_train_eval_leakage": exact_leakage_dropped,
            "near_duplicate": near_dropped,
        },
        "token_stats": token_stats,
        "files": files,
    }
    manifest["dataset_fingerprint"] = hashlib.sha256(
        "".join(info["sha256"] for _, info in sorted(files.items())).encode()
    ).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Boundary-Margin GRPO datasets")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare(load_prepare_config(args.config))
    print(json.dumps({"profile": manifest["profile"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
