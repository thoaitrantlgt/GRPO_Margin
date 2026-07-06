from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import RunConfig, load_run_config

TRAIN_METRICS = (
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "peak_gpu_memory_bytes",
)
EVAL_METRICS = (
    "pass_at_k",
    "completion_accuracy",
    "format_rate",
    "parse_rate",
)
TRAIN_CONFIG_KEYS = (
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "max_completion_length",
    "num_generations",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "use_vllm",
)


@dataclass(slots=True)
class PairConfig:
    name: str
    baseline_config: Path
    method_config: Path
    evaluation_config: Path
    accelerate_config: str
    output_dir: Path
    checkpoint_subdir: str = "final_adapter"


def load_pair_config(path: str | Path) -> PairConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {
        "name",
        "baseline_config",
        "method_config",
        "evaluation_config",
        "accelerate_config",
        "output_dir",
        "checkpoint_subdir",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown paired comparison keys: {', '.join(unknown)}")
    base = path.parent
    return PairConfig(
        name=raw["name"],
        baseline_config=(base / raw["baseline_config"]).resolve(),
        method_config=(base / raw["method_config"]).resolve(),
        evaluation_config=(base / raw["evaluation_config"]).resolve(),
        accelerate_config=raw["accelerate_config"],
        output_dir=Path(raw["output_dir"]),
        checkpoint_subdir=raw.get("checkpoint_subdir", "final_adapter"),
    )


def validate_controlled_pair(baseline: RunConfig, method: RunConfig) -> None:
    controlled_sections = ("model", "data", "rewards", "trainer", "tracking")
    mismatched = [
        section
        for section in controlled_sections
        if dataclasses.asdict(getattr(baseline, section)) != dataclasses.asdict(getattr(method, section))
    ]
    if baseline.experiment.seed != method.experiment.seed:
        mismatched.append("experiment.seed")
    if mismatched:
        raise ValueError(f"Baseline/method are not controlled; mismatched: {', '.join(mismatched)}")
    if baseline.margin.enabled:
        raise ValueError("Baseline config must set margin.enabled=false")
    if not method.margin.enabled:
        raise ValueError("Method config must set margin.enabled=true")


def _metric_delta(baseline: float, method: float) -> dict[str, float | None]:
    delta = method - baseline
    relative = None if baseline == 0 else 100.0 * delta / abs(baseline)
    return {
        "baseline": baseline,
        "method": method,
        "delta": delta,
        "relative_percent": relative,
    }


def build_comparison(
    baseline_train: dict[str, Any],
    method_train: dict[str, Any],
    baseline_eval: dict[str, Any] | None = None,
    method_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training: dict[str, Any] = {}
    for metric in TRAIN_METRICS:
        left = baseline_train.get(metric)
        right = method_train.get(metric)
        if isinstance(left, int | float) and isinstance(right, int | float):
            training[metric] = _metric_delta(float(left), float(right))

    evaluation: dict[str, Any] = {}
    baseline_eval = baseline_eval or {}
    method_eval = method_eval or {}
    for benchmark in sorted(set(baseline_eval) & set(method_eval)):
        benchmark_metrics: dict[str, Any] = {}
        for metric in EVAL_METRICS:
            left = baseline_eval[benchmark].get(metric)
            right = method_eval[benchmark].get(metric)
            if isinstance(left, int | float) and isinstance(right, int | float):
                benchmark_metrics[metric] = _metric_delta(float(left), float(right))
        for metadata in ("num_examples", "num_completions", "pass_at_k_ci95"):
            left = baseline_eval[benchmark].get(metadata)
            right = method_eval[benchmark].get(metadata)
            if left is not None or right is not None:
                benchmark_metrics[metadata] = {"baseline": left, "method": right}
        evaluation[benchmark] = benchmark_metrics
    return {"training": training, "evaluation": evaluation}


def _format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _format_metadata(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return _format_value(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_metadata(item) for item in value) + "]"
    return str(value)


def _runtime_notes(
    baseline: RunConfig,
    method: RunConfig,
    baseline_train: dict[str, Any],
    method_train: dict[str, Any],
) -> list[str]:
    notes = [
        "Training runtime and memory are operational diagnostics, not method-quality claims. "
        "Use evaluation deltas for model-quality comparison.",
        "GRPO surrogate loss can be near zero or negative; relative loss percentages are intentionally not reported.",
    ]
    baseline_trainer = dataclasses.asdict(baseline.trainer)
    method_trainer = dataclasses.asdict(method.trainer)
    mismatched = [
        key
        for key in TRAIN_CONFIG_KEYS
        if baseline_trainer.get(key) != method_trainer.get(key)
    ]
    if mismatched:
        notes.append(
            "Baseline and Boundary-Margin train configs differ on: "
            + ", ".join(mismatched)
            + ". Runtime/memory deltas are not controlled."
        )
    left_runtime = baseline_train.get("train_runtime")
    right_runtime = method_train.get("train_runtime")
    if isinstance(left_runtime, int | float) and isinstance(right_runtime, int | float):
        larger = max(float(left_runtime), float(right_runtime))
        smaller = max(min(float(left_runtime), float(right_runtime)), 1e-9)
        if larger / smaller >= 1.25:
            notes.append(
                "Large runtime gap detected. Check whether both runs used the same batch size, "
                "resume state, hardware, and generation backend before citing speed."
            )
    return notes


def comparison_markdown(
    name: str,
    comparison: dict[str, Any],
    *,
    baseline: RunConfig | None = None,
    method: RunConfig | None = None,
    baseline_train: dict[str, Any] | None = None,
    method_train: dict[str, Any] | None = None,
) -> str:
    notes: list[str] = []
    if baseline is not None and method is not None:
        notes = _runtime_notes(baseline, method, baseline_train or {}, method_train or {})
    lines = [
        f"# GRPO vs Boundary-Margin: {name}",
        "",
        "Positive evaluation delta means Boundary-Margin is better. Training runtime/memory deltas are raw method "
        "minus baseline.",
        "",
    ]
    if notes:
        lines.extend(["## Read this first", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    lines.extend(
        [
        "## Training",
        "",
        "| Metric | GRPO | Boundary-Margin | Delta | Relative |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, values in comparison["training"].items():
        relative = values["relative_percent"]
        relative_text = "N/A" if metric == "train_loss" or relative is None else f"{relative:.2f}%"
        lines.append(
            f"| {metric} | {_format_value(values['baseline'])} | {_format_value(values['method'])} | "
            f"{_format_value(values['delta'])} | {relative_text} |"
        )
    lines.extend(["", "## Evaluation", ""])
    for benchmark, metrics in comparison["evaluation"].items():
        lines.extend(
            [
                f"### {benchmark}",
                "",
                "| Metadata | GRPO | Boundary-Margin |",
                "|---|---:|---:|",
            ]
        )
        for metadata in ("num_examples", "num_completions", "pass_at_k_ci95"):
            values = metrics.get(metadata)
            if isinstance(values, dict):
                lines.append(
                    f"| {metadata} | {_format_metadata(values.get('baseline'))} | "
                    f"{_format_metadata(values.get('method'))} |"
                )
        lines.extend(
            [
                "",
                "| Metric | GRPO | Boundary-Margin | Delta | Relative |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric, values in metrics.items():
            if metric not in EVAL_METRICS:
                continue
            relative = values["relative_percent"]
            lines.append(
                f"| {metric} | {_format_value(values['baseline'])} | {_format_value(values['method'])} | "
                f"{_format_value(values['delta'])} | {'N/A' if relative is None else f'{relative:.2f}%'} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = [
        path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))
    return checkpoints[-1]


def _latest_checkpoint_step(output_dir: Path) -> int | None:
    latest = _latest_checkpoint(output_dir)
    if latest is None:
        return None
    return _checkpoint_step(latest)


def _needs_more_training(output_dir: Path, max_steps: int) -> bool:
    step = _latest_checkpoint_step(output_dir)
    return step is None or step < max_steps


def _eval_checkpoint(output_dir: Path, fallback_subdir: str) -> Path:
    final_adapter = output_dir / fallback_subdir
    if final_adapter.exists():
        return final_adapter
    latest = _latest_checkpoint(output_dir)
    if latest is not None:
        return latest
    raise FileNotFoundError(f"No evaluation checkpoint found in {output_dir}")


def write_comparison_report(pair: PairConfig, baseline: RunConfig, method: RunConfig) -> dict[str, Any]:
    baseline_dir = Path(baseline.experiment.output_dir)
    method_dir = Path(method.experiment.output_dir)
    baseline_train = _read_json(baseline_dir / "train_metrics.json")
    method_train = _read_json(method_dir / "train_metrics.json")
    comparison = build_comparison(
        baseline_train,
        method_train,
        _read_json(baseline_dir / "eval" / "metrics.json"),
        _read_json(method_dir / "eval" / "metrics.json"),
    )
    payload = {
        "name": pair.name,
        "seed": baseline.experiment.seed,
        "baseline_output": str(baseline_dir),
        "method_output": str(method_dir),
        **comparison,
    }
    pair.output_dir.mkdir(parents=True, exist_ok=True)
    (pair.output_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = comparison_markdown(
        pair.name,
        comparison,
        baseline=baseline,
        method=method,
        baseline_train=baseline_train,
        method_train=method_train,
    )
    (pair.output_dir / "comparison.md").write_text(report, encoding="utf-8")
    Path("comparison.md").write_text(report, encoding="utf-8")
    return payload


def _train_command(config_path: Path, accelerate_config: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        accelerate_config,
        "-m",
        "bm_grpo.train",
        "--config",
        str(config_path),
    ]


def _eval_command(eval_config: Path, checkpoint: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bm_grpo.evaluate",
        "--config",
        str(eval_config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]


def run_pair(
    pair: PairConfig,
    dry_run: bool = False,
    force: bool = False,
    report_only: bool = False,
) -> dict[str, Any] | None:
    baseline = load_run_config(pair.baseline_config)
    method = load_run_config(pair.method_config)
    validate_controlled_pair(baseline, method)

    if not report_only:
        for config_path, config in ((pair.baseline_config, baseline), (pair.method_config, method)):
            run_dir = Path(config.experiment.output_dir)
            latest = _latest_checkpoint(run_dir)
            needs_train = _needs_more_training(run_dir, config.trainer.max_steps)
            command = _train_command(config_path, pair.accelerate_config)
            if not needs_train and not force:
                print(
                    f"SKIP TRAIN {config.experiment.name}: latest checkpoint has reached "
                    f"max_steps={config.trainer.max_steps}"
                )
            else:
                if latest is not None:
                    command.extend(["--resume-from", str(latest)])
                print(
                    f"TRAIN {config.experiment.name}: "
                    + (f"resuming from {latest}" if latest is not None else "starting from scratch")
                )
                print(subprocess.list2cmdline(command))
                if not dry_run:
                    subprocess.run(command, check=True)

        for config in (baseline, method):
            run_dir = Path(config.experiment.output_dir)
            checkpoint = _eval_checkpoint(run_dir, pair.checkpoint_subdir)
            eval_dir = run_dir / "eval"
            marker = eval_dir / "metrics.json"
            command = _eval_command(pair.evaluation_config, checkpoint, eval_dir)
            if marker.exists() and not force:
                print(f"SKIP EVAL {config.experiment.name}: {marker} exists")
            else:
                print(subprocess.list2cmdline(command))
                if not dry_run:
                    subprocess.run(command, check=True)

    if dry_run:
        return None
    return write_comparison_report(pair, baseline, method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare GRPO with Boundary-Margin GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = run_pair(
        load_pair_config(args.config),
        dry_run=args.dry_run,
        force=args.force,
        report_only=args.report_only,
    )
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
