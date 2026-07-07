from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


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


def _run_has_reached_max_steps(output_dir: Path, max_steps: int) -> bool:
    latest = _latest_checkpoint(output_dir)
    return latest is not None and _checkpoint_step(latest) >= max_steps


def _final_adapter_or_checkpoint(output_dir: Path) -> Path:
    final_adapter = output_dir / "final_adapter"
    if final_adapter.exists():
        return final_adapter
    latest = _latest_checkpoint(output_dir)
    if latest is not None:
        return latest
    raise FileNotFoundError(f"No checkpoint found for {output_dir}")


def _run_is_complete(output_dir: Path, max_steps: int) -> bool:
    return (
        (output_dir / "train_metrics.json").exists()
        and (output_dir / "final_adapter").exists()
        and _run_has_reached_max_steps(output_dir, max_steps)
    )


def build_runs(matrix_path: str | Path, materialize: bool = True) -> list[tuple[str, Path, list[str]]]:
    matrix_path = Path(matrix_path)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    base_path = matrix_path.parent / matrix["base_config"]
    base = yaml.safe_load(base_path.resolve().read_text(encoding="utf-8"))
    generated_dir = (matrix_path.parent / matrix.get("generated_dir", "generated")).resolve()
    name_prefix = matrix.get("name_prefix")
    if materialize:
        generated_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, Path, list[str]]] = []
    for variant in matrix["variants"]:
        for seed in matrix["seeds"]:
            name_parts = [part for part in (name_prefix, variant["name"], f"seed{seed}") if part]
            name = "_".join(name_parts)
            config = _deep_merge(copy.deepcopy(base), variant.get("overrides", {}))
            config.setdefault("experiment", {})["name"] = name
            config["experiment"]["seed"] = seed
            config["experiment"]["output_dir"] = f"outputs/{name}"
            config_path = generated_dir / f"{name}.yaml"
            if materialize:
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--config_file",
                matrix["accelerate_config"],
                "-m",
                "bm_grpo.train",
                "--config",
                str(config_path),
            ]
            runs.append((name, config_path, command))
    return runs


def _load_max_steps(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(config.get("trainer", {}).get("max_steps", 0))


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_ablation_summary(runs: list[tuple[str, Path, list[str]]], output_dir: Path) -> None:
    payload: dict[str, Any] = {}
    for name, _, _ in runs:
        metrics_path = Path("outputs") / name / "eval" / "metrics.json"
        if metrics_path.exists():
            payload[name] = _read_json(metrics_path)
    if not payload:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    benchmarks = sorted({benchmark for metrics in payload.values() for benchmark in metrics})
    lines = [
        "# Ablation summary",
        "",
        "Higher `pass_at_k`, `completion_accuracy`, `format_rate`, and `parse_rate` are better.",
        "",
    ]
    for benchmark in benchmarks:
        lines.extend(
            [
                f"## {benchmark}",
                "",
                "| Variant | pass_at_k | completion_accuracy | format_rate | parse_rate | num_examples |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in sorted(payload):
            values = payload[name].get(benchmark, {})
            lines.append(
                f"| {name} | {_format_number(values.get('pass_at_k'))} | "
                f"{_format_number(values.get('completion_accuracy'))} | "
                f"{_format_number(values.get('format_rate'))} | "
                f"{_format_number(values.get('parse_rate'))} | "
                f"{_format_number(values.get('num_examples'))} |"
            )
        lines.append("")
    report = "\n".join(lines).rstrip() + "\n"
    (output_dir / "ablation_summary.md").write_text(report, encoding="utf-8")
    Path("ablation_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Boundary-Margin experiment matrix")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip all training and only evaluate existing final_adapter or latest checkpoint for each run.",
    )
    args = parser.parse_args()
    matrix_path = Path(args.matrix)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    runs = build_runs(matrix_path, materialize=not (args.dry_run or args.eval_only))
    if args.eval_only:
        print("EVAL-ONLY: skipping all training commands.")
    else:
        for name, config_path, command in runs:
            run_dir = Path("outputs") / name
            max_steps = _load_max_steps(config_path) if config_path.exists() else 0
            printable = subprocess.list2cmdline(command)
            if _run_is_complete(run_dir, max_steps) and not args.force:
                print(f"SKIP {name}: completed run found in {run_dir}")
                continue
            latest = _latest_checkpoint(run_dir)
            if latest is not None:
                command = [*command, "--resume-from", str(latest)]
                printable = subprocess.list2cmdline(command)
            print(printable)
            if not args.dry_run:
                completed = subprocess.run(command, check=False)
                if completed.returncode:
                    sys.exit(completed.returncode)

    evaluation_config = matrix.get("evaluation_config")
    if evaluation_config:
        eval_config = (matrix_path.parent / evaluation_config).resolve()
        for name, _, _ in runs:
            run_dir = Path("outputs") / name
            eval_dir = run_dir / "eval"
            marker = eval_dir / "metrics.json"
            if marker.exists() and not args.force:
                print(f"SKIP EVAL {name}: {marker} exists")
                continue
            checkpoint = run_dir / "final_adapter" if args.dry_run else _final_adapter_or_checkpoint(run_dir)
            command = _eval_command(eval_config, checkpoint, eval_dir)
            print(subprocess.list2cmdline(command))
            if not args.dry_run:
                completed = subprocess.run(command, check=False)
                if completed.returncode:
                    sys.exit(completed.returncode)
        if not args.dry_run:
            summary_dir = Path(matrix.get("summary_dir", "outputs/ablations")) / matrix_path.stem
            write_ablation_summary(runs, summary_dir)


if __name__ == "__main__":
    main()
