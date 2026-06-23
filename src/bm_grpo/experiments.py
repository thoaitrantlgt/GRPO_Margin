from __future__ import annotations

import argparse
import copy
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


def build_runs(matrix_path: str | Path, materialize: bool = True) -> list[tuple[str, Path, list[str]]]:
    matrix_path = Path(matrix_path)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    base_path = matrix_path.parent / matrix["base_config"]
    base = yaml.safe_load(base_path.resolve().read_text(encoding="utf-8"))
    generated_dir = (matrix_path.parent / matrix.get("generated_dir", "generated")).resolve()
    if materialize:
        generated_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, Path, list[str]]] = []
    for variant in matrix["variants"]:
        for seed in matrix["seeds"]:
            name = f"{variant['name']}_seed{seed}"
            config = _deep_merge(copy.deepcopy(base), variant.get("overrides", {}))
            config.setdefault("experiment", {})["name"] = name
            config["experiment"]["seed"] = seed
            config["experiment"]["output_dir"] = f"outputs/{name}"
            config_path = generated_dir / f"{name}.yaml"
            if materialize:
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [
                "accelerate",
                "launch",
                "--config_file",
                matrix["accelerate_config"],
                "-m",
                "bm_grpo.train",
                "--config",
                str(config_path),
            ]
            runs.append((name, config_path, command))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Boundary-Margin experiment matrix")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name, _, command in build_runs(args.matrix):
        done = Path("outputs") / name / "train_metrics.json"
        printable = subprocess.list2cmdline(command)
        if done.exists() and not args.force:
            print(f"SKIP {name}: {done} exists")
            continue
        print(printable)
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
