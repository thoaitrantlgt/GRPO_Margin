from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .normalize import content_fingerprint

REQUIRED_COLUMNS = {
    "example_id",
    "source",
    "prompt",
    "problem",
    "solution",
    "reference_answer",
    "answer_type",
    "prompt_tokens",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(manifest_path: str | Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Dataset audit requires bm-grpo[train]") from error

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    seen_by_role: dict[str, set[str]] = {}
    content_by_role: dict[str, set[str]] = {}
    errors: list[str] = []
    for role, info in manifest.get("files", {}).items():
        path = base_dir / info["path"]
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        if _sha256_file(path) != info["sha256"]:
            errors.append(f"checksum mismatch: {path.name}")
        table = pq.read_table(path)
        missing = REQUIRED_COLUMNS - set(table.column_names)
        if missing:
            errors.append(f"{role} missing columns: {sorted(missing)}")
            continue
        rows = table.to_pylist()
        ids = [row["example_id"] for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate example_id in {role}")
        if any(int(row["prompt_tokens"]) > manifest["config"]["max_prompt_tokens"] for row in rows):
            errors.append(f"prompt over token limit in {role}")
        if any(not str(row["reference_answer"]).strip() for row in rows):
            errors.append(f"empty reference answer in {role}")
        seen_by_role[role] = set(ids)
        content_by_role[role] = {content_fingerprint(str(row["problem"])) for row in rows}
    roles = sorted(seen_by_role)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = seen_by_role[left] & seen_by_role[right]
            if overlap:
                errors.append(f"{left}/{right} example_id overlap: {len(overlap)}")
            content_overlap = content_by_role[left] & content_by_role[right]
            if content_overlap:
                errors.append(f"{left}/{right} problem-content overlap: {len(content_overlap)}")
    if errors:
        raise ValueError("Dataset audit failed:\n- " + "\n- ".join(errors))
    return {"profile": manifest["profile"], "status": "ok", "files": sorted(seen_by_role)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prepared Boundary-Margin datasets")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
