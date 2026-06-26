from pathlib import Path

import pytest

from bm_grpo.train import _require_existing_path


def test_require_existing_path_raises_helpful_error(tmp_path) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError, match="Run `python -m bm_grpo.data.prepare"):
        _require_existing_path(str(missing), "training dataset")


def test_require_existing_path_accepts_existing_file(tmp_path) -> None:
    existing = tmp_path / "train.parquet"
    existing.write_text("ok", encoding="utf-8")
    _require_existing_path(str(existing), "training dataset")
