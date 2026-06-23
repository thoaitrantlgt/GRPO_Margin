import pytest

import bm_grpo.trainer as trainer_module


def test_trl_version_guard_rejects_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(trainer_module, "version", lambda _: "9.9.9")
    with pytest.raises(RuntimeError, match="supports trl==1.6.0"):
        trainer_module.require_supported_trl()

