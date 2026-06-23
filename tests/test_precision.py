import pytest
import torch

from bm_grpo.config import RunConfig
from bm_grpo.train import resolve_runtime_precision


def test_pre_ampere_gpu_falls_back_to_fp16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.warns(UserWarning) as warnings:
        precision = resolve_runtime_precision(RunConfig())
    assert len(warnings) == 2
    assert precision.model_dtype == torch.float16
    assert precision.fp16 is True
    assert precision.bf16 is False
    assert precision.tf32 is False


def test_ampere_gpu_keeps_bf16_and_tf32(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    precision = resolve_runtime_precision(RunConfig())
    assert precision.model_dtype == torch.bfloat16
    assert precision.bf16 is True
    assert precision.fp16 is False
    assert precision.tf32 is True

