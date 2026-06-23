from __future__ import annotations

import json
import os
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def environment_info() -> dict[str, Any]:
    import torch

    packages = sorted(
        {distribution.metadata["Name"]: distribution.version for distribution in distributions()}.items()
    )
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda_available else None,
        "packages": dict(packages),
    }


def write_environment(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")

