"""Boundary-Margin GRPO."""

from .config import MarginConfig, RunConfig, load_run_config
from .margin import MarginResult, apply_boundary_margin

__all__ = [
    "MarginConfig",
    "MarginResult",
    "RunConfig",
    "apply_boundary_margin",
    "load_run_config",
]

__version__ = "0.1.0"

