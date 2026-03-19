"""Mars rock segmentation package."""

from __future__ import annotations

import os
from pathlib import Path

_MATPLOTLIB_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))
