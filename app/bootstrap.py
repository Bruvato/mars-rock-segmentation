from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_local_venv(project_root: Path) -> None:
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate = project_root / ".venv" / "lib" / version / "site-packages"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
