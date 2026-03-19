from __future__ import annotations

import os
from pathlib import Path


def _configure_matplotlib_cache() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return

    cache_dir = Path.cwd() / ".cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_configure_matplotlib_cache()

from cli import main


if __name__ == "__main__":
    main()
