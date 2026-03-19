from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_local_layout(project_root: Path) -> None:
    src_dir = project_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = project_root / ".venv" / "lib" / version / "site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


bootstrap_local_layout(Path(__file__).resolve().parent)

from mars_rocks import main


if __name__ == "__main__":
    main()
