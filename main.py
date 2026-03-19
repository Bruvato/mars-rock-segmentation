from __future__ import annotations

from pathlib import Path

from app.bootstrap import bootstrap_local_venv

bootstrap_local_venv(Path(__file__).resolve().parent)

from app.cli import main


if __name__ == "__main__":
    main()
