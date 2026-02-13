from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from kalshi_hft.engine import main


if __name__ == "__main__":
    raise SystemExit(main())
