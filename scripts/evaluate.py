#!/usr/bin/env python3
"""Generate the Markdown evaluation/inspection report from installed artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "model"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/generated/evaluation_report.md"),
    )
    args = parser.parse_args()
    path = write_report(args.model_dir, args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
