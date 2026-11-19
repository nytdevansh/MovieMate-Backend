#!/usr/bin/env python3
"""Create kaggle_source.zip: upload it as a Kaggle dataset for the notebook.

The ZIP contains the shared `ml` package and the Kaggle trainer, so the
notebook uses exactly the same preprocessing and model logic as local training.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_bundle(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in (PROJECT_ROOT / "ml").rglob("*.py")
    ) + [PROJECT_ROOT / "training" / "train_kaggle.py"]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(PROJECT_ROOT).as_posix())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "kaggle_source.zip"))
    args = parser.parse_args()
    path = create_bundle(Path(args.output))
    print(f"Created {path} ({path.stat().st_size} bytes)")
    print("Upload this ZIP as a Kaggle input dataset named e.g. 'movie-recommender-source'.")


if __name__ == "__main__":
    main()
