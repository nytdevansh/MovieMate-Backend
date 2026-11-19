#!/usr/bin/env python3
"""Install a Kaggle-exported model ZIP into the Flask project's model/ directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.artifacts import ArtifactError, extract_model_zip, load_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", help="Path to movie_recommender_model.zip")
    parser.add_argument(
        "--target", default=str(PROJECT_ROOT / "model"), help="Installation directory"
    )
    args = parser.parse_args()

    target = Path(args.target)
    try:
        extract_model_zip(args.zip_path, target)
        artifacts = load_artifacts(target)
    except ArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    info = artifacts.info
    print(f"Installed and validated model into {target}")
    print(f"Movies: {info.get('n_movies')}")
    print(f"Vocabulary: {info.get('vocabulary_size')}")
    print(f"Feature dimensions: {info.get('feature_dimensions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
