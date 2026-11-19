#!/usr/bin/env python3
"""Train the recommender locally with the same logic used on Kaggle.

Examples:
  python training/train_local.py
  python training/train_local.py --csv path/to/movies.csv --output-dir model
  python training/train_local.py --weights '{"genres": 4, "overview": 2}'
  python training/train_local.py --report reports/generated/evaluation_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_CSV = PROJECT_ROOT / "Frontend/movie_dataset_builder/output/all_titles.csv"
MOVIES_CSV = PROJECT_ROOT / "Frontend/movie_dataset_builder/output/movies_2023_2026.csv"
DEFAULT_TRAIN_CSV = COMBINED_CSV if COMBINED_CSV.is_file() else MOVIES_CSV
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.artifacts import load_artifacts  # noqa: E402
from ml.config import DEFAULT_FEATURE_WEIGHTS, ModelConfig  # noqa: E402
from ml.evaluate import write_report  # noqa: E402
from ml.trainer import train_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=os.getenv("TRAIN_CSV", str(DEFAULT_TRAIN_CSV)),
        help="TMDB CSV dataset (default: combined movies + series catalogue when present)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("TRAIN_OUTPUT_DIR", str(PROJECT_ROOT / "model")),
        help="Where the six model artifacts are written",
    )
    parser.add_argument(
        "--posters-dir",
        default=os.getenv("TRAIN_POSTERS_DIR", str(
            PROJECT_ROOT / "Frontend/movie_dataset_builder/output/posters"
        )),
        help="Local poster directory (used for missing-poster analysis only)",
    )
    parser.add_argument("--config", help="Optional JSON file with a full model configuration")
    parser.add_argument(
        "--weights",
        help='JSON object overriding feature weights, e.g. {"genres": 4, "title": 3}',
    )
    parser.add_argument(
        "--zip-path",
        nargs="?",
        const="",
        help="Also export a ZIP (default: <output-dir>/../movie_recommender_model.zip)",
    )
    parser.add_argument(
        "--report",
        help="Write a Markdown evaluation/inspection report to this path",
    )
    return parser


def load_config(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig.from_file(args.config) if args.config else ModelConfig()
    if args.weights:
        try:
            overrides = json.loads(args.weights)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--weights must be valid JSON: {exc}") from exc
        if not isinstance(overrides, dict):
            raise SystemExit("--weights must be a JSON object")
        config.feature_weights.update({key: float(value) for key, value in overrides.items()})
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    output_dir = Path(args.output_dir)
    zip_path = None
    if args.zip_path is not None:
        zip_path = Path(args.zip_path) if args.zip_path else output_dir.parent / "movie_recommender_model.zip"

    extra: dict[str, Any] = {}
    posters_dir = Path(args.posters_dir)
    if posters_dir.is_dir():
        extra["local_poster_files_found"] = sum(1 for path in posters_dir.iterdir() if path.is_file())

    result = train_model(args.csv, output_dir, config, zip_path=zip_path, extra_info=extra)
    artifacts = load_artifacts(output_dir)
    print(f"Trained on {result.n_movies} titles ({result.info.get('n_series', 0)} series)")
    print(f"Vocabulary: {result.info['vocabulary_size']} terms")
    print(f"Sparse feature matrix: {result.info['feature_dimensions']} columns, "
          f"{result.info['matrix_nnz']} non-zero values")
    print(f"Training time: {result.info['training_time_seconds']} s")
    print(f"Artifacts: {result.model_dir}")
    if result.zip_path:
        print(f"Export ZIP: {result.zip_path}")
    print(f"Validation: loaded {len(artifacts.metadata)} title records successfully")
    if args.report:
        path = write_report(output_dir, args.report)
        print(f"Inspection report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
