#!/usr/bin/env python3
"""Kaggle-compatible training entry point.

Designed for:
  input  : /kaggle/input/<dataset-slug>/all_titles.csv  (+ optional posters/)
           (or the original movies_2023_2026.csv for a movies-only model)
  output : /kaggle/working/movie_model/
  export : /kaggle/working/movie_recommender_model.zip

No TMDB API token, network access or local Mac is required. The `ml` package
must be importable: the companion notebook unzips kaggle_source.zip into
/kaggle/working/src and adds that directory to sys.path.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Allow running as `python training/train_kaggle.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, Path("/kaggle/working/src"), Path("/kaggle/working")):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ml.artifacts import load_artifacts  # noqa: E402
from ml.config import ModelConfig  # noqa: E402
from ml.evaluate import write_report  # noqa: E402
from ml.preprocessing import EXPECTED_COLUMNS  # noqa: E402
from ml.trainer import train_model  # noqa: E402

DEFAULT_INPUT = Path(os.getenv("KAGGLE_INPUT_DIR", "/kaggle/input"))
DEFAULT_OUTPUT = Path(os.getenv("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-name",
        help="Exact CSV filename inside the input dataset; auto-detected when omitted",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON model configuration")
    parser.add_argument(
        "--no-zip", action="store_true", help="Skip the downloadable model ZIP"
    )
    parser.add_argument(
        "--report", action="store_true", help="Write an inspection report to /kaggle/working"
    )
    return parser


def locate_csv(input_dir: Path, csv_name: str | None = None) -> Path:
    """Find the uploaded CSV by schema, not by a hard-coded filename."""
    import pandas as pd

    if not input_dir.is_dir():
        raise SystemExit(
            f"Kaggle input directory not found: {input_dir}. "
            "Upload the movie CSV as a Kaggle input dataset first."
        )
    if csv_name:
        candidate = input_dir / csv_name
        matches = [candidate] if candidate.is_file() else []
        if not matches:
            matches = list(input_dir.rglob(csv_name))
    else:
        matches = sorted(input_dir.rglob("*.csv"))

    valid: list[Path] = []
    for path in matches:
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        except Exception:
            continue
        if set(EXPECTED_COLUMNS).issubset(set(header.columns)):
            valid.append(path)

    if not valid:
        raise SystemExit(
            "No uploaded CSV with the expected TMDB schema was found under "
            f"{input_dir}. Expected columns include: {', '.join(EXPECTED_COLUMNS[:6])}..."
        )
    def preference(path: Path) -> tuple[int, str]:
        name = path.name.casefold()
        if "all_titles" in name or "combined" in name:
            return 0, str(path)
        if "movie" in name:
            return 1, str(path)
        return 2, str(path)

    valid.sort(key=preference)
    return valid[0]


def locate_posters_dir(input_dir: Path) -> Path | None:
    matches = sorted(
        (path for path in input_dir.rglob("posters") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    return matches[0] if matches else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = locate_csv(args.input_dir, args.csv_name)
    model_dir = args.output_dir / "movie_model"
    config = ModelConfig.from_file(args.config) if args.config else ModelConfig()
    config.validate()

    extra: dict[str, Any] = {
        "training_environment": "kaggle" if Path("/kaggle").exists() else "local_simulation",
        "input_dataset_filename": csv_path.name,
    }
    posters_dir = locate_posters_dir(args.input_dir)
    if posters_dir:
        extra["local_poster_files_found"] = sum(1 for path in posters_dir.iterdir() if path.is_file())

    zip_path = None if args.no_zip else args.output_dir / "movie_recommender_model.zip"
    result = train_model(csv_path, model_dir, config, zip_path=zip_path, extra_info=extra)
    artifacts = load_artifacts(model_dir)

    print(f"Dataset: {csv_path.name}")
    print(f"Trained on {result.n_movies} titles ({result.info.get('n_series', 0)} series)")
    print(f"Vocabulary: {result.info['vocabulary_size']} terms")
    print(f"Sparse matrix: {result.info['matrix_nnz']} non-zero values")
    print(f"Artifacts: {model_dir}")
    if result.zip_path:
        print(f"Downloadable ZIP: {result.zip_path}")

    if args.report:
        report = write_report(model_dir, args.output_dir / "evaluation_report.md")
        print(f"Inspection report: {report}")
    print(f"Validation: loaded {len(artifacts.metadata)} title records from exported artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
