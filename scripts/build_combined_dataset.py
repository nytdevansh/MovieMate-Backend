#!/usr/bin/env python3
"""Rebuild output/all_titles.csv from the existing movie + TV CSVs.

Useful after editing the TV CSV manually or after an interrupted fetch: it does
not call TMDB and never rewrites the original 5,000-movie dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_DIR = PROJECT_ROOT / "Frontend" / "movie_dataset_builder"
if str(BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_DIR))

import build_webseries_dataset as builder  # noqa: E402


def main() -> int:
    if not builder.SERIES_CSV.is_file():
        raise SystemExit(
            f"TV dataset not found: {builder.SERIES_CSV}. "
            "Run Frontend/movie_dataset_builder/build_webseries_dataset.py first."
        )
    series = builder.load_existing()
    movies_kept, series_count = builder.build_combined(series)
    print(f"Wrote {builder.COMBINED_CSV}")
    print(f"Preserved {movies_kept} original movies and included {series_count} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
