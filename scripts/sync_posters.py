#!/usr/bin/env python3
"""Copy/validate dataset-builder posters into assets/posters by TMDB ID.

The Flask app already reads the dataset builder's poster directory directly,
so this script is optional consolidation. It never overwrites a valid local
image and never copies a broken image file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config  # noqa: E402
from backend.services.poster_service import (  # noqa: E402
    PosterService, is_valid_image_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "Frontend/movie_dataset_builder/output/posters"),
    )
    parser.add_argument("--target", default=str(PROJECT_ROOT / "assets/posters"))
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Opt-in: fetch posters from TMDB URLs for IDs without any local file",
    )
    parser.add_argument("--csv", help="Dataset CSV needed with --download-missing")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N actions (0 = all)")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    if not source.is_dir():
        print(f"ERROR: source poster directory not found: {source}", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)

    copied = skipped = invalid = 0
    for path in sorted(source.iterdir()):
        if args.limit and copied >= args.limit:
            break
        if not path.is_file() or path.name.startswith("."):
            continue
        destination = target / path.name
        if is_valid_image_file(destination):
            skipped += 1
            continue
        if not is_valid_image_file(path):
            invalid += 1
            continue
        destination.write_bytes(path.read_bytes())
        copied += 1

    print(f"Copied {copied} valid posters, skipped {skipped} existing, rejected {invalid} broken files")
    print(f"Target: {target}")

    if args.download_missing:
        if not args.csv:
            print("ERROR: --download-missing requires --csv", file=sys.stderr)
            return 1
        import pandas as pd

        from ml.config import ModelConfig
        from ml.preprocessing import clean_dataset

        config = Config.from_env()
        config.POSTER_DIRS = [target]
        config.AUTO_DOWNLOAD_POSTERS = False
        service = PosterService(config)
        raw = pd.read_csv(args.csv, encoding="utf-8-sig")
        records = clean_dataset(raw, ModelConfig()).records
        downloaded = 0
        for record in records:
            if args.limit and downloaded >= args.limit:
                break
            movie_id = int(record["tmdb_id"])
            if service.find_local("posters", movie_id) is not None:
                continue
            if service.ensure_local_poster(movie_id, record.get("poster_url"), force=True):
                downloaded += 1
        print(f"Downloaded {downloaded} missing posters into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
