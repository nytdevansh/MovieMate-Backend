"""Shared test fixtures: tiny synthetic dataset, trained model and Flask client."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backend.app import create_app
from backend.config import Config, PROJECT_ROOT
from ml.config import ModelConfig
from ml.preprocessing import EXPECTED_COLUMNS
from ml.trainer import train_model

# Minimal valid 1x1 JPEG (>64 bytes) used to test local poster resolution.
TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwcJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPDIzM//AABEIAAEAAQMBIgACEQEDEQH/"
    "xAAfAAABBQEBAQEBAQAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQEC"
    "AwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6S"
    "lpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v"
    "/aAAwDAQACEQMRAD8A/v4ooooA/9k="
)


def movie_row(movie_id: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in EXPECTED_COLUMNS}
    row.update({
        "tmdb_id": movie_id,
        "imdb_id": f"tt{movie_id:07d}",
        "title": f"Test Movie {movie_id}",
        "original_title": f"Test Movie {movie_id}",
        "release_date": "2024-05-01",
        "year": 2024,
        "original_language": "en" if movie_id % 2 else "hi",
        "genres": "Drama | Thriller" if movie_id % 2 else "Comedy | Romance",
        "overview": f"An original story about movie {movie_id} and a brave journey.",
        "cast": "Actor One | Actor Two",
        "director": "Director A" if movie_id % 2 else "Director B",
        "runtime_minutes": 120,
        "vote_average": 7.0 + (movie_id % 3),
        "vote_count": 100 + movie_id,
        "popularity": 50.0 - movie_id,
        "poster_url": "https://image.tmdb.org/t/p/w500/test.jpg",
        "adult": False,
    })
    row.update(overrides)
    return row


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    rows = [movie_row(index) for index in range(1, 16)]
    # A duplicate ID with less metadata must lose to the complete record.
    rows.append(movie_row(1, title="Incomplete Duplicate", overview="", cast=""))
    # Deliberately messy metadata that must be handled, never invented.
    rows[2]["vote_average"] = 99.0
    rows[3]["release_date"] = "not-a-date"
    rows[3]["year"] = "not-a-year"
    rows[4]["poster_url"] = "javascript:alert(1)"
    rows[5]["genres"] = "Drama | Drama |  Comedy "
    path = tmp_path / "movies_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_model(tmp_path: Path, synthetic_csv: Path) -> Path:
    model_dir = tmp_path / "model"
    train_model(synthetic_csv, model_dir, ModelConfig())
    return model_dir


@pytest.fixture
def poster_dirs(tmp_path: Path) -> list[Path]:
    posters = tmp_path / "posters"
    backdrops = tmp_path / "backdrops"
    posters.mkdir()
    backdrops.mkdir()
    (posters / "1.jpg").write_bytes(TINY_JPEG)
    (posters / "2.jpg").write_text("this is not an image", encoding="utf-8")
    return [posters, backdrops]


@pytest.fixture
def posters(poster_dirs: list[Path]) -> list[Path]:
    return poster_dirs


@pytest.fixture
def app_config(
    tmp_path: Path,
    synthetic_csv: Path,
    tiny_model: Path,
    poster_dirs: list[Path],
) -> Config:
    return Config({
        "DEBUG": False,
        "TESTING": True,
        "DATASET_PATH": synthetic_csv,
        "MODEL_DIR": tiny_model,
        "POSTER_DIRS": [poster_dirs[0]],
        "BACKDROP_DIRS": [poster_dirs[1]],
        "WEB_DIR": PROJECT_ROOT / "web",
        "ALLOW_REMOTE_POSTERS": True,
        "AUTO_DOWNLOAD_POSTERS": False,
        "REQUIRE_MODEL_AT_STARTUP": True,
    })


@pytest.fixture
def app(app_config: Config):
    return create_app(app_config)


@pytest.fixture
def client(app):
    return app.test_client()
