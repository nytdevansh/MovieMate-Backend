"""Poster resolution, validation and safe static path handling."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services.poster_service import (
    PosterService, is_valid_image_bytes, is_valid_image_file,
)
from tests.conftest import TINY_JPEG


def make_service(poster_dirs, *, allow_remote=True, auto_download=False):
    config = SimpleNamespace(
        POSTER_DIRS=poster_dirs,
        BACKDROP_DIRS=[],
        FALLBACK_POSTER_URL="/media/fallback/poster.svg",
        REMOTE_IMAGE_HOSTS=["image.tmdb.org"],
        ALLOW_REMOTE_POSTERS=allow_remote,
        AUTO_DOWNLOAD_POSTERS=auto_download,
        DOWNLOAD_TIMEOUT_SECONDS=1.0,
        POSTER_CACHE_TTL_SECONDS=0,
    )
    return PosterService(config)


def test_local_poster_preferred_and_no_filesystem_paths(posters: list[Path]):
    service = make_service([posters[0]])
    record = {
        "tmdb_id": 1,
        "title": "Test",
        "poster_url": "https://image.tmdb.org/t/p/w500/x.jpg",
    }
    resolution = service.resolve(record)
    assert resolution.poster_source == "local"
    assert resolution.poster_url == "/media/posters/1.jpg"
    assert str(posters[0]) not in resolution.poster_url
    assert resolution.to_dict()["poster_source"] == "local"


def test_broken_local_file_falls_back_to_remote_or_placeholder(posters: list[Path]):
    service = make_service([posters[0]])
    broken = {"tmdb_id": 2, "title": "Broken", "poster_url": "https://image.tmdb.org/t/p/w500/x.jpg"}
    assert service.resolve(broken).poster_source == "tmdb"

    no_poster = {"tmdb_id": 999, "title": "Missing", "poster_url": "javascript:alert(1)"}
    resolution = service.resolve(no_poster)
    assert resolution.poster_source == "fallback"
    assert resolution.poster_url == "/media/fallback/poster.svg"


def test_remote_posters_can_be_disabled(posters: list[Path]):
    service = make_service([posters[0]], allow_remote=False)
    record = {"tmdb_id": 3, "title": "Remote", "poster_url": "https://image.tmdb.org/t/p/w500/x.jpg"}
    assert service.resolve(record).poster_source == "fallback"


def test_media_path_rejects_traversal_and_unknown_files(posters: list[Path]):
    service = make_service([posters[0]])
    assert service.media_path("posters", "1.jpg") == posters[0] / "1.jpg"
    assert service.media_path("posters", "../etc/passwd") is None
    assert service.media_path("posters", "1.exe") is None
    assert service.media_path("posters", "abc.jpg") is None


def test_existing_local_poster_prevents_download(posters: list[Path], monkeypatch):
    service = make_service([posters[0]], auto_download=True)

    def fail(*args, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert service.ensure_local_poster(1, "https://image.tmdb.org/t/p/w500/x.jpg") == posters[0] / "1.jpg"


def test_image_validation_rejects_html():
    assert is_valid_image_bytes(TINY_JPEG)
    assert is_valid_image_file(Path(__file__)) is False
    assert is_valid_image_bytes(b"<html>not an image</html>") is False
