"""Tests for the first-party remote image proxy."""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from backend.config import Config
from backend.services.poster_service import PosterService
from tests.conftest import TINY_JPEG


def proxy_config(**overrides: object) -> Config:
    values = {
        "ALLOW_REMOTE_POSTERS": True,
        "PROXY_REMOTE_IMAGES": True,
        "REMOTE_IMAGE_CACHE_MAX_AGE": 60,
        "POSTER_DIRS": [],
        "BACKDROP_DIRS": [],
        "REMOTE_IMAGE_HOSTS": ["image.tmdb.org"],
    }
    values.update(overrides)
    return Config(values)  # type: ignore[arg-type]


def test_public_remote_url_uses_first_party_proxy() -> None:
    service = PosterService(proxy_config())
    public = service.public_remote_url("https://image.tmdb.org/t/p/w500/test.jpg")
    assert public is not None
    parsed = urlparse(public)
    assert parsed.path == "/media/remote"
    assert parse_qs(parsed.query)["url"] == [
        "https://image.tmdb.org/t/p/w500/test.jpg"
    ]


def test_public_remote_url_ignores_disallowed_hosts() -> None:
    service = PosterService(proxy_config())
    assert service.public_remote_url("https://example.com/evil.jpg") is None


def test_fetch_remote_image_rejects_disallowed_host() -> None:
    service = PosterService(proxy_config())
    with pytest.raises(ValueError, match="not allowed"):
        service.fetch_remote_image("https://example.com/evil.jpg")


def test_proxy_route_serves_validated_image(client, monkeypatch) -> None:
    service = client.application.extensions["poster_service"]
    monkeypatch.setattr(
        service,
        "fetch_remote_image",
        lambda url: (TINY_JPEG, "image/jpeg"),
    )
    response = client.get(
        "/media/remote?url=https%3A%2F%2Fimage.tmdb.org%2Ft%2Fp%2Fw500%2Ftest.jpg"
    )
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == TINY_JPEG
    assert response.headers["Cache-Control"].startswith("public, max-age=")


def test_proxy_route_rejects_bad_url(client) -> None:
    response = client.get("/media/remote?url=https%3A%2F%2Fexample.com%2Fevil.jpg")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_IMAGE_URL"
