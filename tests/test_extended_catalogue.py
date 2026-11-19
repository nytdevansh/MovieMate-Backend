"""Movies + TV series catalogue, generic APIs and runtime UI enhancement."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app import create_app
from backend.config import Config, PROJECT_ROOT
from backend.services.watch_service import WatchService
from ml.artifacts import load_artifacts
from ml.config import ModelConfig
from ml.ids import parse_content_id
from ml.inference import RecommendationEngine
from ml.preprocessing import clean_dataset
from ml.trainer import train_model
from tests.conftest import movie_row


def series_row(series_id: int, **overrides):
    row = movie_row(series_id, **{
        "content_id": f"tv-{series_id}",
        "media_type": "tv",
        "source_tmdb_id": series_id,
        "model_key": f"tv-{series_id}",
        "poster_key": f"tv-{series_id}",
        "title": f"Test Series {series_id}",
        "original_title": f"Test Series {series_id}",
        "genres": "Drama | Mystery",
        "overview": f"A serialized drama about series {series_id} and a long-running mystery.",
        "director": "Series Creator",
        "number_of_seasons": 2,
        "number_of_episodes": 16,
        "status": "Returning Series",
        "tv_type": "Scripted",
    })
    row.update(overrides)
    return row


def build_combined_csv(tmp_path: Path) -> Path:
    rows = [
        movie_row(1, title="Test Movie One", genres="Drama | History"),
        movie_row(2, title="Test Movie Two", genres="Drama | History"),
        series_row(1),
        series_row(2),
    ]
    path = tmp_path / "combined.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_content_ids_keep_movies_and_series_distinct(tmp_path: Path):
    csv_path = build_combined_csv(tmp_path)
    cleaned = clean_dataset(pd.read_csv(csv_path, encoding="utf-8-sig"), ModelConfig())
    content_ids = {parse_content_id(record["content_id"]).public for record in cleaned.records}
    model_keys = {record["model_key"] for record in cleaned.records}
    assert content_ids == {"movie-1", "movie-2", "tv-1", "tv-2"}
    assert model_keys == {"1", "2", "tv-1", "tv-2"}
    assert cleaned.report.duplicate_ids_dropped == 0


def test_engine_recommends_across_combined_artifacts(tmp_path: Path):
    csv_path = build_combined_csv(tmp_path)
    result = train_model(csv_path, tmp_path / "model")
    artifacts = load_artifacts(result.model_dir)
    engine = RecommendationEngine(artifacts)
    assert engine.has_movie("tv-1")
    assert engine.has_movie(1)
    recommendations = engine.recommend("tv-1", limit=2)
    assert recommendations
    assert all(item.content_id != "tv-1" for item in recommendations)
    assert {item.media_type for item in recommendations} <= {"movie", "tv"}


def test_generic_apis_and_runtime_enhancement(
    tmp_path: Path, poster_dirs: list[Path], monkeypatch
):
    csv_path = build_combined_csv(tmp_path)
    model_dir = tmp_path / "model"
    train_model(csv_path, model_dir)
    config = Config({
        "DEBUG": False,
        "TESTING": True,
        "DATASET_PATH": csv_path,
        "MODEL_DIR": model_dir,
        "POSTER_DIRS": [poster_dirs[0]],
        "BACKDROP_DIRS": [poster_dirs[1]],
        "WEB_DIR": PROJECT_ROOT / "web",
        "ALLOW_REMOTE_POSTERS": False,
        "REQUIRE_MODEL_AT_STARTUP": True,
    })

    def fake_availability(self, record, region=None):
        return {
            "content_id": parse_content_id(record["content_id"]).public,
            "region": region or "IN",
            "providers": {"flatrate": [{"name": "Fake TV", "logo_url": None, "link": "https://example.com"}]},
            "provider_page_url": "https://example.com",
            "cinema": {
                "applicable": record.get("media_type") == "movie",
                "status": "check_live_listings",
                "release_dates": [],
                "links": [{"label": "Ticketing", "url": "https://example.com/tickets"}],
            },
            "errors": [],
            "note": "test payload",
        }

    monkeypatch.setattr(WatchService, "availability", fake_availability)
    app = create_app(config)
    client = app.test_client()

    series = client.get("/api/series?per_page=1")
    assert series.status_code == 200
    assert series.get_json()["data"][0]["content_id"].startswith("tv-")

    detail = client.get("/api/titles/tv-1")
    assert detail.status_code == 200
    assert detail.get_json()["data"]["number_of_seasons"] == 2

    watch = client.get("/api/titles/tv-1/watch?region=US")
    assert watch.status_code == 200
    assert watch.get_json()["data"]["cinema"]["applicable"] is False

    similar = client.get("/api/titles/tv-1/recommendations?limit=2")
    assert similar.status_code == 200
    assert similar.get_json()["data"]

    root = client.get("/")
    assert root.status_code == 200
    assert "Enter your info to sign in" in root.get_data(as_text=True)
    assert client.get("/app").status_code == 404
    assert client.get("/app/").status_code == 404

    for page in (
        "/index.html",
        "/adminlogin.html",
        "/admin.html",
        "/streaming_platform_admin_dashboard.html",
        "/dashboard.html",
        "/style.css",
        "/script.js",
        "/Movie-removebg-preview-2-2.png",
    ):
        assert client.get(page).status_code == 200

    dashboard = client.get("/dashboard.html")
    assert dashboard.status_code == 200
    assert "/js/frontend-enhance.js" in dashboard.get_data(as_text=True)
    assert '<base href="/">' in dashboard.get_data(as_text=True)
    assert client.get("/js/frontend-enhance.js").status_code == 200
    title_page = client.get("/title/tv-1")
    assert title_page.status_code == 200
    assert '<base href="/">' in title_page.get_data(as_text=True)
    assert client.get("/.env").status_code == 404

    source = (PROJECT_ROOT / "Frontend/dashboard.html").read_text(encoding="utf-8")
    assert "frontend-enhance.js" not in source


def test_watch_service_parses_provider_and_release_payload(tmp_path: Path):
    config = Config({"TMDB_BEARER_TOKEN": "test-token", "WATCH_REGION": "US"})
    service = WatchService(config)
    service._get = lambda path: (  # type: ignore[method-assign]
        {
            "results": {
                "US": {
                    "link": "https://www.justwatch.com/us/title",
                    "flatrate": [{
                        "provider_name": "Netflix",
                        "logo_path": "/logo.jpg",
                        "link": "https://www.justwatch.com/us/title",
                    }],
                    "rent": [{"provider_name": "Apple TV", "link": "https://example.com/rent"}],
                }
            }
        } if "watch/providers" in path else {
            "results": [{
                "iso_3166_1": "US",
                "release_dates": [{"release_date": "2026-11-20", "type": 3, "certification": "PG-13"}],
            }]
        }
    )
    record = {
        "content_id": "movie-42",
        "media_type": "movie",
        "tmdb_id": 42,
        "source_tmdb_id": 42,
        "title": "Example Movie",
    }
    result = service.availability(record, "US")
    assert result["providers"]["flatrate"][0]["name"] == "Netflix"
    assert result["providers"]["flatrate"][0]["logo_url"].endswith("/logo.jpg")
    assert result["providers"]["rent"][0]["name"] == "Apple TV"
    assert any(link["label"].startswith("Fandango") for link in result["cinema"]["links"])
    assert result["cinema"]["release_dates"][0]["date"] == "2026-11-20"
