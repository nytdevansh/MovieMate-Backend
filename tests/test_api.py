"""REST API contract tests."""
from __future__ import annotations

from pathlib import Path

from backend.app import create_app
from backend.config import Config, PROJECT_ROOT


def payload(response):
    body = response.get_json()
    assert set(body) == {"success", "data", "meta", "error"}
    return body


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = payload(response)
    assert body["success"] is True
    assert body["data"]["status"] == "ok"
    assert body["data"]["model_loaded"] is True
    assert body["data"]["movies_loaded"] == 15
    assert "MODEL_DIR" not in str(body)


def test_movies_pagination_and_filters(client):
    response = client.get("/api/movies?page=1&per_page=5&sort=rating_desc")
    assert response.status_code == 200
    body = payload(response)
    assert len(body["data"]) == 5
    assert body["meta"]["total"] == 15
    assert body["meta"]["total_pages"] == 3
    assert body["meta"]["has_next"] is True
    ratings = [movie["rating"] for movie in body["data"] if movie["rating"] is not None]
    assert ratings == sorted(ratings, reverse=True)

    genre = client.get("/api/movies?genre=Drama&year=2024&language=en&min_rating=7")
    assert genre.status_code == 200
    assert genre.get_json()["data"]
    assert all("Drama" in movie["genres"] for movie in genre.get_json()["data"])
    assert all(movie["release_year"] == 2024 for movie in genre.get_json()["data"])


def test_movie_detail_and_404(client):
    detail = payload(client.get("/api/movies/1"))["data"]
    for key in (
        "movie_id", "title", "overview", "genres", "cast", "director",
        "india_streaming", "poster_url", "rating", "release_year",
    ):
        assert key in detail
    assert detail["movie_id"] == 1
    assert detail["poster_url"] == "/media/posters/1.jpg"

    missing = client.get("/api/movies/999999")
    assert missing.status_code == 404
    assert payload(missing)["error"]["code"] == "MOVIE_NOT_FOUND"


def test_search(client):
    response = client.get("/api/search?q=Test%20Movie%201&limit=5")
    assert response.status_code == 200
    body = payload(response)
    assert body["meta"]["query"] == "Test Movie 1"
    assert body["data"]
    assert all("poster_url" in movie for movie in body["data"])


def test_recommendations(client):
    response = client.get("/api/recommendations/1?limit=4")
    assert response.status_code == 200
    body = payload(response)
    assert 0 < len(body["data"]) <= 4
    assert body["meta"]["method"] == "tfidf_cosine_content_based"
    for item in body["data"]:
        for key in (
            "movie_id", "title", "poster_url", "genres", "release_year",
            "rating", "similarity_score", "short_description",
        ):
            assert key in item
        assert item["movie_id"] != 1

    assert client.get("/api/recommendations/999999").status_code == 404
    assert client.get("/api/recommendations/1?limit=abc").status_code == 400
    assert client.get("/api/recommendations/1?bogus=1").status_code == 400


def test_personalized_ranking_and_feedback(client):
    first_response = client.get("/api/titles/movie-1/recommendations?limit=5")
    first = payload(first_response)
    ranking = first["meta"]["personalization"]
    assert ranking["personalized"] is True
    assert "repetition_penalty" in ranking["ranking"]
    assert "mm_session=" in first_response.headers.get("Set-Cookie", "")

    second = payload(client.get("/api/titles/movie-1/recommendations?limit=5"))
    assert second["meta"]["personalization"]["penalized_titles"] > 0

    saved = payload(client.post("/api/feedback", json={
        "content_id": "movie-2", "event": "save"
    }))
    assert saved["data"]["reward"] == 1.0
    assert saved["data"]["personalization"] == "session_only"

    assert client.post("/api/feedback", json={
        "content_id": "movie-2", "event": "clap"
    }).status_code == 400
    assert client.post("/api/feedback", json={
        "content_id": "nope", "event": "save"
    }).status_code == 400
    assert client.post("/api/feedback", json={
        "content_id": "movie-2", "event": "save", "extra": 1
    }).status_code == 400
    assert client.post("/api/feedback", json={
        "content_id": "movie-999999", "event": "save"
    }).status_code == 404


def test_ui_polish_is_injected_on_every_frontend_page(client):
    for path in (
        "/", "/index.html", "/adminlogin.html", "/admin.html",
        "/streaming_platform_admin_dashboard.html", "/dashboard.html",
        "/title/movie-1",
    ):
        html = client.get(path).get_data(as_text=True)
        assert '/js/ui-polish.js' in html
    assert client.get("/js/ui-polish.js").status_code == 200


def test_filters_and_trending(client):
    filters = payload(client.get("/api/filters"))["data"]
    assert "Drama" in filters["genres"]
    assert 2024 in filters["years"]
    assert "en" in filters["languages"]

    trending = payload(client.get("/api/trending?limit=6"))["data"]
    assert len(trending) == 6
    assert all("poster_url" in movie for movie in trending)


def test_posters_and_media(client):
    info = payload(client.get("/api/posters/1"))["data"]
    assert info["poster_source"] == "local"
    assert info["poster_url"] == "/media/posters/1.jpg"

    image = client.get("/media/posters/1.jpg")
    assert image.status_code == 200
    assert image.content_type.startswith("image/")
    assert client.get("/media/posters/999999.jpg").status_code == 404
    assert client.get("/media/fallback/poster.svg").status_code == 200
    assert client.get("/media/posters/..%2Fetc%2Fpasswd").status_code == 404


def test_input_validation(client):
    assert client.get("/api/search?q=").status_code == 400
    assert client.get("/api/movies?page=0").status_code == 400
    assert client.get("/api/movies?per_page=99999").status_code == 400
    assert client.get("/api/movies?min_rating=11").status_code == 400
    assert client.get("/api/movies?sort=bogus").status_code == 400
    assert client.get("/api/nope").status_code == 404


def test_static_assets_revalidate_but_media_is_cacheable(client):
    for path in ("/", "/style.css", "/js/frontend-enhance.js", "/logo.png"):
        response = client.get(path)
        assert response.status_code == 200
        assert "max-age=0" in response.headers.get("Cache-Control", "")
    image = client.get("/media/posters/1.jpg")
    assert image.status_code == 200
    assert "max-age=3600" in image.headers.get("Cache-Control", "")


def test_frontend_uses_paths_relative_to_api_base():
    """API_BASE already ends in /api; callers must not repeat that prefix."""
    source = (PROJECT_ROOT / "web/js/frontend-enhance.js").read_text(encoding="utf-8")
    assert 'config.apiBaseUrl || "/api"' in source
    assert '"/api/' not in source
    assert "&limit=60" not in source  # /api/search rejects limits above 50
    # Sidebar <div> elements must become keyboard-operable, not just labelled.
    assert 'if (item.tabIndex < 0) item.tabIndex = 0;' in source
    assert 'aria-current' in source
    assert 'state.lastFocus.focus()' in source
    for path in (
        "/movies", "/series", "/search", "/titles/", "/watch?region=", "/feedback"
    ):
        assert path in source


def test_missing_model_degrades_without_crashing(
    tmp_path: Path, synthetic_csv: Path, poster_dirs: list[Path]
):
    empty_model = tmp_path / "empty_model"
    empty_model.mkdir()
    config = Config({
        "DEBUG": False,
        "TESTING": True,
        "DATASET_PATH": synthetic_csv,
        "MODEL_DIR": empty_model,
        "POSTER_DIRS": [poster_dirs[0]],
        "BACKDROP_DIRS": [poster_dirs[1]],
        "WEB_DIR": PROJECT_ROOT / "web",
        "ALLOW_REMOTE_POSTERS": False,
        "REQUIRE_MODEL_AT_STARTUP": False,
    })
    app = create_app(config)
    test_client = app.test_client()
    health = test_client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["data"]["model_loaded"] is False
    response = test_client.get("/api/recommendations/1")
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert test_client.get("/api/movies?per_page=2").status_code == 200
