"""Flask application factory."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, request, send_from_directory
from werkzeug.exceptions import NotFound

from backend.config import Config
from backend.models.model_loader import load_model_state
from backend.routes import (
    feedback,
    filters,
    health,
    movies,
    posters,
    recommendations,
    search,
    titles,
)
from backend.services.feedback_service import FeedbackService
from backend.services.movie_service import MovieRepository
from backend.services.poster_service import PosterService
from backend.services.recommendation_service import RecommendationService
from backend.services.watch_service import WatchService
from backend.utils.errors import register_error_handlers
from ml.ids import parse_content_id

logger = logging.getLogger(__name__)

# Existing Frontend/ files are treated as immutable user work. Only these HTML
# entry points are served, and the enhancement is injected into the response.
FRONTEND_HTML_FILES = frozenset({
    "index.html",
    "dashboard.html",
    "adminlogin.html",
    "admin.html",
    "streaming_platform_admin_dashboard.html",
})
FRONTEND_ASSET_SUFFIXES = frozenset({
    ".css", ".js", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".ico",
})
# The admin page references an image that was never committed. Keep the HTML
# untouched and serve the existing logo under that name instead of a 404.
FRONTEND_ASSET_ALIASES = {
    "Movie-removebg-preview-2-2.png": "logo2.png",
}


def create_app(config: Config | None = None) -> Flask:
    """Build the app, load dataset and model once, then register routes."""
    config = config or Config.from_env()
    app = Flask(__name__, static_folder=None)
    app.config.update(
        APP_CONFIG=config,
        DEBUG=config.DEBUG,
        TESTING=config.TESTING,
        SECRET_KEY=config.SECRET_KEY,
        MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
        # Frontend assets must revalidate so deployments are not stale for an
        # hour; only /media/* uses the long MEDIA_CACHE_MAX_AGE explicitly.
        SEND_FILE_MAX_AGE_DEFAULT=0,
        JSON_SORT_KEYS=config.JSON_SORT_KEYS,
    )
    register_error_handlers(app)

    # Services are process-wide singletons: no per-request model/dataset loads.
    poster_service = PosterService(config)
    movie_repository = MovieRepository(config.DATASET_PATH, poster_service)
    model_state = load_model_state(config)
    feedback_service = FeedbackService(config)
    recommendation_service = RecommendationService(
        model_state, movie_repository, poster_service, feedback_service
    )
    watch_service = WatchService(config)
    app.extensions["poster_service"] = poster_service
    app.extensions["movie_repository"] = movie_repository
    app.extensions["model_state"] = model_state
    app.extensions["feedback_service"] = feedback_service
    app.extensions["recommendation_service"] = recommendation_service
    app.extensions["watch_service"] = watch_service

    for blueprint in (
        health.bp,
        movies.bp,
        search.bp,
        recommendations.bp,
        feedback.bp,
        filters.bp,
        posters.bp,
        titles.bp,
    ):
        app.register_blueprint(blueprint)

    @app.get("/")
    def frontend_index():
        return serve_frontend_html("index.html")

    @app.get("/js/<path:filename>")
    def frontend_js(filename: str):
        return send_from_directory(config.WEB_DIR / "js", filename, max_age=0)

    # ------------------------------------------------------------ Frontend UI
    def serve_frontend_html(filename: str, initial_title: str | None = None) -> Response:
        if filename not in FRONTEND_HTML_FILES:
            abort(404)
        path = config.FRONTEND_DIR / filename
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NotFound(f"Frontend page not found: {filename}") from exc

        # Relative image/src URLs in the original HTML assume the site root.
        # A <base> tag keeps them working when the page is served from
        # /title/<id> or /frontend/... without editing the protected files.
        source = re.sub(
            r"(<head\b[^>]*>)", r'\1\n<base href="/">', source, count=1, flags=re.IGNORECASE
        )

        scripts = ['<script src="/js/ui-polish.js"></script>']
        if filename == "dashboard.html":
            config_json = {
                "apiBaseUrl": "/api",
                "enhancedPage": True,
                "initialTitle": initial_title,
                "watchRegion": config.WATCH_REGION,
            }
            import json

            scripts.insert(
                0,
                "<script>window.MOVIEMATE_FRONTEND_CONFIG="
                + json.dumps(config_json, ensure_ascii=False).replace("<", "\\u003c")
                + ";</script>",
            )
            scripts.append('<script src="/js/frontend-enhance.js"></script>')
        source = source.replace(
            "</body>", "\n".join(scripts) + "\n</body>", 1
        )
        return Response(source, mimetype="text/html", headers={
            "Cache-Control": "no-store, max-age=0",
        })

    @app.get("/frontend/")
    def frontend_root():
        return serve_frontend_html("index.html")

    @app.get("/frontend/<path:filename>")
    @app.get("/title/<content_id>")
    @app.get("/frontend/title/<content_id>")
    def frontend_page(filename: str | None = None, content_id: str | None = None):
        if content_id is not None:
            if parse_content_id(content_id) is None:
                abort(404)
            return serve_frontend_html("dashboard.html", initial_title=content_id)
        if filename in FRONTEND_HTML_FILES:
            return serve_frontend_html(filename)
        return serve_frontend_asset(filename or "")

    def serve_frontend_asset(filename: str):
        """Serve only safe Frontend assets; never expose .env, CSV or .git data."""
        actual = FRONTEND_ASSET_ALIASES.get(filename, filename)
        relative = Path(actual)
        if (
            not actual
            or relative.is_absolute()
            or ".." in relative.parts
            or any(part.startswith(".") for part in relative.parts)
            or relative.suffix.lower() not in FRONTEND_ASSET_SUFFIXES
            or relative.suffix.lower() == ".html"
        ):
            abort(404)
        if not (config.FRONTEND_DIR / relative).is_file():
            abort(404)
        return send_from_directory(config.FRONTEND_DIR, actual, max_age=0)

    @app.get("/<path:filename>")
    def frontend_root_asset(filename: str):
        if filename in FRONTEND_HTML_FILES:
            return serve_frontend_html(filename)
        return serve_frontend_asset(filename)

    @app.before_request
    def cors_preflight():
        # The app is same-origin by default. These headers also allow a future
        # static-hosted frontend to call the API/media routes cross-origin.
        if (
            request.method == "OPTIONS"
            and (request.path.startswith("/api/") or request.path.startswith("/media/"))
        ):
            return Response(status=204)
        return None

    @app.after_request
    def security_headers(response: Any):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        if request.path.startswith("/api/") or request.path.startswith("/media/"):
            allowed_origin = os.getenv("CORS_ALLOWED_ORIGIN", "*").strip() or "*"
            response.headers.setdefault("Access-Control-Allow-Origin", allowed_origin)
            response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Accept")
            response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    logger.info("Application ready (environment=%s)", config.ENV_NAME)
    return app
