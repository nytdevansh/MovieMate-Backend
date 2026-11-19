"""Poster metadata API and efficient static image delivery."""
from __future__ import annotations

from flask import Blueprint, current_app, send_file

from backend.utils.errors import ApiError
from backend.utils.responses import ok

bp = Blueprint("posters", __name__)


@bp.get("/api/posters/<int:movie_id>")
def poster_info(movie_id: int):
    repository = current_app.extensions["movie_repository"]
    record = repository.get(movie_id)
    if record is None:
        raise ApiError(
            "MOVIE_NOT_FOUND", f"Movie {movie_id} was not found", status_code=404
        )
    poster_service = current_app.extensions["poster_service"]
    data = {"movie_id": movie_id, "title": record.get("title") or ""}
    data.update(poster_service.resolve(record).to_dict())
    return ok(data)


def _serve_media(kind: str, filename: str):
    service = current_app.extensions["poster_service"]
    path = service.media_path(kind, filename)
    if path is None:
        raise ApiError("IMAGE_NOT_FOUND", "Image not found", status_code=404)
    return send_file(
        path,
        mimetype=service.media_mimetype(path),
        conditional=True,
        max_age=current_app.config["APP_CONFIG"].MEDIA_CACHE_MAX_AGE,
    )


@bp.get("/media/posters/<path:filename>")
def serve_poster(filename: str):
    return _serve_media("posters", filename)


@bp.get("/media/backdrops/<path:filename>")
def serve_backdrop(filename: str):
    return _serve_media("backdrops", filename)


@bp.get("/media/fallback/<path:filename>")
def serve_fallback(filename: str):
    if filename != "poster.svg":
        raise ApiError("IMAGE_NOT_FOUND", "Image not found", status_code=404)
    service = current_app.extensions["poster_service"]
    path = service.fallback_path()
    if path is None:
        raise ApiError("IMAGE_NOT_FOUND", "Fallback image is missing", status_code=404)
    return send_file(
        path,
        mimetype="image/svg+xml",
        conditional=True,
        max_age=current_app.config["APP_CONFIG"].MEDIA_CACHE_MAX_AGE,
    )
