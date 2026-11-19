"""Movie catalogue endpoints."""
from __future__ import annotations

from flask import Blueprint, current_app, request

from backend.services.movie_service import SORT_OPTIONS, serialize_movie
from backend.utils.errors import ApiError
from backend.utils.responses import ok, pagination_meta
from backend.utils.validators import (
    optional_text,
    parse_choice,
    parse_float_param,
    parse_int_param,
    validate_no_unknown,
)

bp = Blueprint("movies", __name__)


@bp.get("/api/movies")
def list_movies():
    config = current_app.config["APP_CONFIG"]
    validate_no_unknown(
        request.args,
        {"page", "per_page", "genre", "year", "language", "min_rating", "sort"},
    )
    page = parse_int_param(request.args.get("page"), "page", default=1)
    per_page = parse_int_param(
        request.args.get("per_page"), "per_page", maximum=config.MAX_PER_PAGE,
        default=config.DEFAULT_PER_PAGE,
    )
    year_value = request.args.get("year")
    year = parse_int_param(
        year_value, "year", minimum=1900, maximum=2100
    ) if year_value not in (None, "") else None
    min_rating = parse_float_param(
        request.args.get("min_rating"), "min_rating", maximum=10.0
    ) if request.args.get("min_rating") not in (None, "") else None

    repository = current_app.extensions["movie_repository"]
    poster_service = current_app.extensions["poster_service"]
    records, total = repository.list_movies(
        genre=optional_text(request.args.get("genre"), "genre", 100),
        year=year,
        language=optional_text(request.args.get("language"), "language", 20),
        min_rating=min_rating,
        sort=parse_choice(request.args.get("sort"), "sort", SORT_OPTIONS, "popularity_desc"),
        page=page,
        per_page=per_page,
    )
    return ok(
        [serialize_movie(record, poster_service) for record in records],
        meta=pagination_meta(page, per_page, total),
    )


@bp.get("/api/movies/<int:movie_id>")
def movie_detail(movie_id: int):
    repository = current_app.extensions["movie_repository"]
    record = repository.get(movie_id)
    if record is None:
        raise ApiError(
            "MOVIE_NOT_FOUND", f"Movie {movie_id} was not found", status_code=404
        )
    poster_service = current_app.extensions["poster_service"]
    return ok(serialize_movie(record, poster_service, detail=True))
