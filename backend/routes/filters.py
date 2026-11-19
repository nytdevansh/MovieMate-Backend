"""GET /api/filters and GET /api/trending"""
from __future__ import annotations

from flask import Blueprint, current_app, request

from backend.services.movie_service import MEDIA_TYPES, serialize_movie
from backend.utils.responses import ok
from backend.utils.validators import parse_choice, parse_int_param, validate_no_unknown

bp = Blueprint("filters", __name__)


@bp.get("/api/filters")
def filters():
    repository = current_app.extensions["movie_repository"]
    options = dict(repository.filter_options)
    options["sort_options"] = [
        "popularity_desc", "rating_desc", "release_date_desc", "title_asc"
    ]
    options["total_movies"] = len(repository.by_id)
    options["total_series"] = len(repository.records) - len(repository.by_id)
    options["total_titles"] = len(repository.records)
    return ok(options)


@bp.get("/api/trending")
def trending():
    validate_no_unknown(request.args, {"limit", "media_type"})
    limit = parse_int_param(request.args.get("limit"), "limit", maximum=100, default=12)
    media_type = parse_choice(
        request.args.get("media_type"), "media_type", MEDIA_TYPES, "movie"
    ) if request.args.get("media_type") not in (None, "") else None
    repository = current_app.extensions["movie_repository"]
    poster_service = current_app.extensions["poster_service"]
    results = [
        serialize_movie(record, poster_service)
        for record in repository.trending(limit, media_type=media_type)
    ]
    return ok(results, meta={
        "count": len(results),
        "sort": "popularity_desc",
        "media_type": media_type or "all",
    })
