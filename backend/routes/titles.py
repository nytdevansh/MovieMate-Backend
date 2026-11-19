"""Generic movies + TV series endpoints used by the enhanced Frontend UI."""
from __future__ import annotations

import re

from flask import Blueprint, current_app, request

from backend.routes.session import attach_session_cookie, session_for_request
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
from ml.ids import parse_content_id

bp = Blueprint("titles", __name__)

_REGION_RE = re.compile(r"^[A-Z]{2}$")


def _content_or_404(value: str):
    parsed = parse_content_id(value)
    if parsed is None:
        raise ApiError(
            "INVALID_CONTENT_ID",
            f"Invalid content ID {value!r}. Use movie-<id> or tv-<id>.",
            status_code=400,
        )
    repository = current_app.extensions["movie_repository"]
    record = repository.get_content(parsed.public)
    if record is None:
        raise ApiError(
            "TITLE_NOT_FOUND", f"Title {parsed.public} was not found", status_code=404
        )
    return parsed, record


@bp.get("/api/series")
def list_series():
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
        media_type="tv",
    )
    return ok(
        [serialize_movie(record, poster_service) for record in records],
        meta=pagination_meta(page, per_page, total),
    )


@bp.get("/api/titles/<content_id>")
def title_detail(content_id: str):
    _parsed, record = _content_or_404(content_id)
    poster_service = current_app.extensions["poster_service"]
    return ok(serialize_movie(record, poster_service, detail=True))


@bp.get("/api/titles/<content_id>/watch")
def title_watch(content_id: str):
    validate_no_unknown(request.args, {"region"})
    _parsed, record = _content_or_404(content_id)
    config = current_app.config["APP_CONFIG"]
    region = (request.args.get("region") or config.WATCH_REGION).strip().upper()
    if not _REGION_RE.fullmatch(region):
        raise ApiError(
            "INVALID_PARAMETER", "'region' must be a two-letter country code"
        )
    service = current_app.extensions["watch_service"]
    return ok(service.availability(record, region))


@bp.get("/api/titles/<content_id>/recommendations")
def title_recommendations(content_id: str):
    validate_no_unknown(request.args, {"limit"})
    parsed, _record = _content_or_404(content_id)
    model_state = current_app.extensions["model_state"]
    default_limit = model_state.engine.config.default_top_k if model_state.engine else 10
    limit = parse_int_param(
        request.args.get("limit"), "limit", maximum=1000, default=default_limit
    )
    service = current_app.extensions["recommendation_service"]
    session_id, created = session_for_request()
    results, source, ranking = service.recommend(parsed.public, limit, session_id)
    response, status = ok(
        results,
        meta={
            "content_id": parsed.public,
            "media_type": parsed.media_type,
            "source_title": source.get("title"),
            "returned": len(results),
            "method": "tfidf_cosine_content_based",
            "personalization": ranking,
        },
    )
    return attach_session_cookie(response, session_id, created), status
