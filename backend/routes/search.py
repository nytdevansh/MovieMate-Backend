"""GET /api/search?q=..."""
from __future__ import annotations

from flask import Blueprint, current_app, request

from backend.services.movie_service import serialize_movie
from backend.utils.responses import ok
from backend.utils.validators import clean_query, parse_int_param, validate_no_unknown

bp = Blueprint("search", __name__)


@bp.get("/api/search")
def search():
    config = current_app.config["APP_CONFIG"]
    validate_no_unknown(request.args, {"q", "limit"})
    query = clean_query(request.args.get("q"))
    limit = parse_int_param(
        request.args.get("limit"), "limit", maximum=config.MAX_SEARCH_RESULTS,
        default=config.MAX_SEARCH_RESULTS,
    )
    repository = current_app.extensions["movie_repository"]
    poster_service = current_app.extensions["poster_service"]
    results = [
        serialize_movie(record, poster_service)
        for record in repository.search(query, limit)
    ]
    return ok(results, meta={"query": query, "count": len(results)})
