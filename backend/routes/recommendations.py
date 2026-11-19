"""GET /api/recommendations/<movie_id>?limit=10"""
from __future__ import annotations

from flask import Blueprint, current_app, request

from backend.routes.session import attach_session_cookie, session_for_request
from backend.utils.responses import ok
from backend.utils.validators import parse_int_param, validate_no_unknown

bp = Blueprint("recommendations", __name__)


@bp.get("/api/recommendations/<int:movie_id>")
def recommendations(movie_id: int):
    validate_no_unknown(request.args, {"limit"})
    service = current_app.extensions["recommendation_service"]
    model_state = current_app.extensions["model_state"]
    default_limit = model_state.engine.config.default_top_k if model_state.engine else 10
    limit = parse_int_param(
        request.args.get("limit"), "limit", maximum=1000, default=default_limit
    )
    session_id, created = session_for_request()
    results, source, ranking = service.recommend(movie_id, limit, session_id)
    response, status = ok(
        results,
        meta={
            "movie_id": movie_id,
            "source_title": source.get("title"),
            "returned": len(results),
            "method": "tfidf_cosine_content_based",
            "personalization": ranking,
        },
    )
    return attach_session_cookie(response, session_id, created), status
