"""POST /api/feedback — anonymous learning signals for recommendation ranking."""
from __future__ import annotations

from flask import Blueprint, current_app, request

from backend.routes.session import attach_session_cookie, session_for_request
from backend.utils.errors import ApiError
from backend.utils.responses import ok
from backend.utils.validators import validate_no_unknown
from ml.ids import parse_content_id

bp = Blueprint("feedback", __name__)

_ALLOWED_FIELDS = {"content_id", "event"}


@bp.post("/api/feedback")
def record_feedback():
    validate_no_unknown(request.args, set())
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(
            "INVALID_BODY", "Send a JSON object with 'content_id' and 'event'"
        )
    unknown = sorted(set(body) - _ALLOWED_FIELDS)
    if unknown:
        raise ApiError(
            "INVALID_BODY",
            "Unknown field(s): " + ", ".join(unknown),
            details={"allowed": sorted(_ALLOWED_FIELDS)},
        )

    content_id = body.get("content_id")
    raw_event = body.get("event")
    if not isinstance(raw_event, str):
        raise ApiError("INVALID_PARAMETER", "'event' must be a string")
    event = raw_event.strip().lower()

    parsed = parse_content_id(content_id)
    if parsed is None:
        raise ApiError(
            "INVALID_CONTENT_ID",
            "Invalid content ID. Use movie-<id> or tv-<id>.",
        )

    repository = current_app.extensions["movie_repository"]
    record = repository.get_content(parsed.public)
    if record is None:
        model_state = current_app.extensions["model_state"]
        if model_state.engine and model_state.engine.has_title(parsed.public):
            record = model_state.engine.get_metadata(parsed.model_key)
        else:
            raise ApiError(
                "TITLE_NOT_FOUND", f"Title {parsed.public} was not found", status_code=404
            )

    session_id, created = session_for_request()
    service = current_app.extensions["feedback_service"]
    result = service.record_feedback(session_id, parsed.public, event, record)
    response, status = ok(result, meta={"source": "anonymous_session"})
    return attach_session_cookie(response, session_id, created), status
