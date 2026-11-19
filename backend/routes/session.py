"""Anonymous session cookie helpers for personalisation endpoints."""
from __future__ import annotations

from typing import Any

from flask import current_app, request

from backend.services.feedback_service import SESSION_COOKIE, FeedbackService


def session_for_request() -> tuple[str, bool]:
    """Return the request's session ID and whether a new cookie is needed."""
    service: FeedbackService = current_app.extensions["feedback_service"]
    return service.ensure_session(request.cookies.get(SESSION_COOKIE))


def attach_session_cookie(response: Any, session_id: str, created: bool) -> Any:
    """Attach the HttpOnly session cookie only when a session was created."""
    if not created:
        return response
    config = current_app.config["APP_CONFIG"]
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=int(config.PERSONALIZATION_SESSION_MAX_AGE_SECONDS),
        httponly=True,
        samesite="Lax",
        secure=config.ENV_NAME == "production",
        path="/",
    )
    return response
