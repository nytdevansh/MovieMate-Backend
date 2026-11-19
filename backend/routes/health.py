"""GET /api/health"""
from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, current_app

from backend.utils.responses import ok

bp = Blueprint("health", __name__)


@bp.get("/api/health")
def health():
    config = current_app.config["APP_CONFIG"]
    repository = current_app.extensions["movie_repository"]
    model_state = current_app.extensions["model_state"]
    model_info = model_state.info or {}
    return ok({
        "status": "ok" if model_state.available else "degraded",
        "movies_loaded": len(repository.by_id),
        "series_loaded": len(repository.records) - len(repository.by_id),
        "titles_loaded": len(repository.records),
        "model_loaded": model_state.available,
        "model_schema_version": model_info.get("artifact_schema_version"),
        "model_error": model_state.error,
        "configuration": config.public_summary(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
