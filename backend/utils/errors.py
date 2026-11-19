"""API error types and centralised Flask error handlers."""
from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Expected, client-facing API failure with a stable error code."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def register_error_handlers(app: Flask) -> None:
    """Register JSON-only errors; never leak stack traces or filesystem paths."""

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return jsonify({
            "success": False,
            "data": None,
            "meta": {},
            "error": error.to_dict(),
        }), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        code = {
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
        }.get(error.code, "HTTP_ERROR")
        return jsonify({
            "success": False,
            "data": None,
            "meta": {},
            "error": {
                "code": code,
                "message": error.description or "Request could not be completed",
                "details": {},
            },
        }), error.code or 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        logger.exception("Unhandled API error", exc_info=error)
        return jsonify({
            "success": False,
            "data": None,
            "meta": {},
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected server error occurred",
                "details": {},
            },
        }), 500
