"""Consistent JSON envelope for every API response."""
from __future__ import annotations

from typing import Any

from flask import jsonify

from backend.utils.errors import ApiError


def ok(
    data: Any = None,
    *,
    meta: dict[str, Any] | None = None,
    status: int = 200,
) -> tuple[Any, int]:
    """Success envelope: {success, data, meta, error}."""
    return jsonify({
        "success": True,
        "data": data,
        "meta": meta or {},
        "error": None,
    }), status


def pagination_meta(page: int, per_page: int, total: int) -> dict[str, Any]:
    """Standard pagination metadata (page is 1-based)."""
    import math

    total_pages = math.ceil(total / per_page) if per_page else 0
    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
    }


def error_response(error: ApiError) -> tuple[Any, int]:
    """Error envelope with a stable machine-readable code."""
    return jsonify({
        "success": False,
        "data": None,
        "meta": {},
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
    }), error.status_code
