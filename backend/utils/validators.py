"""Input validation. All failures raise ApiError(400) with actionable messages."""
from __future__ import annotations

import re
from typing import Any

from backend.utils.errors import ApiError

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def parse_int_param(
    value: str | None,
    name: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
    default: int | None = None,
) -> int:
    if value is None or value.strip() == "":
        if default is None:
            raise ApiError("INVALID_PARAMETER", f"Missing required integer parameter '{name}'")
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ApiError("INVALID_PARAMETER", f"'{name}' must be an integer") from None
    if number < minimum or (maximum is not None and number > maximum):
        bound = f"between {minimum} and {maximum}" if maximum else f">= {minimum}"
        raise ApiError("INVALID_PARAMETER", f"'{name}' must be {bound}")
    return number


def parse_float_param(
    value: str | None,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    default: float | None = None,
) -> float:
    if value is None or value.strip() == "":
        if default is None:
            raise ApiError("INVALID_PARAMETER", f"Missing required numeric parameter '{name}'")
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ApiError("INVALID_PARAMETER", f"'{name}' must be a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ApiError("INVALID_PARAMETER", f"'{name}' must be a finite number")
    if number < minimum or (maximum is not None and number > maximum):
        bound = f"between {minimum} and {maximum}" if maximum else f">= {minimum}"
        raise ApiError("INVALID_PARAMETER", f"'{name}' must be {bound}")
    return number


def parse_choice(
    value: str | None, name: str, choices: tuple[str, ...], default: str
) -> str:
    if value is None or value.strip() == "":
        return default
    choice = value.strip().lower()
    if choice not in choices:
        raise ApiError(
            "INVALID_PARAMETER",
            f"'{name}' must be one of: {', '.join(choices)}",
            details={"received": value},
        )
    return choice


def clean_query(value: str | None, *, minimum: int = 1, maximum: int = 100) -> str:
    """Strip control characters and collapse whitespace; reject empty queries."""
    if value is None:
        raise ApiError("INVALID_PARAMETER", "Missing required query parameter 'q'")
    text = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", value)).strip()
    if len(text) < minimum:
        raise ApiError("INVALID_PARAMETER", "Search query 'q' must not be empty")
    if len(text) > maximum:
        raise ApiError(
            "INVALID_PARAMETER",
            f"Search query 'q' must be at most {maximum} characters",
        )
    return text


def optional_text(value: str | None, name: str, maximum: int = 100) -> str | None:
    if value is None or value.strip() == "":
        return None
    text = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", value)).strip()
    if len(text) > maximum:
        raise ApiError(
            "INVALID_PARAMETER",
            f"'{name}' must be at most {maximum} characters",
        )
    return text


def validate_no_unknown(args: Any, allowed: set[str]) -> None:
    """Reject unknown query keys so typos do not silently change results."""
    unknown = sorted(set(args.keys()) - allowed)
    if unknown:
        raise ApiError(
            "INVALID_PARAMETER",
            "Unknown query parameter(s): " + ", ".join(unknown),
            details={"allowed": sorted(allowed)},
        )
