"""Canonical content identifiers shared by the catalogue, model and API.

Movies keep their historical numeric TMDB ID as the model key so existing
clients and trained artifacts remain valid. TV series use ``tv-<tmdb_id>``
because movie and TV IDs live in separate TMDB namespaces and can collide.

The public identifier exposed to the UI is always explicit:
``movie-<tmdb_id>`` or ``tv-<tmdb_id>``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CONTENT_ID_RE = re.compile(r"^(movie|tv|series)[-:]([0-9]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ContentId:
    """A parsed, unambiguous identifier for one catalogue title."""

    media_type: str  # "movie" | "tv"
    source_id: int
    public: str  # movie-123 / tv-123 (or 123 for legacy movie callers)
    model_key: str  # "123" for movies, "tv-123" for series
    poster_key: str  # "123" for movies, "tv-123" for series

    @property
    def source_path(self) -> str:
        return "movie" if self.media_type == "movie" else "tv"


def parse_content_id(value: Any) -> ContentId | None:
    """Parse any accepted identifier form, or return None when invalid.

    Accepted forms:
      * int / numeric string -> movie (legacy compatibility)
      * ``movie-123`` / ``movie:123`` -> movie
      * ``tv-123`` / ``tv:123`` / ``series-123`` -> TV series
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None

    match = _CONTENT_ID_RE.fullmatch(text)
    if match:
        kind = "tv" if match.group(1).lower() in {"tv", "series"} else "movie"
        source_id = int(match.group(2))
    elif text.isdigit():
        kind, source_id = "movie", int(text)
    else:
        return None

    if source_id <= 0:
        return None
    return make_content_id(kind, source_id)


def make_content_id(media_type: str, source_id: int) -> ContentId:
    """Build a ContentId from a validated media type and TMDB ID."""
    kind = "tv" if str(media_type).strip().lower() in {"tv", "series"} else "movie"
    identifier = int(source_id)
    if identifier <= 0:
        raise ValueError("TMDB ID must be a positive integer")
    if kind == "movie":
        key = str(identifier)
        return ContentId(kind, identifier, f"movie-{identifier}", key, key)
    key = f"tv-{identifier}"
    return ContentId(kind, identifier, key, key, key)


def content_id_from_record(record: Any) -> ContentId:
    """Derive the canonical ID from a cleaned dataset record."""
    if record is None:
        raise ValueError("A dataset record is required")
    if hasattr(record, "get"):
        raw_id = record.get("content_id") or record.get("tmdb_id")
        media_type = record.get("media_type") or "movie"
        source_id = record.get("source_tmdb_id") or record.get("tmdb_id")
    else:  # pragma: no cover - defensive for mapping-like objects
        raw_id = getattr(record, "content_id", None) or getattr(record, "tmdb_id")
        media_type = getattr(record, "media_type", "movie")
        source_id = getattr(record, "source_tmdb_id", None) or getattr(record, "tmdb_id")

    parsed = parse_content_id(raw_id)
    if parsed is not None:
        return parsed
    try:
        identifier = int(source_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot derive a content ID from {raw_id!r}") from exc
    return make_content_id(media_type, identifier)


__all__ = ["ContentId", "content_id_from_record", "make_content_id", "parse_content_id"]
