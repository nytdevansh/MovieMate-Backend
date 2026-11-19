"""In-memory catalogue loaded once at application startup.

The catalogue contains movies and TV series. Movie-only helpers keep their
original names/signatures; generic ``content_id`` helpers add TV support.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from backend.services.poster_service import PosterService
from ml.config import ModelConfig
from ml.ids import content_id_from_record, parse_content_id
from ml.inference import shorten
from ml.preprocessing import clean_dataset, split_list

logger = logging.getLogger(__name__)

SORT_OPTIONS: tuple[str, ...] = (
    "popularity_desc",
    "rating_desc",
    "release_date_desc",
    "title_asc",
)
MEDIA_TYPES: tuple[str, ...] = ("movie", "tv")


def _date_number(value: Any) -> int:
    text = str(value or "")
    return int(text.replace("-", "")) if len(text) == 10 and text[4] == "-" else 0


class MovieRepository:
    """Cleaned catalogue with filtering, search and discovery helpers."""

    def __init__(self, dataset_path: Any, poster_service: PosterService) -> None:
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Movie dataset not found: {path}. Set DATASET_PATH to the TMDB CSV."
            )
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Could not read movie dataset {path}: {exc}") from exc
        cleaned = clean_dataset(raw, ModelConfig())
        self.records: list[dict[str, Any]] = cleaned.records
        self.by_id: dict[int, dict[str, Any]] = {
            int(record["tmdb_id"]): record
            for record in self.records
            if str(record.get("media_type")) == "movie"
        }
        self.by_content_id: dict[str, dict[str, Any]] = {}
        for record in self.records:
            self.by_content_id[content_id_from_record(record).public] = record
        self.cleaning_report = cleaned.report
        self.poster_service = poster_service
        self.filter_options = self._build_filter_options()
        logger.info(
            "Loaded %s titles (%s movies, %s series) from the catalogue",
            len(self.records),
            len(self.by_id),
            len(self.records) - len(self.by_id),
        )

    def get(self, movie_id: int) -> dict[str, Any] | None:
        """Legacy movie-only lookup by numeric TMDB ID."""
        return self.by_id.get(int(movie_id))

    def get_content(self, content_id: int | str) -> dict[str, Any] | None:
        """Lookup by canonical movie-<id> / tv-<id> identifier."""
        parsed = parse_content_id(content_id)
        if parsed is None:
            return None
        return self.by_content_id.get(parsed.public)

    def list_movies(
        self,
        *,
        genre: str | None = None,
        year: int | None = None,
        language: str | None = None,
        min_rating: float | None = None,
        sort: str = "popularity_desc",
        page: int = 1,
        per_page: int = 24,
        media_type: str | None = "movie",
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (page of records, total matching count)."""
        matches = [
            record for record in self.records
            if (media_type is None or record.get("media_type") == media_type)
            and self._matches(record, genre, year, language, min_rating)
        ]
        matches.sort(key=self._sort_key(sort))
        start = (page - 1) * per_page
        return matches[start:start + per_page], len(matches)

    def search(self, query: str, limit: int, media_type: str | None = None) -> list[dict[str, Any]]:
        """Rank title, original title, genres, cast, director and overview matches."""
        target = query.casefold()
        terms = [term for term in target.split() if term]
        scored: list[tuple[float, float, str, dict[str, Any]]] = []
        for record in self.records:
            if media_type is not None and record.get("media_type") != media_type:
                continue
            score = self._search_score(record, target, terms)
            if score > 0:
                scored.append((score, self._number(record.get("popularity")), str(record.get("title", "")), record))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].casefold()))
        return [item[3] for item in scored[:limit]]

    def trending(self, limit: int, media_type: str | None = None) -> list[dict[str, Any]]:
        records = [
            record for record in self.records
            if media_type is None or record.get("media_type") == media_type
        ]
        records = sorted(
            records,
            key=lambda record: (
                -self._number(record.get("popularity")),
                -self._number(record.get("vote_count")),
                str(content_id_from_record(record).public),
            ),
        )
        return records[:limit]

    def _build_filter_options(self) -> dict[str, Any]:
        genres: dict[str, str] = {}
        languages: dict[str, str] = {}
        years: set[int] = set()
        ratings: list[float] = []
        for record in self.records:
            for genre in split_list(record.get("genres")):
                genres.setdefault(genre.casefold(), genre)
            language = str(record.get("original_language") or "")
            if language:
                languages.setdefault(language.casefold(), language)
            if record.get("year") is not None:
                years.add(int(record["year"]))
            if isinstance(record.get("vote_average"), (int, float)):
                ratings.append(float(record["vote_average"]))
        return {
            "genres": sorted(genres.values(), key=str.casefold),
            "years": sorted(years, reverse=True),
            "languages": sorted(languages.values(), key=str.casefold),
            "rating_range": {
                "min": round(min(ratings), 1) if ratings else None,
                "max": round(max(ratings), 1) if ratings else None,
            },
        }

    # ----------------------------------------------------------------- private
    @staticmethod
    def _matches(
        record: Mapping[str, Any],
        genre: str | None,
        year: int | None,
        language: str | None,
        min_rating: float | None,
    ) -> bool:
        if genre and genre.casefold() not in {
            value.casefold() for value in split_list(record.get("genres"))
        }:
            return False
        if year is not None and record.get("year") != year:
            return False
        if language and str(record.get("original_language") or "").casefold() != language.casefold():
            return False
        if min_rating is not None:
            rating = record.get("vote_average")
            if not isinstance(rating, (int, float)) or rating < min_rating:
                return False
        return True

    @staticmethod
    def _sort_key(sort: str) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
        if sort == "rating_desc":
            return lambda r: (
                -MovieRepository._number(r.get("vote_average")),
                -MovieRepository._number(r.get("vote_count")),
                str(r.get("title", "")).casefold(),
            )
        if sort == "release_date_desc":
            return lambda r: (
                -_date_number(r.get("release_date")),
                str(r.get("title", "")).casefold(),
            )
        if sort == "title_asc":
            return lambda r: (
                str(r.get("title", "")).casefold(),
                str(content_id_from_record(r).public),
            )
        return lambda r: (
            -MovieRepository._number(r.get("popularity")),
            -MovieRepository._number(r.get("vote_count")),
            str(content_id_from_record(r).public),
        )

    @staticmethod
    def _search_score(
        record: Mapping[str, Any], target: str, terms: list[str]
    ) -> float:
        title = str(record.get("title") or "").casefold()
        original = str(record.get("original_title") or "").casefold()
        genres = str(record.get("genres") or "").casefold()
        cast = str(record.get("cast") or "").casefold()
        director = str(record.get("director") or "").casefold()
        overview = str(record.get("overview") or "").casefold()

        score = 0.0
        if title == target:
            score += 100.0
        elif title.startswith(target):
            score += 70.0
        elif target in title:
            score += 50.0
        if original and target in original:
            score += 30.0
        if terms:
            if all(term in f"{title} {original}" for term in terms):
                score += 10.0
            if any(term in genres for term in terms):
                score += 20.0
            if any(term in cast for term in terms) or any(term in director for term in terms):
                score += 12.0
            if any(term in overview for term in terms):
                score += 5.0
        return score

    @staticmethod
    def _number(value: Any) -> float:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def serialize_movie(
    record: Mapping[str, Any],
    poster_service: PosterService,
    *,
    detail: bool = False,
) -> dict[str, Any]:
    """Convert an internal record to public JSON (no file paths or secrets)."""
    identity = content_id_from_record(record)
    resolution = poster_service.resolve(record)
    data: dict[str, Any] = {
        "content_id": identity.public,
        "media_type": identity.media_type,
        "source_id": identity.source_id,
        "movie_id": identity.source_id,  # backward-compatible alias
        "title": record.get("title") or "",
        "release_year": record.get("year"),
        "genres": split_list(record.get("genres")),
        "rating": record.get("vote_average"),
        "poster_url": resolution.poster_url,
        "poster_source": resolution.poster_source,
        "backdrop_url": resolution.backdrop_url,
        "short_description": shorten(str(record.get("overview") or ""), 180),
        "original_language": record.get("original_language") or None,
    }
    if not detail:
        return data

    data.update({
        "imdb_id": record.get("imdb_id") or None,
        "original_title": record.get("original_title") or None,
        "release_date": record.get("release_date") or None,
        "runtime_minutes": record.get("runtime_minutes"),
        "vote_count": record.get("vote_count"),
        "popularity": record.get("popularity"),
        "overview": record.get("overview") or "",
        "production_countries": split_list(record.get("production_countries")),
        "cast": split_list(record.get("cast")),
        "director": split_list(record.get("director")),
        "india_streaming": split_list(record.get("india_streaming")),
        "india_rent": split_list(record.get("india_rent")),
        "india_buy": split_list(record.get("india_buy")),
        "homepage": record.get("homepage") or None,
        "adult": bool(record.get("adult")),
        "backdrop_source": resolution.backdrop_source,
        "number_of_seasons": record.get("number_of_seasons"),
        "number_of_episodes": record.get("number_of_episodes"),
        "status": record.get("status") or None,
        "tv_type": record.get("tv_type") or None,
    })
    return data
