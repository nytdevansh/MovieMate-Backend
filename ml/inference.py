"""Content-based recommendation inference (no Flask dependencies)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ml.artifacts import ArtifactError, LoadedArtifacts
from ml.features import cosine_scores, vectorize_fields
from ml.ids import parse_content_id
from ml.preprocessing import normalize_free_text, split_list


class UnknownMovieError(KeyError):
    """Raised when a content ID is not present in the trained model."""


@dataclass(frozen=True)
class Recommendation:
    content_id: str
    media_type: str
    source_id: int
    poster_key: str
    movie_id: int  # backward-compatible alias for source_id
    title: str
    poster_url: str
    genres: list[str]
    release_year: int | None
    rating: float | None
    similarity_score: float
    short_description: str
    vote_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "media_type": self.media_type,
            "source_id": self.source_id,
            "movie_id": self.movie_id,
            "title": self.title,
            "poster_url": self.poster_url,
            "genres": self.genres,
            "release_year": self.release_year,
            "rating": self.rating,
            "similarity_score": self.similarity_score,
            "short_description": self.short_description,
        }


class RecommendationEngine:
    """Retrieves similar titles with sparse cosine similarity.

    The query vector is rebuilt from the stored per-field metadata with the
    exact vectorizer and weights used during training. Movie model keys remain
    numeric strings for backward compatibility; TV series use ``tv-<id>``.
    """

    def __init__(self, artifacts: LoadedArtifacts) -> None:
        self.artifacts = artifacts
        self.config = artifacts.config
        self.metadata = artifacts.metadata
        self.id_to_index: dict[str, int] = {
            str(key): int(value) for key, value in artifacts.id_mapping["id_to_index"].items()
        }
        self.index_to_id: list[str] = [
            str(value) for value in artifacts.id_mapping["index_to_id"]
        ]

    @staticmethod
    def model_key(content_id: int | str) -> str:
        parsed = parse_content_id(content_id)
        return parsed.model_key if parsed else str(content_id)

    def has_movie(self, movie_id: int | str) -> bool:
        return self.model_key(movie_id) in self.metadata

    def has_title(self, content_id: int | str) -> bool:
        """ clearer alias used by the generic titles API."""
        return self.has_movie(content_id)

    def get_metadata(self, movie_id: int | str) -> dict[str, Any]:
        key = self.model_key(movie_id)
        try:
            return self.metadata[key]
        except KeyError as exc:
            raise UnknownMovieError(f"Title {movie_id!r} is not in the trained model") from exc

    def limit_for(self, requested: int | None) -> int:
        if requested is None:
            return self.config.default_top_k
        if isinstance(requested, bool) or not isinstance(requested, int):
            raise ValueError("limit must be an integer or None")
        return max(1, min(requested, self.config.max_top_k))

    def recommend(
        self,
        movie_id: int | str,
        limit: int | None = None,
        exclude_ids: Sequence[int | str] | None = None,
    ) -> list[Recommendation]:
        """Return the top-K most similar titles, excluding the query itself."""
        record = self.get_metadata(movie_id)
        top_k = self.limit_for(limit)
        query_vector = vectorize_fields([record], self.artifacts.vectorizer, self.config)
        scores = cosine_scores(query_vector, self.artifacts.matrix)

        key = self.model_key(movie_id)
        query_index = self.id_to_index[key]
        scores[query_index] = -np.inf
        for excluded in exclude_ids or []:
            index = self.id_to_index.get(self.model_key(excluded))
            if index is not None:
                scores[index] = -np.inf

        popularity = self._popularity_array()
        # Deterministic ordering: score, then popularity, then canonical ID.
        order = np.lexsort((self._ids_array(), -popularity, -scores))
        excluded_keys: set[tuple[str, str, str]] = set()
        results: list[Recommendation] = []

        for index in order:
            score = float(scores[int(index)])
            if not math.isfinite(score) or score <= 0.0:
                continue
            candidate_key = self.index_to_id[int(index)]
            candidate = self.metadata[candidate_key]
            dedupe_key = self._dedupe_key(candidate)
            if self.config.deduplicate_by_title_year and dedupe_key in excluded_keys:
                continue
            excluded_keys.add(dedupe_key)
            results.append(self._to_recommendation(candidate, score))
            if len(results) >= top_k:
                break
        return results

    def _to_recommendation(self, record: dict[str, Any], score: float) -> Recommendation:
        parsed = parse_content_id(record.get("content_id") or record.get("tmdb_id"))
        if parsed is None:
            raise UnknownMovieError(f"Could not parse content ID from {record.get('content_id')!r}")
        genres = split_list(record.get("genres"))
        vote_count = record.get("vote_count")
        return Recommendation(
            content_id=parsed.public,
            media_type=parsed.media_type,
            source_id=parsed.source_id,
            poster_key=parsed.poster_key,
            movie_id=parsed.source_id,
            title=str(record.get("title") or ""),
            poster_url=str(record.get("poster_url") or ""),
            genres=genres,
            release_year=record.get("year"),
            rating=record.get("vote_average"),
            similarity_score=round(score, 6),
            short_description=shorten(str(record.get("overview") or ""), 180),
            vote_count=int(vote_count) if vote_count is not None else None,
        )

    @staticmethod
    def _dedupe_key(record: dict[str, Any]) -> tuple[str, str, str]:
        title = normalize_free_text(record.get("title")).casefold()
        year = str(record.get("year") or "")
        media_type = "tv" if str(record.get("media_type")) == "tv" else "movie"
        return media_type, title, year

    def _popularity_array(self) -> np.ndarray:
        values = []
        for model_key in self.index_to_id:
            value = self.metadata[model_key].get("popularity")
            values.append(float(value) if isinstance(value, (int, float)) else 0.0)
        return np.asarray(values, dtype=np.float64)

    def _ids_array(self) -> np.ndarray:
        return np.asarray(self.index_to_id, dtype=str)


def shorten(text: str, max_length: int) -> str:
    """Truncate overviews on a word boundary without adding invented content."""
    text = normalize_free_text(text)
    if len(text) <= max_length:
        return text
    cut = text[:max_length].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "…" if cut else text[:max_length]


__all__ = [
    "ArtifactError",
    "Recommendation",
    "RecommendationEngine",
    "UnknownMovieError",
    "shorten",
]
