"""Bridges the Flask-independent engine to API responses."""
from __future__ import annotations

from typing import Any

from backend.models.model_loader import ModelState
from backend.services.feedback_service import FeedbackService
from backend.services.movie_service import MovieRepository, serialize_movie
from backend.services.poster_service import PosterService
from backend.utils.errors import ApiError
from ml.ids import parse_content_id
from ml.inference import UnknownMovieError


class RecommendationService:
    def __init__(
        self,
        model_state: ModelState,
        movie_repository: MovieRepository,
        poster_service: PosterService,
        feedback_service: FeedbackService,
    ) -> None:
        self.model_state = model_state
        self.movies = movie_repository
        self.posters = poster_service
        self.feedback = feedback_service

    def recommend(
        self,
        content_id: int | str,
        limit: int,
        session_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Return (recommendation payloads, source payload, ranking metadata)."""
        if not self.model_state.available or self.model_state.engine is None:
            raise ApiError(
                "MODEL_UNAVAILABLE",
                "The recommendation model is not loaded. Train a model or install an export.",
                status_code=503,
                details={"reason": self.model_state.error or "model not initialised"},
            )

        parsed = parse_content_id(content_id)
        if parsed is None:
            raise ApiError(
                "INVALID_CONTENT_ID",
                f"Invalid content ID {content_id!r}. Use movie-<id> or tv-<id>.",
                status_code=400,
            )

        engine = self.model_state.engine
        model_key = parsed.model_key
        if not engine.has_movie(model_key):
            if self.movies.get_content(parsed.public) is not None:
                raise ApiError(
                    "MOVIE_NOT_IN_MODEL",
                    f"Title {parsed.public} exists in the catalogue but not in the trained model",
                    status_code=422,
                )
            raise ApiError(
                "MOVIE_NOT_FOUND", f"Title {parsed.public} was not found", status_code=404
            )

        source_record = self.movies.get_content(parsed.public) or engine.get_metadata(model_key)
        # Retrieve a wider candidate pool: the temporal penalty must be able to
        # promote fresh titles instead of only reordering the same top slice.
        candidate_limit = min(engine.config.max_top_k, max(limit * 4, limit + 20))
        try:
            candidates = engine.recommend(model_key, limit=candidate_limit)
        except UnknownMovieError as exc:
            raise ApiError("MOVIE_NOT_FOUND", str(exc), status_code=404) from exc

        if session_id and self.feedback:
            results, ranking_meta = self.feedback.rank(
                session_id, source_record, candidates, limit
            )
        else:
            results = candidates[:limit]
            ranking_meta = {
                "ranking": "tfidf_cosine_content_based",
                "personalized": False,
            }

        payload: list[dict[str, Any]] = []
        for recommendation in results:
            record = self.movies.get_content(recommendation.content_id) or engine.get_metadata(
                recommendation.content_id
            )
            resolution = self.posters.resolve(record)
            data = recommendation.to_dict()
            data["poster_url"] = resolution.poster_url
            data["poster_source"] = resolution.poster_source
            payload.append(data)

        return (
            payload,
            serialize_movie(source_record, self.posters, detail=False),
            ranking_meta,
        )
