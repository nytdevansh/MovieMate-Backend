"""Recommendation engine behaviour tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from ml.artifacts import load_artifacts
from ml.inference import RecommendationEngine, UnknownMovieError


@pytest.fixture
def engine(tiny_model: Path) -> RecommendationEngine:
    return RecommendationEngine(load_artifacts(tiny_model))


def test_recommendations_exclude_self_and_are_ranked(engine: RecommendationEngine):
    results = engine.recommend(1, limit=5)
    assert 0 < len(results) <= 5
    assert all(result.movie_id != 1 for result in results)
    assert len({result.movie_id for result in results}) == len(results)
    scores = [result.similarity_score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_recommendation_payload_fields(engine: RecommendationEngine):
    result = engine.recommend(1, limit=1)[0]
    payload = result.to_dict()
    for key in (
        "movie_id", "title", "poster_url", "genres", "release_year",
        "rating", "similarity_score", "short_description",
    ):
        assert key in payload
    assert isinstance(payload["genres"], list)


def test_limit_is_clamped_and_unknown_id_fails(engine: RecommendationEngine):
    assert len(engine.recommend(1, limit=999)) <= engine.config.max_top_k
    assert len(engine.recommend(1, limit=None)) == engine.config.default_top_k
    with pytest.raises(UnknownMovieError):
        engine.recommend(999999)


def test_explicit_exclusion(engine: RecommendationEngine):
    first = engine.recommend(1, limit=5)
    excluded = [first[0].movie_id, first[1].movie_id]
    second = engine.recommend(1, limit=5, exclude_ids=excluded)
    assert all(result.movie_id not in excluded for result in second)


def test_short_description_is_truncated_without_invention():
    from ml.inference import shorten

    assert shorten("short overview", 180) == "short overview"
    long_text = "word " * 100
    result = shorten(long_text, 40)
    assert len(result) <= 41 and result.endswith("…")
