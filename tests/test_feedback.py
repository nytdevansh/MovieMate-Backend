"""Time-decayed repetition penalty and contextual-bandit learning tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import Config
from backend.services.feedback_service import _SESSION_ID_RE, FeedbackService
from backend.utils.errors import ApiError
from ml.artifacts import load_artifacts
from ml.inference import RecommendationEngine


@pytest.fixture
def service() -> FeedbackService:
    return FeedbackService(Config({
        "REPETITION_PENALTY_STRENGTH": 0.9,
        "REPETITION_PENALTY_HALF_LIFE_SECONDS": 60.0,
        "REPETITION_PENALTY_WINDOW_SECONDS": 120.0,
        "RL_EXPLORATION_RATE": 0.0,
    }))


def candidates(tiny_model: Path):
    engine = RecommendationEngine(load_artifacts(tiny_model))
    return engine.recommend(1, limit=20)


def test_recent_recommendations_are_penalized_and_promote_fresh_titles(
    service: FeedbackService, tiny_model: Path
):
    session, created = service.ensure_session(None)
    assert created is True

    first, first_meta = service.rank(session, None, candidates(tiny_model), 5, now=1000.0)
    second, second_meta = service.rank(session, None, candidates(tiny_model), 5, now=1001.0)

    assert first_meta["penalized_titles"] == 0
    assert second_meta["penalized_titles"] > 0
    assert first[0].content_id != second[0].content_id
    assert not {item.content_id for item in first[:2]} & {
        item.content_id for item in second[:2]
    }


def test_penalty_expires_after_the_timer_window(
    service: FeedbackService, tiny_model: Path
):
    session, _created = service.ensure_session(None)
    pool = candidates(tiny_model)
    service.rank(session, None, pool, 5, now=0.0)
    _later, meta = service.rank(session, None, pool, 5, now=1000.0)
    assert meta["penalized_titles"] == 0


def test_feedback_updates_title_and_genre_q_values(service: FeedbackService):
    session, _created = service.ensure_session(None)
    record = {"genres": "Drama | Thriller", "media_type": "movie"}

    saved = service.record_feedback(session, "movie-1", "save", record, now=10.0)
    dismissed = service.record_feedback(session, "movie-2", "dismiss", record, now=11.0)

    assert saved["reward"] == 1.0
    assert dismissed["reward"] == -1.0
    state = service._sessions[session]
    assert state.titles["movie-1"].q_value > 0.0
    assert state.titles["movie-2"].q_value < 0.0
    assert state.contexts["genre:drama"].q_value < 0.0
    assert "genre:thriller" in state.contexts


def test_feedback_validates_input(service: FeedbackService):
    session, _created = service.ensure_session(None)
    with pytest.raises(ApiError) as bad_event:
        service.record_feedback(session, "movie-1", "like")
    assert bad_event.value.status_code == 400

    with pytest.raises(ApiError) as bad_id:
        service.record_feedback(session, "not-a-title", "save")
    assert bad_id.value.status_code == 400


def test_session_ids_are_never_reused_from_arbitrary_input(service: FeedbackService):
    session, created = service.ensure_session("../../etc/passwd")
    assert created is True
    assert _SESSION_ID_RE.fullmatch(session)
