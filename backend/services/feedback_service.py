"""Session-scoped personalisation for recommendation ranking.

Two small, transparent mechanisms sit on top of the content-based model:

* a **time-decay repetition penalty** — every title we show is remembered with
  a timestamp. The memory fades exponentially and is discarded after a window,
  so the same titles stop dominating without being banned forever;
* an **epsilon-greedy contextual bandit** — opens and saves are positive
  rewards, removes and "show fewer like this" are negative rewards. Q-values are
  updated for the title and for the source genres. A deterministic epsilon
  jitter keeps exploration small and stable for each session.

State is anonymous, in-process and keyed by an HttpOnly cookie. No account,
profile, or personal data is stored, and state disappears on restart.
"""
from __future__ import annotations

import hashlib
import math
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from backend.utils.errors import ApiError
from ml.ids import parse_content_id
from ml.inference import Recommendation
from ml.preprocessing import split_list

SESSION_COOKIE = "mm_session"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{22,64}$")

# Bounded rewards. These are training signals, not claims about a user.
EVENT_REWARDS: dict[str, float] = {
    "open": 0.35,
    "save": 1.0,
    "remove": -0.35,
    "dismiss": -1.0,
}


@dataclass
class TitleMemory:
    impressions: deque[float] = field(default_factory=deque)
    q_value: float = 0.0
    updates: int = 0
    positive: int = 0
    negative: int = 0


@dataclass
class ContextMemory:
    q_value: float = 0.0
    updates: int = 0


@dataclass
class SessionMemory:
    titles: dict[str, TitleMemory] = field(default_factory=dict)
    contexts: dict[str, ContextMemory] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class FeedbackService:
    """Rank and learn without touching the trained TF-IDF artifacts."""

    def __init__(self, config: Any) -> None:
        self.penalty_half_life = float(
            getattr(config, "REPETITION_PENALTY_HALF_LIFE_SECONDS", 21600)
        )
        self.penalty_window = float(
            getattr(config, "REPETITION_PENALTY_WINDOW_SECONDS", 86400)
        )
        self.penalty_strength = min(
            0.95, max(0.0, float(getattr(config, "REPETITION_PENALTY_STRENGTH", 0.55)))
        )
        self.learning_rate = min(
            1.0, max(0.0, float(getattr(config, "RL_LEARNING_RATE", 0.25)))
        )
        self.exploration_rate = min(
            0.25, max(0.0, float(getattr(config, "RL_EXPLORATION_RATE", 0.08)))
        )
        self.title_bonus = float(getattr(config, "RL_TITLE_BONUS", 0.45))
        self.context_bonus = float(getattr(config, "RL_CONTEXT_BONUS", 0.25))
        self.session_max_age = float(
            getattr(config, "PERSONALIZATION_SESSION_MAX_AGE_SECONDS", 604800)
        )
        self.max_sessions = int(getattr(config, "PERSONALIZATION_MAX_SESSIONS", 5000))
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------------------- sessions
    def new_session_id(self) -> str:
        return secrets.token_urlsafe(24)

    def ensure_session(self, session_id: str | None) -> tuple[str, bool]:
        """Return a usable session ID and whether it was newly created."""
        now = time.time()
        with self._lock:
            if session_id and _SESSION_ID_RE.fullmatch(session_id):
                state = self._sessions.get(session_id)
                if state is not None and now - state.updated_at <= self.session_max_age:
                    state.updated_at = now
                    return session_id, False
            session_id = self.new_session_id()
            self._sessions[session_id] = SessionMemory(updated_at=now)
            self._trim(now)
            return session_id, True

    def _trim(self, now: float) -> None:
        if len(self._sessions) <= self.max_sessions:
            return
        expired = [
            key for key, value in self._sessions.items()
            if now - value.updated_at > self.session_max_age
        ]
        for key in expired:
            self._sessions.pop(key, None)
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].updated_at)
            self._sessions.pop(oldest, None)

    # ----------------------------------------------------------------- ranking
    def rank(
        self,
        session_id: str,
        source_record: Mapping[str, Any] | None,
        candidates: Sequence[Recommendation],
        limit: int,
        *,
        now: float | None = None,
    ) -> tuple[list[Recommendation], dict[str, Any]]:
        """Apply the time penalty and bandit scores, then take ``limit`` titles."""
        timestamp = time.time() if now is None else now
        contexts = self._contexts(source_record)
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = SessionMemory(updated_at=timestamp)
            self._forget_old_impressions(state, timestamp)

            scored: list[tuple[float, int, Recommendation, float, float]] = []
            for position, candidate in enumerate(candidates):
                memory = state.titles.get(candidate.content_id)
                repetition = self._repetition_factor(memory, timestamp)
                title_q = memory.q_value if memory else 0.0
                context_q = (
                    sum(
                        state.contexts[context].q_value
                        for context in contexts
                        if context in state.contexts
                    ) / len(contexts)
                    if contexts else 0.0
                )
                base = max(0.0, float(candidate.similarity_score))
                multiplier = 1.0 + self.title_bonus * title_q + self.context_bonus * context_q
                multiplier = max(0.45, min(1.75, multiplier))
                exploration = self._exploration(session_id, candidate.content_id, base)
                score = base * (1.0 - repetition) * multiplier + exploration
                scored.append((score, -position, candidate, repetition, exploration))

            scored.sort(key=lambda row: (-row[0], row[1], row[2].content_id))
            selected = scored[:max(0, limit)]
            for _score, _position, candidate, _repetition, _exploration in selected:
                memory = state.titles.setdefault(
                    candidate.content_id, TitleMemory()
                )
                memory.impressions.append(timestamp)
            state.updated_at = timestamp

            penalized = sum(1 for row in scored if row[3] > 0.0)
            selected_penalized = sum(1 for row in selected if row[3] > 0.0)
            explored = sum(1 for row in selected if row[4] != 0.0)
            return [row[2] for row in selected], {
                "ranking": "time_decayed_repetition_penalty_epsilon_greedy_bandit",
                "personalized": True,
                "timer_half_life_seconds": int(self.penalty_half_life),
                "penalty_strength": round(self.penalty_strength, 3),
                "penalized_titles": penalized,
                "penalized_in_results": selected_penalized,
                "explored_titles": explored,
                "learning_rate": round(self.learning_rate, 3),
            }

    def _forget_old_impressions(self, state: SessionMemory, now: float) -> None:
        for memory in state.titles.values():
            impressions = memory.impressions
            while impressions and now - impressions[0] > self.penalty_window:
                impressions.popleft()

    def _repetition_factor(self, memory: TitleMemory | None, now: float) -> float:
        if memory is None or not memory.impressions:
            return 0.0
        decayed = sum(
            math.pow(0.5, max(0.0, now - shown) / self.penalty_half_life)
            for shown in memory.impressions
        )
        return min(0.9, self.penalty_strength * decayed)

    def _exploration(self, session_id: str, content_id: str, base: float) -> float:
        """Small deterministic jitter so exploration is reproducible per session."""
        if self.exploration_rate <= 0.0 or base <= 0.0:
            return 0.0
        digest = hashlib.sha256(
            f"{session_id}:{content_id}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64)
        if unit >= self.exploration_rate:
            return 0.0
        jitter = int.from_bytes(digest[8:16], "big") / float(2**64)
        return base * (jitter - 0.5) * 0.5

    # ---------------------------------------------------------------- learning
    def record_feedback(
        self,
        session_id: str,
        content_id: str | int,
        event: str,
        record: Mapping[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Update the anonymous Q-values from one bounded reward event."""
        parsed = parse_content_id(content_id)
        if parsed is None:
            raise ApiError(
                "INVALID_CONTENT_ID",
                f"Invalid content ID {content_id!r}. Use movie-<id> or tv-<id>.",
                status_code=400,
            )
        if event not in EVENT_REWARDS:
            raise ApiError(
                "INVALID_PARAMETER",
                "'event' must be one of: " + ", ".join(EVENT_REWARDS),
                details={"allowed": list(EVENT_REWARDS)},
            )
        reward = EVENT_REWARDS[event]
        timestamp = time.time() if now is None else now
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = SessionMemory(updated_at=timestamp)

            memory = state.titles.setdefault(parsed.public, TitleMemory())
            memory.q_value = round(
                (1.0 - self.learning_rate) * memory.q_value
                + self.learning_rate * reward,
                6,
            )
            memory.updates += 1
            if reward > 0:
                memory.positive += 1
            elif reward < 0:
                memory.negative += 1

            for context in self._contexts(record):
                context_memory = state.contexts.setdefault(context, ContextMemory())
                context_memory.q_value = round(
                    (1.0 - self.learning_rate) * context_memory.q_value
                    + self.learning_rate * reward,
                    6,
                )
                context_memory.updates += 1
            state.updated_at = timestamp

        return {
            "content_id": parsed.public,
            "event": event,
            "reward": reward,
            "personalization": "session_only",
        }

    @staticmethod
    def _contexts(record: Mapping[str, Any] | None) -> list[str]:
        if not record:
            return []
        media_type = "tv" if str(record.get("media_type")) == "tv" else "movie"
        genres = [genre.casefold() for genre in split_list(record.get("genres"))[:3]]
        return [f"genre:{genre}" for genre in genres] + [f"type:{media_type}"]


__all__ = ["EVENT_REWARDS", "FeedbackService", "SESSION_COOKIE"]
