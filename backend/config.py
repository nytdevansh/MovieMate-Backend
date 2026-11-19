"""Environment-driven configuration for development and production."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOVIES_DATASET = "Frontend/movie_dataset_builder/output/movies_2023_2026.csv"
COMBINED_DATASET = "Frontend/movie_dataset_builder/output/all_titles.csv"
# Deployed checkouts ship a compact copy of the combined catalogue in data/.
# Local checkouts keep using the builder output when it is available, and the
# original movie CSV remains the final fallback.
DEPLOY_DATASET = "data/all_titles.csv"
DEFAULT_DATASET = (
    COMBINED_DATASET
    if (PROJECT_ROOT / COMBINED_DATASET).is_file()
    else DEPLOY_DATASET
    if (PROJECT_ROOT / DEPLOY_DATASET).is_file()
    else MOVIES_DATASET
)
DEFAULT_POSTER_DIRS = (
    "assets/posters",
    "Frontend/movie_dataset_builder/output/posters",
    "Frontend/movie_dataset_builder/output/tv_posters",
)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError as exc:
        raise ValueError(f"Expected an integer, got {value!r}") from exc


def _as_list(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


class Config:
    """Base configuration. All secrets come from the environment, never code."""

    ENV_NAME = "development"

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        # The dataset builder's .env is the existing home of the TMDB token.
        # It is read only for server-side watch-provider lookups and never
        # appears in an API response or the browser.
        load_dotenv(PROJECT_ROOT / "Frontend/movie_dataset_builder/.env")
        self.DEBUG = _as_bool(os.getenv("FLASK_DEBUG"), False)
        self.TESTING = False
        self.HOST = os.getenv("FLASK_HOST", "127.0.0.1")
        self.PORT = _as_int(os.getenv("FLASK_PORT"), 5000)
        self.SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-secret-change-me")

        self.DATASET_PATH = self._resolve_path(
            os.getenv("DATASET_PATH", DEFAULT_DATASET)
        )
        self.MODEL_DIR = self._resolve_path(os.getenv("MODEL_DIR", "model"))
        self.POSTER_DIRS = [
            self._resolve_path(value)
            for value in _as_list(os.getenv("POSTER_DIRS"), DEFAULT_POSTER_DIRS)
        ]
        self.BACKDROP_DIRS = [
            self._resolve_path(value)
            for value in _as_list(os.getenv("BACKDROP_DIRS"), ("assets/backdrops",))
        ]
        self.FALLBACK_POSTER_URL = os.getenv(
            "FALLBACK_POSTER_URL", "/media/fallback/poster.svg"
        )
        self.WEB_DIR = PROJECT_ROOT / "web"
        self.FRONTEND_DIR = PROJECT_ROOT / "Frontend"
        self.ASSETS_DIR = PROJECT_ROOT / "assets"

        self.TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN", "").strip()
        self.TMDB_API_ROOT = os.getenv("TMDB_API_ROOT", "https://api.themoviedb.org/3").rstrip("/")
        self.WATCH_REGION = (os.getenv("WATCH_REGION") or "IN").strip().upper()[:2]
        self.WATCH_TIMEOUT_SECONDS = float(os.getenv("WATCH_TIMEOUT_SECONDS", "6"))
        self.WATCH_CACHE_TTL_SECONDS = float(os.getenv("WATCH_CACHE_TTL_SECONDS", "21600"))

        self.ALLOW_REMOTE_POSTERS = _as_bool(os.getenv("ALLOW_REMOTE_POSTERS"), True)
        self.REMOTE_IMAGE_HOSTS = _as_list(
            os.getenv("REMOTE_IMAGE_HOSTS"), ("image.tmdb.org",)
        )
        self.AUTO_DOWNLOAD_POSTERS = _as_bool(os.getenv("AUTO_DOWNLOAD_POSTERS"), False)
        self.DOWNLOAD_TIMEOUT_SECONDS = float(
            os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "10")
        )
        self.POSTER_CACHE_TTL_SECONDS = float(
            os.getenv("POSTER_CACHE_TTL_SECONDS", "60")
        )
        self.MEDIA_CACHE_MAX_AGE = _as_int(os.getenv("MEDIA_CACHE_MAX_AGE"), 3600)
        # Browser-to-TMDB image requests can be blocked by some networks. Serving
        # validated images through our own origin avoids those client-side
        # failures; only allow-listed image hosts are ever fetched server-side.
        self.PROXY_REMOTE_IMAGES = _as_bool(os.getenv("PROXY_REMOTE_IMAGES"), True)
        self.REMOTE_IMAGE_CACHE_MAX_AGE = _as_int(
            os.getenv("REMOTE_IMAGE_CACHE_MAX_AGE"), 86400
        )

        self.REQUIRE_MODEL_AT_STARTUP = _as_bool(os.getenv("REQUIRE_MODEL_AT_STARTUP"), True)
        self.DEFAULT_PER_PAGE = _as_int(os.getenv("DEFAULT_PER_PAGE"), 24)
        self.MAX_PER_PAGE = _as_int(os.getenv("MAX_PER_PAGE"), 100)
        self.MAX_SEARCH_RESULTS = _as_int(os.getenv("MAX_SEARCH_RESULTS"), 50)

        # Anonymous personalisation: time-decayed repetition penalty plus a
        # small epsilon-greedy contextual bandit. All values are tunable and
        # deliberately conservative so the content model still leads ranking.
        self.REPETITION_PENALTY_HALF_LIFE_SECONDS = float(
            os.getenv("REPETITION_PENALTY_HALF_LIFE_SECONDS", "21600")
        )
        self.REPETITION_PENALTY_WINDOW_SECONDS = float(
            os.getenv("REPETITION_PENALTY_WINDOW_SECONDS", "86400")
        )
        self.REPETITION_PENALTY_STRENGTH = float(
            os.getenv("REPETITION_PENALTY_STRENGTH", "0.55")
        )
        self.RL_LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "0.25"))
        self.RL_EXPLORATION_RATE = float(os.getenv("RL_EXPLORATION_RATE", "0.08"))
        self.RL_TITLE_BONUS = float(os.getenv("RL_TITLE_BONUS", "0.45"))
        self.RL_CONTEXT_BONUS = float(os.getenv("RL_CONTEXT_BONUS", "0.25"))
        self.PERSONALIZATION_SESSION_MAX_AGE_SECONDS = float(
            os.getenv("PERSONALIZATION_SESSION_MAX_AGE_SECONDS", "604800")
        )
        self.PERSONALIZATION_MAX_SESSIONS = _as_int(
            os.getenv("PERSONALIZATION_MAX_SESSIONS"), 5000
        )

        self.JSON_SORT_KEYS = False
        self.SEND_FILE_MAX_AGE_DEFAULT = self.MEDIA_CACHE_MAX_AGE
        self.MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # API is read-only JSON

        for key, value in (overrides or {}).items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown configuration key: {key}")
            setattr(self, key, value)

    @classmethod
    def from_env(cls, overrides: Mapping[str, Any] | None = None) -> "Config":
        environment = os.getenv("FLASK_ENV", "development").strip().lower()
        if environment in {"production", "prod"}:
            return ProductionConfig(overrides)
        if environment in {"testing", "test"}:
            return TestingConfig(overrides)
        return Config(overrides)

    @staticmethod
    def _resolve_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def public_summary(self) -> dict[str, Any]:
        """Safe for responses: no filesystem paths, secrets or tokens."""
        return {
            "environment": self.ENV_NAME,
            "dataset_present": self.DATASET_PATH.is_file(),
            "local_poster_dirs": len(self.POSTER_DIRS),
            "remote_posters_enabled": self.ALLOW_REMOTE_POSTERS,
            "fallback_poster_enabled": bool(self.FALLBACK_POSTER_URL),
            "watch_provider_lookups_enabled": bool(self.TMDB_BEARER_TOKEN),
        }


class DevelopmentConfig(Config):
    ENV_NAME = "development"

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__({"DEBUG": True, **(overrides or {})})


class ProductionConfig(Config):
    ENV_NAME = "production"

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__(overrides)
        self.DEBUG = False
        self.REQUIRE_MODEL_AT_STARTUP = True
        if self.SECRET_KEY == "dev-only-secret-change-me":
            raise RuntimeError(
                "SECRET_KEY must be set to a random value in production"
            )
        if not self.DATASET_PATH.is_file():
            raise RuntimeError(
                f"Production dataset not found: {self.DATASET_PATH}. "
                "Set DATASET_PATH to the generated TMDB CSV."
            )
        if not self.MODEL_DIR.is_dir():
            raise RuntimeError(
                f"Production model directory not found: {self.MODEL_DIR}. "
                "Train locally or install the Kaggle export first."
            )


class TestingConfig(Config):
    ENV_NAME = "testing"

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            {
                "DEBUG": False,
                "TESTING": True,
                "ALLOW_REMOTE_POSTERS": False,
                "AUTO_DOWNLOAD_POSTERS": False,
                "REQUIRE_MODEL_AT_STARTUP": False,
                **(overrides or {}),
            }
        )
