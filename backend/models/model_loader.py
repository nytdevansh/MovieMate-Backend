"""Load and validate the trained model exactly once at app startup."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ml.artifacts import ArtifactError, load_artifacts
from ml.inference import RecommendationEngine

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    engine: RecommendationEngine | None
    info: dict[str, Any]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.engine is not None


def load_model_state(config: Any) -> ModelState:
    """Validate artifacts; fail fast in strict mode, degrade gracefully otherwise."""
    try:
        artifacts = load_artifacts(config.MODEL_DIR)
        engine = RecommendationEngine(artifacts)
        logger.info(
            "Loaded model: %s movies, %s features",
            artifacts.info.get("n_movies"),
            artifacts.info.get("feature_dimensions"),
        )
        return ModelState(engine=engine, info=artifacts.info)
    except ArtifactError as exc:
        if config.REQUIRE_MODEL_AT_STARTUP:
            raise RuntimeError(
                f"Model validation failed at startup ({config.MODEL_DIR}): {exc}"
            ) from exc
        logger.warning("Model unavailable; recommendation endpoints will return 503: %s", exc)
        return ModelState(engine=None, info={}, error=str(exc))
