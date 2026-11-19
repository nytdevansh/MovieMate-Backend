"""Configuration shared by training and inference.

Feature weights live here (and in model_config.json), never in route code.
Changing a weight requires retraining; the effective weights are always
recorded in the artifact so inference uses exactly the training recipe.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Ordered list of the six textual signals used by the baseline model.
FEATURE_FIELDS: tuple[str, ...] = (
    "title",
    "genres",
    "overview",
    "cast",
    "director",
    "original_language",
)

DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "genres": 3.0,
    "title": 2.0,
    "cast": 2.0,
    "director": 2.0,
    "original_language": 2.0,
    "overview": 1.0,
}

ARTIFACT_FILES: tuple[str, ...] = (
    "tfidf_vectorizer.joblib",
    "movie_matrix.npz",
    "movie_metadata.json",
    "movie_id_mapping.json",
    "model_config.json",
    "model_info.json",
)

ARTIFACT_SCHEMA_VERSION = 1


class ModelConfigError(ValueError):
    """Raised when a model configuration is invalid."""


@dataclass(frozen=True)
class TfidfSettings:
    """Settings forwarded verbatim to sklearn.feature_extraction.text.TfidfVectorizer."""

    lowercase: bool = True
    strip_accents: str | None = "unicode"
    stop_words: str | None = None
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    max_features: int | None = 200_000
    sublinear_tf: bool = True
    norm: str | None = None  # final weighted matrix is normalised instead

    def to_sklearn_kwargs(self) -> dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "strip_accents": self.strip_accents,
            "stop_words": self.stop_words,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "norm": self.norm,
        }

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "lowercase": self.lowercase,
            "strip_accents": self.strip_accents,
            "stop_words": self.stop_words,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "norm": self.norm,
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TfidfSettings":
        ngram = data.get("ngram_range", [1, 2])
        if not isinstance(ngram, (list, tuple)) or len(ngram) != 2:
            raise ModelConfigError("tfidf.ngram_range must contain exactly two integers")
        return cls(
            lowercase=bool(data.get("lowercase", True)),
            strip_accents=data.get("strip_accents", "unicode"),
            stop_words=data.get("stop_words"),
            ngram_range=(int(ngram[0]), int(ngram[1])),
            min_df=int(data.get("min_df", 2)),
            max_df=float(data.get("max_df", 0.95)),
            max_features=(
                int(data["max_features"]) if data.get("max_features") is not None else None
            ),
            sublinear_tf=bool(data.get("sublinear_tf", True)),
            norm=data.get("norm"),
        )


@dataclass
class ModelConfig:
    feature_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS)
    )
    tfidf: TfidfSettings = field(default_factory=TfidfSettings)
    id_column: str = "tmdb_id"
    default_top_k: int = 10
    max_top_k: int = 50
    deduplicate_by_title_year: bool = True
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def validate(self) -> None:
        unknown = set(self.feature_weights) - set(FEATURE_FIELDS)
        missing = set(FEATURE_FIELDS) - set(self.feature_weights)
        if unknown or missing:
            raise ModelConfigError(
                "feature_weights must define exactly these fields: "
                f"{list(FEATURE_FIELDS)}; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for name, weight in self.feature_weights.items():
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ModelConfigError(f"feature weight for {name!r} must be a number")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ModelConfigError(f"feature weight for {name!r} must be positive and finite")
        if self.default_top_k < 1 or self.max_top_k < self.default_top_k:
            raise ModelConfigError("require 1 <= default_top_k <= max_top_k")
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ModelConfigError(
                f"unsupported artifact schema version {self.schema_version!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        ngram = self.tfidf.ngram_range
        if ngram[0] < 1 or ngram[1] < ngram[0]:
            raise ModelConfigError("tfidf.ngram_range must satisfy 1 <= low <= high")
        if self.tfidf.min_df < 1:
            raise ModelConfigError("tfidf.min_df must be >= 1")
        if not 0 < self.tfidf.max_df <= 1:
            raise ModelConfigError("tfidf.max_df must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_fields": list(FEATURE_FIELDS),
            "feature_weights": dict(self.feature_weights),
            "tfidf": self.tfidf.to_dict(),
            "id_column": self.id_column,
            "default_top_k": self.default_top_k,
            "max_top_k": self.max_top_k,
            "deduplicate_by_title_year": self.deduplicate_by_title_year,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        config = cls(
            feature_weights={
                key: float(value)
                for key, value in dict(data.get("feature_weights", DEFAULT_FEATURE_WEIGHTS)).items()
            },
            tfidf=TfidfSettings.from_dict(dict(data.get("tfidf") or {})),
            id_column=str(data.get("id_column", "tmdb_id")),
            default_top_k=int(data.get("default_top_k", 10)),
            max_top_k=int(data.get("max_top_k", 50)),
            deduplicate_by_title_year=bool(data.get("deduplicate_by_title_year", True)),
            schema_version=int(data.get("schema_version", ARTIFACT_SCHEMA_VERSION)),
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
