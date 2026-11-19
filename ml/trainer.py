"""Single training entry point used by both local and Kaggle scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ml.artifacts import export_zip, save_artifacts
from ml.config import ARTIFACT_SCHEMA_VERSION, ModelConfig
from ml.evaluate import dataset_profile
from ml.features import combined_text, fit_vectorizer, vectorize_fields
from ml.preprocessing import clean_dataset
from ml.preprocessing import DatasetSchemaError, DataProcessingError


class TrainingError(RuntimeError):
    """Raised with context when training cannot complete."""


@dataclass
class TrainingResult:
    model_dir: Path
    info: dict[str, Any]
    artifact_sizes: dict[str, int]
    zip_path: Path | None

    @property
    def n_movies(self) -> int:
        return int(self.info["n_movies"])


def train_model(
    csv_path: str | Path,
    model_dir: str | Path,
    config: ModelConfig | None = None,
    zip_path: str | Path | None = None,
    extra_info: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Load CSV -> clean -> TF-IDF -> weighted sparse matrix -> artifacts.

    The same function is called by training/train_local.py and
    training/train_kaggle.py so both produce byte-compatible bundles.
    """
    config = config or ModelConfig()
    config.validate()
    source = Path(csv_path)
    if not source.is_file():
        raise TrainingError(f"Dataset CSV not found: {source}")

    started = time.perf_counter()
    try:
        raw = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        raise TrainingError(f"Could not read dataset CSV {source}: {exc}") from exc

    try:
        cleaned = clean_dataset(raw, config)
    except (DatasetSchemaError, DataProcessingError) as exc:
        raise TrainingError(f"Dataset validation failed for {source}: {exc}") from exc

    records = cleaned.records
    if len(records) < 2:
        raise TrainingError("At least two valid movies are needed to train a recommender")

    try:
        vectorizer = fit_vectorizer((combined_text(record) for record in records), config)
        matrix = vectorize_fields(records, vectorizer, config)
    except Exception as exc:
        raise TrainingError(f"TF-IDF feature extraction failed: {exc}") from exc

    model_keys = [str(record.get("model_key") or record["tmdb_id"]) for record in records]
    metadata = {key: record for key, record in zip(model_keys, records)}
    id_mapping = {
        "id_column": "content_id",
        "index_to_id": model_keys,
        "id_to_index": {key: index for index, key in enumerate(model_keys)},
    }
    n_series = sum(str(record.get("media_type")) == "tv" for record in records)

    training_seconds = round(time.perf_counter() - started, 3)
    profile = dataset_profile(raw, cleaned.report)
    info: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": "content_based_tfidf_cosine",
        "n_movies": len(records),
        "n_titles": len(records),
        "n_series": n_series,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "feature_dimensions": int(matrix.shape[1]),
        "matrix_nnz": int(matrix.nnz),
        "matrix_density_percent": round(
            100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 5
        ),
        "training_time_seconds": training_seconds,
        "feature_weights": dict(config.feature_weights),
        "dataset_filename": source.name,
        "dataset_profile": profile,
        "libraries": {
            "python": _python_version(),
            "pandas": pd.__version__,
            "scikit_learn": _sklearn_version(),
            "scipy": _scipy_version(),
        },
        "evaluation_note": (
            "No relevance labels or interaction logs are available. Precision@K, "
            "Recall@K and NDCG@K are not measurable; use manual inspection instead."
        ),
        "visual_features_used": False,
    }
    if extra_info:
        info.update(dict(extra_info))

    directory = Path(model_dir)
    # First pass writes all six files, then real on-disk sizes are measured and
    # written into model_info.json with a second pass.
    save_artifacts(directory, vectorizer, matrix, metadata, id_mapping, config, info)
    sizes = {
        path.name: path.stat().st_size
        for path in directory.iterdir() if path.is_file()
    }
    info["artifact_sizes_bytes"] = sizes
    info["total_artifact_bytes"] = sum(sizes.values())
    save_artifacts(directory, vectorizer, matrix, metadata, id_mapping, config, info)

    zip_result = export_zip(directory, zip_path) if zip_path else None
    return TrainingResult(
        model_dir=directory, info=info, artifact_sizes=sizes, zip_path=zip_result
    )


def _python_version() -> str:
    import platform

    return platform.python_version()


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def _scipy_version() -> str:
    import scipy

    return scipy.__version__
