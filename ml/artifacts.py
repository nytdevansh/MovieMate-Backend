"""Serialisation, validation and ZIP export of model artifacts."""
from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.config import ARTIFACT_FILES, ARTIFACT_SCHEMA_VERSION, ModelConfig, ModelConfigError
from ml.ids import parse_content_id

MODEL_ZIP_NAME = "movie_recommender_model.zip"


class ArtifactError(RuntimeError):
    """Raised when a model artifact bundle is missing, corrupt or inconsistent."""


@dataclass
class LoadedArtifacts:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    metadata: dict[str, dict[str, Any]]
    id_mapping: dict[str, Any]
    config: ModelConfig
    info: dict[str, Any]
    _model_dir: str = ""

    @property
    def model_dir(self) -> Path:
        return Path(self._model_dir)

    def set_model_dir(self, path: str | Path) -> None:
        self._model_dir = str(path)


def save_artifacts(
    output_dir: str | Path,
    vectorizer: TfidfVectorizer,
    matrix: sparse.csr_matrix,
    metadata: dict[str, dict[str, Any]],
    id_mapping: dict[str, Any],
    config: ModelConfig,
    info: Mapping[str, Any],
) -> dict[str, int]:
    """Write the six canonical artifacts and return their sizes in bytes."""
    import joblib  # imported lazily so importing ml never requires joblib at runtime

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, directory / "tfidf_vectorizer.joblib", compress=3)
    sparse.save_npz(directory / "movie_matrix.npz", matrix.tocsr(), compressed=True)
    _write_json(directory / "movie_metadata.json", metadata)
    _write_json(directory / "movie_id_mapping.json", id_mapping)
    config.save(directory / "model_config.json")
    _write_json(directory / "model_info.json", dict(info))
    return {name: (directory / name).stat().st_size for name in ARTIFACT_FILES}


def load_artifacts(model_dir: str | Path) -> LoadedArtifacts:
    """Load and validate a model bundle before any request can use it."""
    import joblib

    directory = Path(model_dir)
    if not directory.is_dir():
        raise ArtifactError(f"Model directory does not exist: {directory}")
    missing = [name for name in ARTIFACT_FILES if not (directory / name).is_file()]
    if missing:
        raise ArtifactError(
            f"Incomplete model bundle in {directory}. Missing: {', '.join(missing)}. "
            "Train a model or install the Kaggle export first."
        )

    try:
        config = ModelConfig.from_file(directory / "model_config.json")
        metadata = _read_json(directory / "movie_metadata.json")
        id_mapping = _read_json(directory / "movie_id_mapping.json")
        info = _read_json(directory / "model_info.json")
        vectorizer = joblib.load(directory / "tfidf_vectorizer.joblib")
        matrix = sparse.load_npz(directory / "movie_matrix.npz")
    except ModelConfigError:
        raise
    except Exception as exc:  # joblib/scipy/json can raise many exception types
        raise ArtifactError(f"Could not read model artifacts in {directory}: {exc}") from exc

    if not isinstance(metadata, dict) or not metadata:
        raise ArtifactError("movie_metadata.json must be a non-empty object keyed by canonical model ID")
    if not isinstance(info, dict):
        raise ArtifactError("model_info.json must contain a JSON object")
    if not isinstance(vectorizer, TfidfVectorizer):
        raise ArtifactError("tfidf_vectorizer.joblib does not contain a TfidfVectorizer")
    if not sparse.issparse(matrix) or matrix.ndim != 2:
        raise ArtifactError("movie_matrix.npz must contain a 2-D sparse matrix")
    matrix = matrix.tocsr()
    if not np.isfinite(matrix.data).all():
        raise ArtifactError("movie_matrix.npz contains non-finite values")

    index_to_id = id_mapping.get("index_to_id") if isinstance(id_mapping, dict) else None
    id_to_index = id_mapping.get("id_to_index") if isinstance(id_mapping, dict) else None
    if not isinstance(index_to_id, list) or not isinstance(id_to_index, dict):
        raise ArtifactError(
            "movie_id_mapping.json must define 'index_to_id' (list) and 'id_to_index' (object)"
        )
    if len(index_to_id) != len(metadata) or len(id_to_index) != len(metadata):
        raise ArtifactError(
            "Artifact inconsistency: matrix mapping length "
            f"({len(index_to_id)}/{len(id_to_index)}) != metadata count ({len(metadata)})"
        )
    if matrix.shape[0] != len(metadata):
        raise ArtifactError(
            f"Artifact inconsistency: matrix has {matrix.shape[0]} rows but "
            f"metadata has {len(metadata)} titles"
        )
    vocabulary_size = len(vectorizer.vocabulary_)
    from ml.config import FEATURE_FIELDS

    expected_columns = vocabulary_size * len(FEATURE_FIELDS)
    if matrix.shape[1] != expected_columns:
        raise ArtifactError(
            f"Artifact inconsistency: matrix has {matrix.shape[1]} columns but "
            f"{len(FEATURE_FIELDS)} weighted field blocks over a {vocabulary_size}-term "
            f"vocabulary require {expected_columns} columns"
        )
    for key, index in id_to_index.items():
        key = str(key)
        if key not in metadata or str(index_to_id[int(index)]) != key:
            raise ArtifactError(f"Invalid movie ID mapping for {key!r}")
    for model_key, record in metadata.items():
        parsed = parse_content_id(model_key)
        if not isinstance(record, dict) or parsed is None or parsed.model_key != str(model_key):
            raise ArtifactError(f"Invalid metadata record for title {model_key!r}")
        if str(record.get("tmdb_id")) != str(parsed.source_id):
            raise ArtifactError(f"Metadata TMDB ID does not match model key {model_key!r}")

    artifacts = LoadedArtifacts(
        vectorizer=vectorizer,
        matrix=matrix,
        metadata=metadata,
        id_mapping=id_mapping,
        config=config,
        info=info,
    )
    artifacts.set_model_dir(directory)
    return artifacts


def export_zip(model_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    """Bundle the six artifacts into a downloadable ZIP (files at ZIP root)."""
    directory = Path(model_dir)
    target = Path(zip_path) if zip_path else directory.parent / MODEL_ZIP_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ARTIFACT_FILES:
            archive.write(directory / name, arcname=name)
    return target


def extract_model_zip(zip_path: str | Path, target_dir: str | Path) -> Path:
    """Safely extract an exported model ZIP into target_dir.

    Both layouts are accepted: six files at the ZIP root, or inside a
    'movie_model/' folder. Path traversal and absolute paths are rejected.
    """
    archive_path = Path(zip_path)
    target = Path(target_dir)
    if not archive_path.is_file():
        raise ArtifactError(f"Model ZIP not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            path = Path(member)
            if path.is_absolute() or ".." in path.parts:
                raise ArtifactError(f"Unsafe path in model ZIP: {member!r}")
        names = archive.namelist()
        prefix = "movie_model/" if any(n.startswith("movie_model/") for n in names) else ""
        if not all(f"{prefix}{name}" in names for name in ARTIFACT_FILES):
            raise ArtifactError(
                "Model ZIP must contain the six canonical artifacts "
                f"({', '.join(ARTIFACT_FILES)}), optionally under movie_model/"
            )
        target.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_FILES:
            with archive.open(f"{prefix}{name}") as source:
                with (target / name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    return target


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
