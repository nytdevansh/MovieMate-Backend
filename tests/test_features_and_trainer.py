"""Artifact creation, sparse matrix shape and corruption detection."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from scipy import sparse

from ml.artifacts import ARTIFACT_FILES, ArtifactError, load_artifacts
from ml.config import FEATURE_FIELDS, ModelConfig
from ml.features import build_field_texts, vectorize_fields
from ml.inference import RecommendationEngine
from ml.trainer import train_model


def test_training_produces_valid_sparse_artifacts(tiny_model: Path):
    for name in ARTIFACT_FILES:
        assert (tiny_model / name).is_file(), name

    artifacts = load_artifacts(tiny_model)
    assert artifacts.config.feature_weights == ModelConfig().feature_weights
    assert sparse.issparse(artifacts.matrix) and artifacts.matrix.format == "csr"
    assert artifacts.matrix.shape[0] == len(artifacts.metadata) == 15
    assert artifacts.matrix.shape[1] == len(FEATURE_FIELDS) * len(artifacts.vectorizer.vocabulary_)
    assert artifacts.matrix.nnz < artifacts.matrix.shape[0] * artifacts.matrix.shape[1]
    assert set(artifacts.config.feature_weights) == set(FEATURE_FIELDS)

    config_json = json.loads((tiny_model / "model_config.json").read_text())
    assert config_json["feature_weights"]["genres"] == 3.0
    assert config_json["deduplicate_by_title_year"] is True
    info = json.loads((tiny_model / "model_info.json").read_text())
    assert info["n_movies"] == 15
    assert info["visual_features_used"] is False


def test_custom_weights_are_persisted_and_used(synthetic_csv: Path, tmp_path: Path):
    config = ModelConfig()
    config.feature_weights.update({"genres": 6.0, "overview": 0.5})
    train_model(synthetic_csv, tmp_path / "weighted", config)
    artifacts = load_artifacts(tmp_path / "weighted")
    assert artifacts.config.feature_weights["genres"] == 6.0

    record = artifacts.metadata[next(iter(artifacts.metadata))]
    vector = vectorize_fields([record], artifacts.vectorizer, artifacts.config)
    assert sparse.issparse(vector)
    assert vector.shape[0] == 1
    assert build_field_texts(record)["genres"]


def test_corrupt_bundle_is_rejected(tiny_model: Path, tmp_path: Path):
    missing = tmp_path / "missing"
    shutil.copytree(tiny_model, missing)
    (missing / "movie_matrix.npz").unlink()
    with pytest.raises(ArtifactError, match="Incomplete model bundle"):
        load_artifacts(missing)

    broken = tmp_path / "broken"
    shutil.copytree(tiny_model, broken)
    mapping = json.loads((broken / "movie_id_mapping.json").read_text())
    mapping["index_to_id"] = mapping["index_to_id"][:-1]
    (broken / "movie_id_mapping.json").write_text(json.dumps(mapping))
    with pytest.raises(ArtifactError, match="inconsistency"):
        load_artifacts(broken)


def test_engine_uses_loaded_artifacts(tiny_model: Path):
    engine = RecommendationEngine(load_artifacts(tiny_model))
    assert engine.has_movie(1)
    assert not engine.has_movie(999999)
    assert engine.get_metadata(1)["tmdb_id"] == 1
