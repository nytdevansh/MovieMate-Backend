"""Text normalisation and sparse TF-IDF feature construction.

Each of the six configured fields is transformed with ONE shared vectorizer
and then scaled by its configured weight. The stacked sparse matrix is
L2-normalised, so a dot product between two rows is cosine similarity.

Only a query-sized dense vector is ever created; no dense N x N similarity
matrix is materialised.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ml.config import FEATURE_FIELDS, ModelConfig
from ml.preprocessing import normalize_free_text, normalize_list_field


def build_field_texts(record: Mapping[str, Any]) -> dict[str, str]:
    """Build the six textual signals from one cleaned movie record."""
    return {
        "title": " ".join(
            part for part in (
                normalize_free_text(record.get("title")),
                normalize_free_text(record.get("original_title")),
            ) if part
        ),
        "genres": normalize_list_field(record.get("genres")),
        "overview": normalize_free_text(record.get("overview")),
        "cast": normalize_list_field(record.get("cast")),
        "director": normalize_list_field(record.get("director")),
        "original_language": normalize_free_text(record.get("original_language")),
    }


def combined_text(record: Mapping[str, Any]) -> str:
    """Corpus text used to fit the shared vocabulary and IDF weights."""
    fields = build_field_texts(record)
    return " ".join(fields[name] for name in FEATURE_FIELDS if fields.get(name))


def fit_vectorizer(texts: Iterable[str], config: ModelConfig) -> TfidfVectorizer:
    """Fit the shared TF-IDF vectorizer on combined movie texts."""
    vectorizer = TfidfVectorizer(**config.tfidf.to_sklearn_kwargs())
    vectorizer.fit(list(texts))
    if len(vectorizer.vocabulary_) == 0:
        raise ValueError(
            "TF-IDF vocabulary is empty. Check that the dataset contains "
            "non-empty title/genre/overview text."
        )
    return vectorizer


def vectorize_fields(
    records: Sequence[Mapping[str, Any]],
    vectorizer: TfidfVectorizer,
    config: ModelConfig,
) -> sparse.csr_matrix:
    """Return the weighted, L2-normalised sparse feature matrix for records."""
    if not records:
        raise ValueError("Cannot vectorize an empty record list")
    field_texts = [build_field_texts(record) for record in records]
    blocks: list[sparse.csr_matrix] = []
    for name in FEATURE_FIELDS:
        texts = [fields[name] for fields in field_texts]
        block = vectorizer.transform(texts).tocsr()
        weight = float(config.feature_weights[name])
        if weight != 1.0:
            block = block * weight
        blocks.append(block)
    matrix = sparse.hstack(blocks, format="csr", dtype=np.float32)
    normalize(matrix, norm="l2", copy=False)
    return matrix


def cosine_scores(query_vector: sparse.csr_matrix, matrix: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity of one normalised query against all normalised rows.

    The result is a dense vector of length N (one score per movie), not an
    N x N matrix. Sparse-by-sparse multiplication keeps this memory-efficient.
    """
    if query_vector.shape[0] != 1:
        raise ValueError("query_vector must contain exactly one row")
    if query_vector.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: query={query_vector.shape[1]}, "
            f"matrix={matrix.shape[1]}"
        )
    return np.asarray((matrix @ query_vector.T).todense()).ravel()
