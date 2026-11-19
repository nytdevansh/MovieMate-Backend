"""Dataset profiling and manual recommendation inspection.

There are no relevance labels or interaction logs in the TMDB dataset, so
Precision@K / Recall@K / NDCG@K cannot be measured as real-world quality
metrics here. This module produces descriptive statistics and concrete
recommendation examples for human inspection instead of invented accuracy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ml.artifacts import load_artifacts
from ml.inference import RecommendationEngine
from ml.preprocessing import CleaningReport

PROFILE_FIELDS: tuple[str, ...] = (
    "title", "original_title", "release_date", "year", "original_language",
    "genres", "overview", "cast", "director", "vote_average", "vote_count",
    "popularity", "poster_url", "poster_local_path", "india_streaming",
)


def dataset_profile(raw: pd.DataFrame, report: CleaningReport) -> dict[str, Any]:
    """Compute missing-data percentages and duplicate statistics from real data."""
    total = int(len(raw))
    missing: dict[str, dict[str, float | int]] = {}
    for column in PROFILE_FIELDS:
        if column not in raw.columns:
            continue
        series = raw[column]
        count = int(series.isna().sum())
        if series.dtype == object:
            count = int(
                series.map(
                    lambda value: pd.isna(value)
                    or (isinstance(value, str) and not value.strip())
                ).sum()
            )
        missing[column] = {
            "missing_count": count,
            "missing_percent": round(100.0 * count / total, 2) if total else 0.0,
        }
    return {
        "rows_in": total,
        "rows_out": report.rows_out,
        "duplicate_tmdb_ids_dropped": report.duplicate_ids_dropped,
        "invalid_ids_dropped": report.invalid_ids_dropped,
        "missing_by_column": missing,
        "missing_poster_urls": {
            "count": report.missing_posters,
            "percent": round(100.0 * report.missing_posters / report.rows_out, 2)
            if report.rows_out else 0.0,
        },
        "cleaning_report": report.to_dict(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def choose_examples(engine: RecommendationEngine, count: int = 5) -> list[str]:
    """Deterministically pick well-described titles for manual inspection."""
    def richness(record: Mapping[str, Any]) -> tuple[int, int]:
        filled = sum(
            1 for key in ("title", "genres", "overview", "cast", "director", "original_language")
            if record.get(key)
        )
        return filled, int(record.get("vote_count") or 0)

    ranked = sorted(
        engine.metadata.items(),
        key=lambda item: (-richness(item[1])[0], -richness(item[1])[1], str(item[0])),
    )
    return [str(key) for key, _ in ranked[:count]]


def inspect_examples(
    engine: RecommendationEngine,
    content_ids: Sequence[int | str],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, dict[str, Any]]:
    """Produce top-5/top-10 examples for each selected title."""
    results: dict[str, dict[str, Any]] = {}
    for content_id in content_ids:
        source = engine.get_metadata(content_id)
        key = engine.model_key(content_id)
        entry: dict[str, Any] = {
            "content_id": key,
            "title": source.get("title"),
            "genres": source.get("genres"),
        }
        for k in k_values:
            entry[f"top_{k}"] = [
                {
                    "content_id": rec.content_id,
                    "title": rec.title,
                    "genres": rec.genres,
                    "release_year": rec.release_year,
                    "rating": rec.rating,
                    "similarity_score": rec.similarity_score,
                }
                for rec in engine.recommend(content_id, limit=k)
            ]
        results[key] = entry
    return results


def write_report(
    model_dir: str | Path,
    output_path: str | Path,
    sample_ids: Sequence[int | str] | None = None,
    examples_per_k: int = 5,
) -> Path:
    """Write a Markdown evaluation/inspection report from trained artifacts."""
    artifacts = load_artifacts(model_dir)
    engine = RecommendationEngine(artifacts)
    info = artifacts.info
    profile = info.get("dataset_profile") or {}
    ids = list(sample_ids) if sample_ids else choose_examples(engine, examples_per_k)
    examples = inspect_examples(engine, ids)

    lines: list[str] = [
        "# Model evaluation and inspection report",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "## What this report does and does not measure",
        "",
        "The source dataset has no relevance labels, user accounts, likes or watch history. "
        "Therefore **Precision@K, Recall@K and NDCG@K cannot be meaningfully established** "
        "as real-world recommendation quality metrics. Any such number would be fabricated. "
        "The results below are descriptive statistics plus manual inspection examples.",
        "",
        "## Dataset",
        "",
        f"- Rows read: {profile.get('rows_in', info.get('n_titles', info.get('n_movies', 'n/a')))}",
        f"- Titles used after ID validation and de-duplication: "
        f"{info.get('n_titles', info.get('n_movies', profile.get('rows_out', 'n/a')))}",
        f"- Titles: {info.get('n_titles', info.get('n_movies', 'n/a'))}; series among them: {info.get('n_series', 0)}",
        f"- Duplicate tmdb_id rows dropped: "
        f"{profile.get('duplicate_tmdb_ids_dropped', 'n/a')}",
        f"- Invalid tmdb_id rows dropped: {profile.get('invalid_ids_dropped', 'n/a')}",
        f"- Missing overview: "
        f"{_pct(profile, 'overview')}",
        f"- Missing genres: {_pct(profile, 'genres')}",
        f"- Missing cast: {_pct(profile, 'cast')}",
        f"- Missing director: {_pct(profile, 'director')}",
        f"- Missing poster URL: "
        f"{(profile.get('missing_poster_urls') or {}).get('percent', 'n/a')}%",
        f"- Missing India streaming providers: {_pct(profile, 'india_streaming')}",
        "",
        "## Model",
        "",
        f"- Vocabulary size: {info.get('vocabulary_size', 'n/a')}",
        f"- Feature dimensions: {info.get('feature_dimensions', 'n/a')}",
        f"- Non-zero entries in sparse matrix: {info.get('matrix_nnz', 'n/a')}",
        f"- Matrix density: {info.get('matrix_density_percent', 'n/a')}%",
        f"- Training time: {info.get('training_time_seconds', 'n/a')} seconds",
        f"- Total artifact size: {info.get('total_artifact_bytes', 'n/a')} bytes",
        f"- Feature weights: {info.get('feature_weights', {})}",
        "",
        "Poster images are **not** used as ML features in this baseline. Only the six "
        "textual metadata fields listed in the model config are vectorised. The architecture "
        "keeps poster lookup separate so visual embeddings can be added and evaluated later.",
        "",
        "## Manual inspection: top-5 and top-10 recommendations",
        "",
    ]
    for content_id, entry in examples.items():
        lines.extend([
            f"### {entry['title']} (content_id={content_id})",
            "",
            f"Source genres: {entry['genres'] or 'none'}",
            "",
        ])
        for key in ("top_5", "top_10"):
            lines.append(f"**{key.replace('_', ' ').upper()}**")
            lines.append("")
            lines.append("| # | Title | Genres | Year | Rating | Cosine |")
            lines.append("|---|-------|--------|------|--------|--------|")
            for position, rec in enumerate(entry[key], start=1):
                lines.append(
                    f"| {position} | {_escape(rec['title'])} | "
                    f"{_escape(', '.join(rec['genres'][:3]))} | "
                    f"{rec['release_year'] or '-'} | {rec['rating'] if rec['rating'] is not None else '-'} | "
                    f"{rec['similarity_score']:.4f} |"
                )
            lines.append("")
    lines.extend([
        "## How to turn this into a measured evaluation",
        "",
        "Collect explicit feedback (likes/dislikes, watch history, watchlist additions) "
        "with user IDs, then build a time-split holdout and compute Precision@K, "
        "Recall@K and NDCG@K against those labels. Until then, use the examples above "
        "to check whether genres, cast, director and overview produce plausible neighbours.",
        "",
    ])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _pct(profile: Mapping[str, Any], column: str) -> str:
    entry = (profile.get("missing_by_column") or {}).get(column) or {}
    return f"{entry.get('missing_percent', 'n/a')}%"


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")
