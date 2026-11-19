"""Schema validation and deterministic cleaning for TMDB movie datasets.

Rules of this module:
  * never invent a value (missing metadata stays missing -> None / "");
  * tmdb_id is the primary key; invalid IDs and duplicate IDs are dropped with
    an explicit count, keeping the most complete record;
  * all transformations are deterministic so local and Kaggle training agree.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from ml.config import ModelConfig
from ml.ids import ContentId, make_content_id, parse_content_id

EXPECTED_COLUMNS: tuple[str, ...] = (
    "tmdb_id", "imdb_id", "title", "original_title", "release_date", "year",
    "original_language", "production_countries", "genres", "overview",
    "cast", "director", "runtime_minutes", "vote_average", "vote_count",
    "popularity", "poster_path", "poster_url", "poster_local_path",
    "backdrop_path", "backdrop_url", "india_streaming", "india_rent",
    "india_buy", "homepage", "adult",
)

# Pipe-delimited fields produced by the dataset builder.
LIST_COLUMNS: tuple[str, ...] = (
    "genres", "cast", "director", "production_countries",
    "india_streaming", "india_rent", "india_buy",
)
FREE_TEXT_COLUMNS: tuple[str, ...] = (
    "title", "original_title", "overview", "imdb_id", "release_date",
    "poster_path", "poster_url", "poster_local_path", "backdrop_path",
    "backdrop_url", "homepage",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "vote_average", "vote_count", "popularity", "runtime_minutes",
)
# Optional columns added for the combined movies + TV catalogue. Movie-only
# CSVs remain valid; these fields are derived rather than required.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "content_id", "media_type", "source_tmdb_id", "model_key", "poster_key",
    "number_of_seasons", "number_of_episodes", "status", "tv_type",
)

_MISSING_TOKENS = frozenset({"", "nan", "none", "null", "n/a", "na"})
_WHITESPACE = re.compile(r"\s+")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


class DatasetSchemaError(ValueError):
    """Raised when the CSV does not have the expected TMDB columns."""


class DataProcessingError(ValueError):
    """Raised when a dataset cannot be cleaned meaningfully."""


def validate_schema(frame: pd.DataFrame, required: Sequence[str] = EXPECTED_COLUMNS) -> None:
    """Check that all required columns are present. Extra columns are allowed."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetSchemaError(
            "Dataset is missing required column(s): "
            + ", ".join(missing)
            + ". Expected TMDB schema: "
            + ", ".join(required)
        )


def is_missing(value: Any) -> bool:
    """True for None, pandas NaN/NaT and empty/placeholder strings."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, bool) and result:
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TOKENS
    return False


def normalize_free_text(value: Any) -> str:
    """Unicode-normalise free text, collapse whitespace and strip. No invention."""
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return _WHITESPACE.sub(" ", text).strip()


def normalize_list_field(value: Any) -> str:
    """Normalise a pipe-delimited list: trim, drop empties, dedupe, keep order."""
    if is_missing(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts: Iterable[Any] = value
    else:
        parts = str(value).split("|")
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        token = normalize_free_text(part)
        if token and token.lower() not in seen:
            seen.add(token.lower())
            result.append(token)
    return " | ".join(result)


def split_list(value: Any) -> list[str]:
    """Split a pipe-delimited field into a clean, JSON-friendly list."""
    if is_missing(value):
        return []
    parts = value if isinstance(value, (list, tuple, set)) else str(value).split("|")
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = normalize_free_text(part)
        if token and token.lower() not in seen:
            seen.add(token.lower())
            result.append(token)
    return result


def parse_number(
    value: Any, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    """Parse a finite number in range; invalid values become None (never clamped)."""
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def parse_year(value: Any, release_date: str = "") -> int | None:
    """Return a plausible four-digit year from the year column or release date."""
    number = parse_number(value, minimum=1900, maximum=2100)
    if number is not None:
        return int(number)
    match = _DATE_RE.match(release_date.strip())
    if match:
        parsed = parse_number(match.group(1), minimum=1900, maximum=2100)
        if parsed is not None:
            return int(parsed)
    return None


def valid_http_url(value: Any, allowed_hosts: Iterable[str] | None = None) -> str:
    """Return the URL when it is a syntactically valid http(s) URL, else ''."""
    text = normalize_free_text(value)
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    if allowed_hosts is not None:
        hosts = {host.lower().strip() for host in allowed_hosts if host.strip()}
        if hosts and parsed.hostname and parsed.hostname.lower() not in hosts:
            return ""
    return text


@dataclass
class CleaningReport:
    """Counts recorded (never silently inferred) during cleaning."""

    rows_in: int = 0
    rows_out: int = 0
    invalid_ids_dropped: int = 0
    duplicate_ids_dropped: int = 0
    invalid_release_dates: int = 0
    invalid_years: int = 0
    invalid_vote_average: int = 0
    invalid_vote_count: int = 0
    invalid_popularity: int = 0
    invalid_runtime: int = 0
    invalid_poster_urls: int = 0
    empty_overviews: int = 0
    empty_titles: int = 0
    empty_genres: int = 0
    empty_cast: int = 0
    empty_directors: int = 0
    missing_posters: int = 0

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in vars(self).items()}


@dataclass
class CleanedDataset:
    frame: pd.DataFrame
    records: list[dict[str, Any]] = field(default_factory=list)
    report: CleaningReport = field(default_factory=CleaningReport)


def _completeness(row: pd.Series) -> int:
    """How much usable metadata a row has (used only to choose a duplicate winner)."""
    score = 0
    for column in EXPECTED_COLUMNS:
        if column in row and not is_missing(row[column]):
            score += 1
    return score


def _json_safe(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def clean_dataset(raw: pd.DataFrame, config: ModelConfig | None = None) -> CleanedDataset:
    """Validate, de-duplicate and normalise a raw TMDB dataframe.

    Duplicate IDs keep the most complete row, with a deterministic tie-break on
    vote_count and title. Rows without a usable tmdb_id are dropped.
    """
    config = config or ModelConfig()
    validate_schema(raw)
    report = CleaningReport(rows_in=int(len(raw)))
    frame = raw.copy()

    id_values = pd.to_numeric(frame[config.id_column], errors="coerce")
    valid_id = id_values.notna() & (id_values % 1 == 0)
    report.invalid_ids_dropped = int((~valid_id).sum())
    frame = frame.loc[valid_id].copy()
    frame[config.id_column] = id_values.loc[valid_id].astype("int64")

    if frame.empty:
        raise DataProcessingError("No rows with a valid tmdb_id were found in the dataset")

    # Identity columns are derived deterministically before de-duplication so a
    # movie TMDB ID can never displace a TV series with the same numeric ID.
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    identities: list[ContentId] = []
    for row in frame.to_dict(orient="records"):
        media_type = normalize_free_text(row.get("media_type")).lower()
        if media_type not in {"movie", "tv"}:
            parsed = parse_content_id(row.get("content_id"))
            media_type = parsed.media_type if parsed else "movie"
        try:
            source_id = int(row["tmdb_id"])
        except (TypeError, ValueError):
            source_id = 0
        identities.append(make_content_id(media_type, source_id))
    frame["content_id"] = [item.public for item in identities]
    frame["media_type"] = [item.media_type for item in identities]
    frame["source_tmdb_id"] = [item.source_id for item in identities]
    frame["model_key"] = [item.model_key for item in identities]
    frame["poster_key"] = [item.poster_key for item in identities]

    frame["_completeness"] = frame.apply(_completeness, axis=1)
    frame["_vote_count_sort"] = pd.to_numeric(
        frame.get("vote_count"), errors="coerce"
    ).fillna(-1.0)
    frame["_title_sort"] = frame.get("title", "").astype(str)
    frame = frame.sort_values(
        ["content_id", "_completeness", "_vote_count_sort", "_title_sort"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    report.duplicate_ids_dropped = int(frame.duplicated(subset=["content_id"]).sum())
    frame = frame.drop_duplicates(subset=["content_id"], keep="first").copy()

    for column in LIST_COLUMNS:
        frame[column] = frame[column].map(normalize_list_field)
    for column in FREE_TEXT_COLUMNS:
        frame[column] = frame[column].map(normalize_free_text)

    frame[config.id_column] = frame[config.id_column].astype("int64")
    release_dates = frame["release_date"].fillna("").astype(str).str.strip()
    parsed_dates = pd.to_datetime(release_dates, format="%Y-%m-%d", errors="coerce")
    report.invalid_release_dates = int(parsed_dates.isna().sum())
    frame["release_date"] = release_dates.where(parsed_dates.notna(), "")

    years: list[int | None] = []
    for value, date_value in zip(frame.get("year", pd.Series(dtype=object)), release_dates):
        year = parse_year(value, date_value)
        years.append(year)
    report.invalid_years = sum(year is None for year in years)
    frame["year"] = years

    for column, report_key, minimum, maximum in (
        ("vote_average", "invalid_vote_average", 0.0, 10.0),
        ("vote_count", "invalid_vote_count", 0.0, None),
        ("popularity", "invalid_popularity", 0.0, None),
        ("runtime_minutes", "invalid_runtime", 0.0, None),
    ):
        parsed = [
            parse_number(value, minimum=minimum, maximum=maximum)
            for value in frame[column]
        ]
        setattr(report, report_key, sum(value is None for value in parsed))
        frame[column] = parsed

    # Only TMDB image URLs are retained as remote fallbacks.
    poster_urls = [
        valid_http_url(value, {"image.tmdb.org"}) for value in frame["poster_url"]
    ]
    backdrop_urls = [
        valid_http_url(value, {"image.tmdb.org"}) for value in frame["backdrop_url"]
    ]
    report.invalid_poster_urls = sum(not url for url in poster_urls)
    frame["poster_url"] = poster_urls
    frame["backdrop_url"] = backdrop_urls
    frame["homepage"] = [
        valid_http_url(value) for value in frame["homepage"]
    ]

    frame["adult"] = frame["adult"].map(_to_bool)
    for column in ("number_of_seasons", "number_of_episodes"):
        parsed = [parse_number(value, minimum=0) for value in frame[column]]
        frame[column] = [int(value) if value is not None else None for value in parsed]
    for column in ("status", "tv_type"):
        frame[column] = frame[column].map(normalize_free_text)
    frame = frame.drop(columns=["_completeness", "_vote_count_sort", "_title_sort"])
    frame = frame.reset_index(drop=True)

    report.rows_out = int(len(frame))
    report.empty_overviews = int((frame["overview"] == "").sum())
    report.empty_titles = int((frame["title"] == "").sum())
    report.empty_genres = int((frame["genres"] == "").sum())
    report.empty_cast = int((frame["cast"] == "").sum())
    report.empty_directors = int((frame["director"] == "").sum())
    report.missing_posters = int((frame["poster_url"] == "").sum())

    public_columns = EXPECTED_COLUMNS + OPTIONAL_COLUMNS
    ordered_columns = [column for column in public_columns if column in frame.columns]
    frame = frame[ordered_columns + [c for c in frame.columns if c not in ordered_columns]]
    records = [
        {key: _json_safe(value) for key, value in row.items() if key in public_columns}
        for row in frame.to_dict(orient="records")
    ]
    return CleanedDataset(frame=frame, records=records, report=report)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}
