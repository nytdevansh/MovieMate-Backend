"""Schema and cleaning behaviour tests."""
from __future__ import annotations

import pandas as pd
import pytest

from ml.config import ModelConfig
from ml.preprocessing import (
    DataProcessingError,
    DatasetSchemaError,
    clean_dataset,
    normalize_list_field,
    parse_number,
    parse_year,
    split_list,
    valid_http_url,
)


def test_schema_validation_lists_missing_columns():
    frame = pd.DataFrame([{"tmdb_id": 1, "title": "Only two columns"}])
    with pytest.raises(DatasetSchemaError) as exc:
        clean_dataset(frame, ModelConfig())
    message = str(exc.value)
    assert "missing required column(s)" in message and "genres" in message


def test_cleaning_deduplicates_and_never_invents_values(synthetic_csv):
    raw = pd.read_csv(synthetic_csv, encoding="utf-8-sig")
    result = clean_dataset(raw, ModelConfig())

    assert result.report.rows_in == 16
    assert result.report.duplicate_ids_dropped == 1
    assert result.report.rows_out == 15
    records = {int(record["tmdb_id"]): record for record in result.records}
    assert records[1]["title"] == "Test Movie 1"  # more complete duplicate wins
    assert records[3]["vote_average"] is None  # invalid rating is not clamped
    assert records[4]["release_date"] is None  # invalid date stays missing
    assert records[4]["year"] is None  # no invented year
    assert records[5]["poster_url"] is None  # unsafe URL rejected
    assert split_list(records[6]["genres"]) == ["Drama", "Comedy"]


def test_invalid_id_rows_are_reported():
    from tests.conftest import movie_row

    frame = pd.DataFrame([movie_row(1, tmdb_id=value) for value in ["", "abc", 7.5]])
    with pytest.raises(DataProcessingError, match="valid tmdb_id"):
        clean_dataset(frame, ModelConfig())


def test_normalisation_helpers():
    assert normalize_list_field(None) == ""
    assert normalize_list_field(["Drama", " Drama ", "", "Comedy"]) == "Drama | Comedy"
    assert split_list("Drama |  Comedy | Drama") == ["Drama", "Comedy"]
    assert parse_number("7.5", minimum=0, maximum=10) == 7.5
    assert parse_number(99, minimum=0, maximum=10) is None
    assert parse_year("nonsense", "2025-02-11") == 2025
    assert parse_year(None, "") is None
    assert valid_http_url("https://image.tmdb.org/t/p/w500/x.jpg", {"image.tmdb.org"}).startswith("https://")
    assert valid_http_url("https://evil.example/x.jpg", {"image.tmdb.org"}) == ""
