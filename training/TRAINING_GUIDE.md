# Training guide

This guide covers local training, Kaggle training, artifact validation, and inspection. Both environments execute the same `ml.trainer.train_model()` function, so results differ only if the input CSV, weights, or library versions differ.

## 1. What gets trained

A content-based retriever over six textual metadata fields:

| Field | Default weight | Source column(s) |
|---|---|---|
| `genres` | 3 | `genres` |
| `title` | 2 | `title` + `original_title` |
| `cast` | 2 | `cast` |
| `director` | 2 | `director` |
| `original_language` | 2 | `original_language` |
| `overview` | 1 | `overview` |

One shared `TfidfVectorizer` (1–2 grams, sublinear term frequency, Unicode accent stripping, `min_df=2`, `max_df=0.95`, up to 200,000 terms, no built-in normalisation) is fitted on the combined text of all movies. Each field is transformed separately, scaled by its weight, and concatenated into one CSR matrix, which is then L2-normalised. At query time, `X @ query.T` gives cosine similarity for all movies without building a dense N×N matrix.

Posters are **not** ML features. They are resolved by the web layer at request time.

## 2. Dataset contract

The input must be a CSV with these 26 columns (extra columns are allowed):

```
tmdb_id, imdb_id, title, original_title, release_date, year,
original_language, production_countries, genres, overview, cast,
director, runtime_minutes, vote_average, vote_count, popularity,
poster_path, poster_url, poster_local_path, backdrop_path,
backdrop_url, india_streaming, india_rent, india_buy, homepage, adult
```

Cleaning (`ml/preprocessing.py`) is deterministic and conservative:

- invalid or missing `tmdb_id` rows are dropped; duplicates keep the most complete row (tie-break: vote count, then title);
- missing metadata stays missing — no values are invented;
- invalid dates/ratings/counts are set to missing, not clamped; `year` and `release_date` can repair each other;
- pipe-delimited fields are trimmed, de-duplicated, and normalised with `split_list()` used by API serializers;
- remote image URLs are kept only when they are valid HTTP(S) URLs on `image.tmdb.org`.

All counts are recorded in `model_info.json` → `dataset_profile`.

## 3. Local training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Recommended full command: artifacts + export ZIP + inspection report
python training/train_local.py --report reports/generated/evaluation_report.md --zip-path
```

Defaults:

- CSV: `Frontend/movie_dataset_builder/output/movies_2023_2026.csv` (override with `--csv` or `TRAIN_CSV`)
- Output: `model/` (override with `--output-dir` or `TRAIN_OUTPUT_DIR`)
- ZIP: `movie_recommender_model.zip` next to the output directory when `--zip-path` is given without a value
- Poster directory: the builder's `posters/` folder, used only to count local images for the report (`--posters-dir` / `TRAIN_POSTERS_DIR`)

Expected output on the real 5,000-row dataset:

```
Trained on 5000 movies
Vocabulary: 53337 terms
Sparse feature matrix: 320022 columns, 423465 non-zero values
Training time: ~2 s
Artifacts: .../model
Export ZIP: .../movie_recommender_model.zip
Validation: loaded 5000 movies successfully
Inspection report: reports/generated/evaluation_report.md
```

### Changing weights

Override individual weights (all six fields are still required internally; omitted keys keep their defaults):

```bash
python training/train_local.py --weights '{"genres": 4, "overview": 2}'
```

Use a full configuration file for TF-IDF settings or top-K policy:

```bash
python training/train_local.py --config path/to/model_config.json
```

Start from the generated `model/model_config.json`, edit `feature_weights` / `tfidf`, then retrain. Weights must be positive, finite numbers; TF-IDF n-gram bounds and document-frequency limits are validated before training. Changing weights **requires retraining**: inference always uses the weights saved in `model_config.json`.

### Reusing the artifacts

After training, the Flask app discovers `model/` automatically. Verify without starting the server:

```bash
python -c "from ml.artifacts import load_artifacts; a=load_artifacts('model'); print(a.info['n_movies'], a.info['vocabulary_size'])"
```

## 4. Artifacts

`ml/trainer.train_model()` writes six files:

| File | Purpose |
|---|---|
| `tfidf_vectorizer.joblib` | Fitted shared vectorizer and IDF |
| `movie_matrix.npz` | L2-normalised weighted sparse matrix (CSR) |
| `movie_metadata.json` | Cleaned per-movie metadata keyed by string `tmdb_id` |
| `movie_id_mapping.json` | `index_to_id` list and `id_to_index` lookup |
| `model_config.json` | Effective weights, TF-IDF settings, schema version, top-K policy |
| `model_info.json` | Real dataset statistics, sizes, timing, library versions, evaluation note |

`load_artifacts()` rejects incomplete bundles, unsupported schema versions, wrong types, non-finite matrix values, mapping/metadata count mismatches, column counts that are not `vocabulary_size × 6`, and inconsistent IDs. `export_zip()` bundles the six files at the ZIP root.

## 5. Kaggle training

Kaggle needs no TMDB token and no access to your Mac. The source code travels as a ZIP because Kaggle cannot see your local folders.

### Step 1 — create the source bundle (local machine)

```bash
python scripts/make_kaggle_bundle.py
# -> kaggle_source.zip  (ml/*.py + training/train_kaggle.py)
```

Regenerate this ZIP whenever `ml/` changes; otherwise Kaggle will train with stale code.

### Step 2 — upload two datasets to Kaggle

In Kaggle → *Add Data*:

1. Upload `kaggle_source.zip` (e.g. as dataset `movie-recommender-source`).
2. Upload your movie CSV as a dataset (e.g. `movie-dataset`) containing `movies_2023_2026.csv`. A `posters/` folder may be included but is not needed for training.

Attach both datasets to a new notebook.

### Step 3 — import and run the notebook

Create a Kaggle Notebook, then *File → Import Notebook* and upload [`kaggle_train.ipynb`](kaggle_train.ipynb). Run all cells:

1. Cell 1 explains the inputs.
2. Cell 2 finds `/kaggle/input/**/kaggle_source.zip`, extracts it into `/kaggle/working/src`, and adds it to `sys.path`.
3. Cell 3 calls `train_kaggle.main(['--report'])`. The CSV is discovered **by schema** (required columns above), not by a hardcoded folder name or filename.
4. Cell 4 validates the artifacts and prints sample recommendations.

Headless/script variant inside Kaggle:

```bash
python /kaggle/working/src/training/train_kaggle.py --report
# or explicitly:
python train_kaggle.py --input-dir /kaggle/input/movie-dataset --output-dir /kaggle/working
```

Useful flags: `--csv-name` (force a filename), `--config`, `--no-zip`, `--report`. Environment overrides: `KAGGLE_INPUT_DIR`, `KAGGLE_OUTPUT_DIR`.

### Step 4 — download outputs

```
/kaggle/working/movie_model/                       six artifacts
/kaggle/working/movie_recommender_model.zip        download this file
/kaggle/working/evaluation_report.md               with --report
```

### Step 5 — install locally

```bash
python scripts/install_kaggle_model.py ~/Downloads/movie_recommender_model.zip
python run.py
```

The installer accepts ZIPs with the six files at the root or under `movie_model/`, rejects unsafe paths and incomplete bundles, validates the artifacts, and prints real movie/vocabulary counts. Restart the Flask process so it loads the new artifacts.

### Simulating Kaggle locally

You can exercise the same entry point without Kaggle paths:

```bash
python training/train_kaggle.py --input-dir Frontend/movie_dataset_builder/output \
  --output-dir /tmp/movie-kaggle-sim --report
```

The `training_environment` field in `model_info.json` will read `local_simulation`.

## 6. Inspection and evaluation report

```bash
python training/train_local.py --report reports/generated/evaluation_report.md --zip-path
# or, from existing artifacts:
python scripts/evaluate.py --model-dir model --output reports/generated/evaluation_report.md
```

The report contains:

- real dataset statistics (rows in/out, dropped IDs, missing-data percentages);
- model statistics (vocabulary, dimensions, non-zero count, matrix density, timing, artifact sizes, weights);
- deterministic top-5/top-10 recommendation examples for five well-described movies.

**No accuracy metrics are claimed.** The TMDB dataset has no relevance labels, user IDs, likes, or watch history, so Precision@K, Recall@K, and NDCG@K cannot be meaningfully computed. Any such number would be fabricated. To measure ranking quality later, collect explicit feedback with a time-split holdout, then compute those metrics against real labels.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Dataset is missing required column(s)` | CSV does not match the 26-column contract; re-run the dataset builder or pass the correct `--csv`. |
| `TF-IDF vocabulary is empty` | All text fields are empty after cleaning; check the CSV contents and pipe delimiters. |
| `At least two valid movies are needed` | Too many rows dropped by ID validation; inspect `dataset_profile`. |
| `Artifact inconsistency` at startup | Mismatched metadata/mapping/matrix; retrain into a clean output directory. |
| `MOVIE_NOT_IN_MODEL` from the API | Dataset was refreshed without retraining; retrain or reinstall artifacts. |
| Kaggle: `ModuleNotFoundError: ml` | Run the notebook's source-extraction cell first, or regenerate/upload `kaggle_source.zip`. |
| Kaggle: `No uploaded CSV with the expected TMDB schema` | Attach the CSV dataset to the notebook; `--csv-name` can force a specific filename. |
| Different local/Kaggle results | Compare `model_config.json` weights, `model_info.json` library versions, and `dataset_filename`; regenerate the source bundle if `ml/` changed. |
| `--weights` rejected | Must be a JSON object of positive finite numbers; all six fields remain in the effective config. |
| ZIP install rejected | Archive must contain all six canonical artifacts at the root or under `movie_model/`, with no unsafe paths. |

## 8. Reproducibility notes

- Cleaning, ID deduplication, and recommendation tie-breaks are deterministic (stable sorts, explicit ID tie-breaks).
- `model_info.json` records library versions and the exact dataset filename; check it when comparing runs.
- Retrain whenever the dataset, weights, TF-IDF settings, or `ml/` code changes.
- Keep the last known-good artifact bundle (or ZIP) for rollback before replacing `model/`.
