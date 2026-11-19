# MovieMate — Movie & Web Series Recommendation System

A production-conscious, content-based recommendation platform for movies **and TV series**. It uses the TMDB datasets produced by your existing builder plus a new TV-series builder, trains a sparse TF-IDF + cosine-similarity retriever locally **or on Kaggle**, serves it through a modular Flask REST API, and displays real posters, live streaming links, cinema links and model recommendations in vanilla HTML/CSS/JS. A session-scoped time penalty fades repeated suggestions, and a tiny contextual bandit learns from opens, saves and hides so recommendations gradually fit the person browsing.

The original 5,000-movie CSV, every existing poster/card and all protected `Frontend/` HTML, CSS and image files are preserved. Only your original UI is served: the sign-in page owns `/`, the Netflix-style dashboard is at `/dashboard.html`, and the admin pages remain at their original filenames. The generated `/app` UI has been removed; its runtime enhancement script still powers the original dashboard's detail pages.

---

## A. Project overview and architecture

```
movie_recommendation/                  <- workspace root
├── Frontend/                          # PRESERVED: existing git repo, UI and image files
│   ├── index.html dashboard.html ...  # byte-for-byte unchanged user work
│   └── movie_dataset_builder/
│       ├── build_dataset.py           # unchanged TMDB movie collection pipeline
│       ├── build_webseries_dataset.py # new TMDB TV + combined catalogue builder
│       └── output/
│           ├── movies_2023_2026.csv   # original 5,000 movies (unchanged)
│           ├── webseries_tmdb.csv     # fetched TV series
│           ├── all_titles.csv         # movies + series used by the app/model
│           ├── posters/<tmdb_id>.jpg  # 4,958 movie posters
│           └── tv_posters/tv-<id>.jpg # TV posters
├── backend/                           # Flask REST API (app factory + blueprints)
│   ├── app.py  config.py
│   ├── routes/    health, movies, search, recommendations, feedback, filters, posters, titles
│   ├── services/  movie_service, recommendation_service, feedback_service, poster_service, watch_service
│   ├── models/    model_loader
│   └── utils/     validators, responses, errors
├── ml/                                # Shared, Flask-independent ML code
│   ├── config.py        feature weights + TF-IDF settings (single source of truth)
│   ├── ids.py           movie-<id> / tv-<id> identity helpers
│   ├── preprocessing.py schema validation, cleaning, deduplication, normalisation
│   ├── features.py      six weighted sparse field blocks -> one L2-normalised CSR matrix
│   ├── artifacts.py     save / validate / ZIP-export the six model artifacts
│   ├── trainer.py       one training entry point for local and Kaggle
│   ├── inference.py     sparse cosine retrieval engine
│   └── evaluate.py      dataset profile + manual top-5/top-10 inspection
├── training/                          # train_local.py, train_kaggle.py, kaggle_train.ipynb
├── scripts/                           # Kaggle bundle, model install, poster sync, report
├── web/js/frontend-enhance.js         # runtime enhancement for the preserved dashboard
├── assets/fallback/poster.svg         # committed fallback image
├── assets/posters/ , assets/backdrops/  # optional local image storage (gitignored)
├── data/all_titles.csv                # compact catalogue committed for deployment
├── model/                             # six committed deployable artifacts
├── reports/generated/                 # generated inspection report
├── tests/                             # 41 automated tests
├── docs/                              # ARCHITECTURE.md, API.md, PRODUCTION.md
├── requirements.txt  requirements-dev.txt
├── .env.example  .gitignore  pytest.ini
├── run.py (dev)  wsgi.py (production)
└── README.md
```

**Data flow**

```
TMDB API -> build_dataset.py        -> movies_2023_2026.csv + posters/<id>.jpg
         -> build_webseries_dataset -> webseries_tmdb.csv + tv_posters/tv-<id>.jpg
                                       |
                          all_titles.csv (all movies + series)
                                      |
                       ml.trainer (local OR Kaggle)
                                      |
     model/{tfidf_vectorizer.joblib, movie_matrix.npz, movie_metadata.json,
            movie_id_mapping.json, model_config.json, model_info.json}
                                      |
     Flask startup: load once, validate, build catalogue -> REST JSON
                                      |
                              /                              \
       REST JSON API                          web/js/frontend-enhance.js
       (all clients)                           (preserved Frontend UI + detail pages)
```

Design decisions:

- **Explicit content IDs.** Movies keep their historical numeric TMDB ID as the model key; public IDs are `movie-<id>` / `tv-<id>`. Bare numeric IDs still mean "movie", so existing links keep working. TV IDs live in a separate TMDB namespace and can never collide with movie IDs or posters (`tv-<id>.jpg`).
- **No dense N×N similarity matrix.** Each query vectorises one title, multiplies `CSR_matrix @ query.T` (sparse × sparse → N×1), and sorts. Memory stays proportional to the sparse matrix, not 6,000².
- **Posters are presentation, not ML features.** The baseline uses only the six textual fields. Poster lookup is a separate service; image embeddings can be added later without changing the API contract.
- **Honest availability links.** Watch providers come from TMDB/JustWatch; cinema links open real ticketing searches (BookMyShow, Fandango, Google). TMDB does not provide live showtimes, so the UI says so instead of inventing seat availability.
- **Personalisation stays honest.** The sparse cosine score still leads. Recently shown titles receive an exponential, time-bounded penalty, and anonymous open/save/hide signals update title and genre Q-values through an epsilon-greedy contextual bandit. No relevance labels are invented.
- **Protected Frontend work.** `Frontend/*.html`, CSS and images are served unchanged from disk. Response-time injections add `ui-polish.js` to every page and `frontend-enhance.js` plus config to the dashboard; files are never edited.
- **Single origin.** Flask serves the original UI, the APIs and `/media/*` images, so no CORS or framework is needed.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## B. Technology stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.10+, Flask 3, Flask blueprints, REST JSON API |
| Data | Pandas, NumPy, SciPy sparse matrices |
| ML | scikit-learn `TfidfVectorizer`, cosine similarity via sparse dot product, joblib serialization |
| Ranking | Time-decayed repetition penalty + session-scoped epsilon-greedy contextual bandit |
| Frontend | Pure HTML5, CSS3, vanilla JavaScript (ES2020), Fetch API |
| Tests | pytest (41 tests) |
| Training | Identical `ml/` code locally and in Kaggle; `/kaggle/input` → `/kaggle/working` |
| Optional ops | gunicorn + reverse proxy for production (no extra Python packages required by the app) |

No React/Next/Vue/Angular, no microservices, no database, no API secrets in code.

---

## C. Dataset preparation

The existing movie builder already produced the original dataset; the new TV builder adds series without touching it:

```bash
cd Frontend/movie_dataset_builder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add TMDB_BEARER_TOKEN (API Read Access Token)

python build_dataset.py       # unchanged original 5,000-movie pipeline

# New: TV series + combined catalogue (defaults: 1,000 series, local posters)
TARGET_SERIES=1000 DOWNLOAD_POSTERS=true python build_webseries_dataset.py
```

`build_dataset.py` writes `output/movies_2023_2026.csv`, `output/movies_2023_2026.jsonl`, `output/posters/<tmdb_id>.jpg` and checkpoints into `run_state.json`. `build_webseries_dataset.py` writes `output/webseries_tmdb.csv`, `output/webseries_tmdb.jsonl`, `output/tv_posters/tv-<id>.jpg`, `output/webseries_run_state.json`, and finally `output/all_titles.csv`. Both are safe to re-run; completed IDs are reused and the script refuses to write a combined CSV that lost any original movie ID.

The app prefers `all_titles.csv` when present and falls back to the original movie CSV on a fresh checkout. Set `DATASET_PATH` to override either choice.

Expected base columns (all are validated on load):

`tmdb_id, imdb_id, title, original_title, release_date, year, original_language, production_countries, genres, overview, cast, director, runtime_minutes, vote_average, vote_count, popularity, poster_path, poster_url, poster_local_path, backdrop_path, backdrop_url, india_streaming, india_rent, india_buy, homepage, adult`

The combined CSV additionally contains `content_id, media_type, source_tmdb_id, model_key, poster_key, number_of_seasons, number_of_episodes, status, tv_type`. Movie-only CSVs remain valid; these fields are derived during cleaning.

Handling implemented in `ml/preprocessing.py`:

- missing values → `None`/empty strings, **never invented**;
- duplicate canonical `content_id` → most complete row kept, deterministic tie-break, count reported; movie ID 123 and series ID 123 are distinct;
- invalid dates/years → missing (a valid `year` column is used when the date is invalid, and vice versa);
- invalid ratings outside 0–10, negative counts, non-finite numbers → missing (not clamped);
- missing/broken posters → TMDB URL → committed fallback SVG;
- unsafe or non-TMDB image URLs are rejected;
- text is NFKC-normalised, whitespace collapsed, pipe lists de-duplicated.

**Observed combined-catalogue facts** (from the real CSVs, not assumptions): 6,000 rows (5,000 movies + 1,000 TV series), 0 duplicate canonical IDs, 0 invalid IDs; missing overview 4.68%, genres 2.28%, cast 3.17%, director 7.87%, poster URL 0.88%, India streaming providers 79.05%. The original movie rows are all from 2023 because its builder reached its 5,000-movie target while processing 2023; TV series span multiple years and the movie pipeline itself supports 2023–2026.

---

## D. Local model training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Train from the combined movies + series catalogue (default when present)
python training/train_local.py --report reports/generated/evaluation_report.md --zip-path

# Custom feature weights (all six keys are still required internally; overrides are merged)
python training/train_local.py --weights '{"genres": 4, "overview": 2}'

# Full configuration file (see model/model_config.json after one training run)
python training/train_local.py --config path/to/model_config.json
```

The trainer writes the six canonical artifacts to `model/`:

| Artifact | Purpose |
|---|---|
| `tfidf_vectorizer.joblib` | shared fitted vectorizer and IDF |
| `movie_matrix.npz` | L2-normalised weighted sparse feature matrix (CSR) |
| `movie_metadata.json` | cleaned per-title metadata keyed by model key (`"123"` for movies, `"tv-123"` for series) |
| `movie_id_mapping.json` | matrix row ↔ canonical content ID mapping |
| `model_config.json` | effective feature weights and TF-IDF settings |
| `model_info.json` | real dataset stats, vocabulary/feature sizes, timing, library versions |

Default weights: **genres 3, title 2, cast 2, director 2, original language 2, overview 1**. Each field is transformed with the same fitted vectorizer, scaled by its weight, concatenated sparsely, then row-normalised; therefore cosine scores are comparable and weights are genuinely configurable (not faked by repeated text).

---

## E. Kaggle notebook setup

Kaggle must not need your Mac or TMDB token. Do this once:

1. Locally create the source bundle (shared `ml/` code + Kaggle trainer):

   ```bash
   python scripts/make_kaggle_bundle.py
   # -> kaggle_source.zip
   ```

2. In Kaggle → *Add Data*, upload:
   - `kaggle_source.zip` as a dataset (e.g. slug `movie-recommender-source`);
   - your catalogue CSV as a dataset (e.g. slug `movie-dataset`) containing `all_titles.csv` (or the original `movies_2023_2026.csv` for a movies-only model), optionally a `posters/` folder.

3. Create a new Kaggle Notebook, attach both datasets, then either:
   - upload [`training/kaggle_train.ipynb`](training/kaggle_train.ipynb) via *File → Import Notebook*, or
   - paste its cells into a fresh notebook.

4. Run all cells. Cell 2 extracts the source bundle into `/kaggle/working/src` and puts it on `sys.path`; cell 3 calls `train_kaggle.py`. The CSV is auto-detected **by schema**, so the dataset folder name does not matter.

Headless/script variant (inside Kaggle):

```bash
python /kaggle/working/src/training/train_kaggle.py --report
# or: python train_kaggle.py --input-dir /kaggle/input/movie-dataset --output-dir /kaggle/working
```

---

## F. Kaggle artifact export

After the notebook finishes:

```
/kaggle/working/movie_model/
    tfidf_vectorizer.joblib
    movie_matrix.npz
    movie_metadata.json
    movie_id_mapping.json
    model_config.json
    model_info.json
/kaggle/working/movie_recommender_model.zip   <- download this
/kaggle/working/evaluation_report.md          <- when --report was used
```

The ZIP contains the six files at its root (the installer also accepts a `movie_model/` folder inside the ZIP). No poster images are embedded; the Flask app resolves them from your local directories or TMDB URLs.

---

## G. Installing Kaggle-trained artifacts locally

```bash
# from the project root
python scripts/install_kaggle_model.py ~/Downloads/movie_recommender_model.zip

# verify what was installed
python -c "from ml.artifacts import load_artifacts; i=load_artifacts('model').info; print(i['n_movies'], i['vocabulary_size'])"
```

The installer rejects path traversal, refuses incomplete bundles, validates matrix/mapping/metadata consistency, and prints the real movie/vocabulary counts. It overwrites only the six canonical files in `model/`.

---

## H. Flask backend setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # requirements.txt in production
cp .env.example .env
python run.py                                          # http://127.0.0.1:5000
```

- The dataset and model are loaded **once at startup**, not per request.
- If artifacts are missing/invalid, startup fails by default (`REQUIRE_MODEL_AT_STARTUP=true`). With it set to `false`, health reports `degraded` and recommendation endpoints return `503 MODEL_UNAVAILABLE` while the rest of the API keeps working.
- `DevelopmentConfig` (default), `TestingConfig` (`FLASK_ENV=testing`) and `ProductionConfig` (`FLASK_ENV=production`) are in `backend/config.py`. Production requires `SECRET_KEY`, the dataset and a valid model directory.
- All paths/secrets come from environment variables; no filesystem path is returned by an endpoint.

---

## I. Frontend setup

There is no build step. Flask serves the original files directly: `/`, `/index.html`, `/dashboard.html`, `/adminlogin.html`, `/admin.html`, `/streaming_platform_admin_dashboard.html`, `/style.css`, `/script.js`, images under `/tvshow/` and `/js/frontend-enhance.js`.

Your original UI owns `/` (also `/index.html` and `/frontend/`), with the dashboard at `/dashboard.html`. Its files are not changed: Flask injects `ui-polish.js` on every page, and a config script plus `frontend-enhance.js` on the dashboard. `/app` and `/app/` now return `404`. The dashboard script keeps all existing cards/images, fills empty Movies, TV Shows, Trending, My List, Shuffle and Search sections from the API, binds every card to a detail page, and renders watch/cinema links plus model recommendations in an overlay. The × on a suggested title sends a negative learning signal; opening a title and adding/removing it from My List send positive/negative signals. My List uses browser `localStorage`; learning uses an anonymous, HttpOnly cookie and in-memory state. Safe Frontend assets (`css`, `js`, images) are served from any path; `.env`, hidden files, CSV/JSON data and traversal attempts return `404`. `/title/<content_id>` opens the dashboard directly on a title (e.g. `/title/tv-60574`).

If the API is hosted separately, set this before loading `frontend-enhance.js` (Flask normally injects it automatically):

```html
<script>window.MOVIEMATE_FRONTEND_CONFIG = { apiBaseUrl: "https://your-api.example.com/api", watchRegion: "IN" };</script>
```

`web/js/frontend-enhance.js` fills empty dashboard sections, binds cards to detail pages and renders streaming/cinema links plus personalised model recommendations from the API. `web/js/ui-polish.js` adds skip links, focus states, reduced-motion support and small target-size fixes without editing the original files. Nothing is hardcoded in the data layer.

> The filesystem is case-insensitive on macOS, so the enhancement script lives in `web/` — never `frontend/` — to avoid colliding with your existing `Frontend/` directory.

---

## J. Running the complete application

```bash
# 1. dataset already present in Frontend/movie_dataset_builder/output/
# 2. train (once)
python training/train_local.py --report reports/generated/evaluation_report.md --zip-path

# 3. run API + frontend
python run.py
```

Open:

- Original UI (root): <http://127.0.0.1:5000/>
- Enhanced dashboard: <http://127.0.0.1:5000/dashboard.html>
- Direct title detail page: <http://127.0.0.1:5000/title/tv-60574>
- Health: <http://127.0.0.1:5000/api/health>
- Example recommendations: <http://127.0.0.1:5000/api/titles/movie-872585/recommendations?limit=10>

Run the tests at any time:

```bash
python -m pytest          # 41 passed
```

---

## K. API endpoint documentation

Every response uses the envelope `{ "success": bool, "data": ..., "meta": {...}, "error": null | {code, message, details} }`.

| Endpoint | Description |
|---|---|
| `GET /api/health` | status, movie/series counts, model status, safe config summary |
| `GET /api/movies` | paginated **movie-only** catalogue with filters and sorting |
| `GET /api/movies/<movie_id>` | full movie detail (legacy numeric IDs) |
| `GET /api/series` | paginated **TV-series-only** catalogue with the same filters |
| `GET /api/search?q=…&limit=…` | ranked title/cast/director/genre search across both media types |
| `GET /api/recommendations/<movie_id>?limit=10` | top-K similar movies (legacy route) |
| `GET /api/titles/<content_id>` | full detail for `movie-<id>` or `tv-<id>` |
| `GET /api/titles/<content_id>/watch?region=IN` | streaming providers, release dates, cinema/ticketing links |
| `GET /api/titles/<content_id>/recommendations?limit=10` | top-K similar titles across movies and series, with time/feedback personalisation |
| `POST /api/feedback` | anonymous `open` / `save` / `remove` / `dismiss` reward signal for ranking |
| `GET /api/filters` | genres, years, languages, rating range, sort options, content counts |
| `GET /api/posters/<movie_id>` | resolved poster/backdrop URLs and source |
| `GET /api/trending?limit=12&media_type=movie\|tv` | most popular titles (defaults to movies) |
| `GET /media/posters/<id>.jpg`, `/media/backdrops/...`, `/media/fallback/poster.svg` | cached image bytes; TV posters use `tv-<id>.jpg` |

Filters supported on `/api/movies`: `page`, `per_page`, `genre`, `year`, `language`, `min_rating`, `sort` (`popularity_desc`, `rating_desc`, `release_date_desc`, `title_asc`).

Example — recommendations:

```bash
curl "http://127.0.0.1:5000/api/recommendations/872585?limit=2"
```

```json
{
  "success": true,
  "data": [
    {
      "movie_id": 1152711,
      "title": "Inside Christopher Nolan's Oppenheimer",
      "poster_url": "/media/posters/1152711.jpg",
      "genres": ["Documentary"],
      "release_year": 2023,
      "rating": 6.5,
      "similarity_score": 0.5462,
      "short_description": "…",
      "poster_source": "local"
    }
  ],
  "meta": {
    "movie_id": 872585,
    "source_title": "Oppenheimer",
    "returned": 2,
    "method": "tfidf_cosine_content_based",
    "personalization": {
      "ranking": "time_decayed_repetition_penalty_epsilon_greedy_bandit",
      "personalized": true,
      "timer_half_life_seconds": 21600,
      "penalty_strength": 0.55,
      "penalized_titles": 0,
      "penalized_in_results": 0,
      "explored_titles": 0,
      "learning_rate": 0.25
    }
  },
  "error": null
}
```

`meta.personalization` reports how ranking was adjusted; it never exposes the session ID or Q-values. The response sets an anonymous `mm_session` cookie (HttpOnly, SameSite=Lax). A title shown now is penalised on later requests with exponential decay (default half-life 6 hours, forgotten after 24 hours), so fresh candidates from the wider pool can surface. `POST /api/feedback` with `{"content_id": "movie-1", "event": "save"}` supplies a bounded reward to the bandit. State is in-memory and resets on restart.

Example — paginated movies:

```bash
curl "http://127.0.0.1:5000/api/movies?genre=Drama&year=2023&min_rating=7&page=1&per_page=2&sort=rating_desc"
```

`meta` contains `page`, `per_page`, `total`, `total_pages`, `has_next`, `has_previous`.

Error codes: `INVALID_PARAMETER`/`INVALID_CONTENT_ID` (400), `NOT_FOUND`/`MOVIE_NOT_FOUND`/`TITLE_NOT_FOUND`/`IMAGE_NOT_FOUND` (404), `METHOD_NOT_ALLOWED` (405), `MOVIE_NOT_IN_MODEL` (422), `MODEL_UNAVAILABLE` (503), `INTERNAL_ERROR` (500).

Full request/response reference: [`docs/API.md`](docs/API.md).

---

## L. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Model validation failed at startup` | Train (`python training/train_local.py`) or install the Kaggle ZIP; check `MODEL_DIR`. |
| `Dataset CSV not found` | Set `DATASET_PATH` in `.env`; defaults to the builder output. |
| `MODEL_UNAVAILABLE` on recommendations | Model missing when strict startup was disabled; train/install and restart. |
| Posters show the fallback SVG | No valid local file and remote disabled/invalid URL; run `python scripts/sync_posters.py` or set `ALLOW_REMOTE_POSTERS=true`. |
| Search returns nothing | Query must be 1–100 characters; search uses title/cast/director/genre/overview tokens. |
| `INVALID_PARAMETER` | Check bounds: `page ≥ 1`, `per_page ≤ 100`, `min_rating ≤ 10`, known `sort` value, no unknown query keys. |
| Kaggle: no CSV found | Dataset must contain the 26 TMDB columns; attach it as an input dataset and re-run cell 3. |
| Kaggle: `ModuleNotFoundError: ml` | Run cell 2 first or upload `kaggle_source.zip`; it extracts `ml/` into `/kaggle/working/src`. |
| Different results local vs Kaggle | Both use `ml.trainer.train_model`; compare `model_config.json` and `model_info.json` weights/versions. |
| Port 5000 in use | `FLASK_PORT=5001 python run.py`. |

---

## M. Environment variables

Copy [`.env.example`](.env.example). Full list:

| Variable | Default | Meaning |
|---|---|---|
| `FLASK_ENV` | `development` | `development`, `testing`, `production` |
| `FLASK_DEBUG` | `false` | Debug mode (never enable in production) |
| `FLASK_HOST` / `FLASK_PORT` | `127.0.0.1` / `5000` | Dev bind address |
| `SECRET_KEY` | dev placeholder | Required in production |
| `DATASET_PATH` | `all_titles.csv`, else original movie CSV | TMDB CSV location |
| `MODEL_DIR` | `model` | Installed artifacts directory |
| `POSTER_DIRS` | `assets/posters,<builder>/output/posters,<builder>/output/tv_posters` | Search order for local posters |
| `BACKDROP_DIRS` | `assets/backdrops` | Search order for backdrops |
| `FALLBACK_POSTER_URL` | `/media/fallback/poster.svg` | Last-resort image |
| `ALLOW_REMOTE_POSTERS` | `true` | Use TMDB URLs when no local image |
| `REMOTE_IMAGE_HOSTS` | `image.tmdb.org` | Host allowlist (SSRF-safe) |
| `AUTO_DOWNLOAD_POSTERS` | `false` | Download once per missing ID at request time (default off) |
| `DOWNLOAD_TIMEOUT_SECONDS` | `10` | Image fetch timeout |
| `POSTER_CACHE_TTL_SECONDS` | `60` | Resolution cache TTL |
| `MEDIA_CACHE_MAX_AGE` | `3600` | `Cache-Control` for `/media/*` |
| `REQUIRE_MODEL_AT_STARTUP` | `true` | Fail fast if artifacts missing/invalid |
| `WATCH_REGION` | `IN` | Default two-letter region for provider/cinema links |
| `WATCH_TIMEOUT_SECONDS` | `6` | Per-request TMDB timeout for availability lookups |
| `WATCH_CACHE_TTL_SECONDS` | `21600` | In-process availability cache TTL |
| `TMDB_BEARER_TOKEN` / `TMDB_API_ROOT` | builder `.env` / TMDB v3 | Server-side watch lookups only |
| `DEFAULT_PER_PAGE` / `MAX_PER_PAGE` | `24` / `100` | Pagination limits |
| `MAX_SEARCH_RESULTS` | `50` | Search result cap |
| `REPETITION_PENALTY_HALF_LIFE_SECONDS` | `21600` | Half-life of the time-decayed repeat penalty (6 hours) |
| `REPETITION_PENALTY_WINDOW_SECONDS` | `86400` | Impressions older than this are forgotten (24 hours) |
| `REPETITION_PENALTY_STRENGTH` | `0.55` | Maximum fractional score reduction for one recent impression |
| `RL_LEARNING_RATE` | `0.25` | Q-value update rate for session feedback |
| `RL_EXPLORATION_RATE` | `0.08` | Fraction of candidates receiving small deterministic jitter |
| `RL_TITLE_BONUS` / `RL_CONTEXT_BONUS` | `0.45` / `0.25` | Strength of title and genre Q-values in ranking |
| `PERSONALIZATION_SESSION_MAX_AGE_SECONDS` | `604800` | Anonymous session lifetime (7 days) |
| `PERSONALIZATION_MAX_SESSIONS` | `5000` | In-memory session cap before oldest sessions are evicted |
| `TRAIN_CSV`, `TRAIN_OUTPUT_DIR`, `TRAIN_POSTERS_DIR` | builder paths | Local training defaults |

The TMDB token is read from `Frontend/movie_dataset_builder/.env` (or the environment) by the dataset builder and the server-side watch service. It is never returned by an endpoint, sent to the browser, or stored in artifacts.

---

## N. Deployment instructions

### Live Render deployment

- App: <https://moviemate-3rui.onrender.com>
- Backend source: <https://github.com/nytdevansh/MovieMate-Backend>
- Frontend source: <https://github.com/nytdevansh/MovieMate>
- Render service: `moviemate` (Web Service, Python, free plan, health check `/api/health`)

The single Render Web Service serves the original `Frontend/` pages, the injected
UI enhancement scripts, `/api/*` and `/media/*` from one origin, so no CORS or
cross-origin poster rewriting is needed. TMDB posters and backdrops are emitted
as `/media/remote?url=...` and fetched server-side from the allow-listed TMDB
host, so the images also load on browser networks that block `image.tmdb.org`
directly. `render.yaml` describes the same
service for infrastructure-as-code setup. The compact combined catalogue is
committed at `data/all_titles.csv` and the six model artifacts are committed in
`model/`, so a fresh checkout starts without a training or download step. The
TMDB bearer token, when configured, is supplied only as a server-side Render
environment variable.

### Manual / self-hosted

```bash
pip install -r requirements.txt gunicorn
export FLASK_ENV=production
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export DATASET_PATH=/srv/moviemate/data/all_titles.csv
export MODEL_DIR=/srv/moviemate/model
export POSTER_DIRS=/srv/moviemate/assets/posters,/srv/moviemate/data/posters,/srv/moviemate/data/tv_posters
export WATCH_REGION=IN
# Optional: set TMDB_BEARER_TOKEN from your secret manager for live watch lookups.
gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000 wsgi:app
```

- Serve behind nginx/Caddy with TLS; proxy `/`, `/api`, `/css`, `/js`, `/media`, `/frontend` and `/title` to gunicorn.
- Each worker loads the dataset and sparse model once (memory: roughly the artifact size plus catalogue); 2 workers are plenty for this read-only workload.
- `Cache-Control` and conditional ETags are set on `/media/*`; put a CDN in front if traffic is high.
- Set `ALLOW_REMOTE_POSTERS` and `REMOTE_IMAGE_HOSTS` intentionally; keep `AUTO_DOWNLOAD_POSTERS=false` in production and populate images offline with `scripts/sync_posters.py`.
- For true image-based retrieval later, add a new artifact (e.g. image embeddings) and a new weighted block in `ml/features.py`; the API response shape does not need to change.

See [`docs/PRODUCTION.md`](docs/PRODUCTION.md) for a deployment checklist.

---

## O. TMDB attribution and image/data licensing

- This product uses the TMDB API but is not endorsed or certified by TMDB. The TMDB logo and this notice appear in the frontend footer and must be retained.
- Watch-provider data is powered by **JustWatch**; keep the credit and check its current attribution requirements.
- TMDB images are copyrighted by their respective owners. Downloading a poster for local display does **not** grant redistribution rights. Review TMDB's current API terms before commercial use or bundling images with a distributed artifact.
- The model artifacts contain textual metadata derived from TMDB (titles, cast names, overviews); treat them under the same terms. Do not expose the TMDB bearer token in logs, artifacts, frontend code or error messages.

---

## Verification results (real, from this repository)

- Local training on 6,000 titles (5,000 movies + 1,000 TV series): **3.2 s**, vocabulary **62,219**, sparse matrix **6,000 × 373,314**, **509,569** non-zero values, artifact bundle **~13.4 MB**.
- `python -m pytest`: **46 passed**.
- Live Flask smoke checks: every remaining page (`/`, `/index.html`, `/adminlogin.html`, `/admin.html`, `/streaming_platform_admin_dashboard.html`, `/dashboard.html`, `/title/<id>`), protected Frontend images/CSS/JS, `/media/posters/tv-<id>.jpg`, `/media/fallback/poster.svg`, all REST endpoints (200), invalid parameters (400), unknown IDs (404), removed `/app` (404), and `/.env` (404).
- Browser check: preserved dashboard cards stay intact while API-backed sections load; clicking any card opens the keyboard-accessible detail overlay with real streaming providers, cinema-search links and personalised model recommendations. Repeating a request lowers previously shown titles, and the × action, opens and My List changes feed the session bandit. The sign-in pages gain labels, larger targets and skip links through `ui-polish.js` without editing their files. TV detail pages correctly say cinema showtimes are not applicable, and a `<base href="/">` injection keeps all original relative images working from `/title/<id>`. Frontend assets use `Cache-Control: no-cache, max-age=0` so deployments are visible immediately, while `/media/*` uses `public, max-age=3600`.
- Manual inspection examples and missing-data percentages: [`reports/generated/evaluation_report.md`](reports/generated/evaluation_report.md).
- No accuracy claim is made: with no relevance labels or interaction data, Precision@K/Recall@K/NDCG@K cannot be meaningfully established.

### Remaining limitations

- The original movie CSV contains only 2023 releases (its builder stopped at 5,000 within 2023). TV series span multiple years; refresh/extend the movie dataset to cover 2024–2026 fully.
- One-vote 10.0 ratings and mostly missing India provider data are source-data artefacts, not model predictions; the API reports them as-is.
- My List is browser-local `localStorage`, not a server-side account/watchlist; collaborative/hybrid filtering and offline ranking metrics remain future work.
- Poster images are not ML features; visual-embedding retrieval would require its own implementation and evaluation.
