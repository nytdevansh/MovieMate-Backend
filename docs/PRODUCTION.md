# Production deployment

This app is read-only, single-origin, and needs no database, message queue, or frontend build step. The main operational concerns are (1) shipping a valid model artifact set, (2) making the CSV and poster directories available, and (3) keeping image delivery cheap.

## 1. Prerequisites

- Python 3.10+ with `requirements.txt` installed; add `gunicorn` separately.
- The TMDB CSV at a stable path (`DATASET_PATH`).
- A validated model directory (`MODEL_DIR`) containing all six artifacts — trained locally or installed from the Kaggle ZIP.
- Optional local poster/backdrop directories. The app can also use allowlisted remote TMDB URLs and a committed fallback SVG.
- A reverse proxy for TLS (nginx, Caddy, or a managed equivalent).

## 2. Pre-flight checklist

Run these on the target machine before starting the server:

```bash
python -m pip install -r requirements.txt gunicorn

export FLASK_ENV=production
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export DATASET_PATH=/srv/moviemate/data/all_titles.csv
export MODEL_DIR=/srv/moviemate/model
export POSTER_DIRS=/srv/moviemate/assets/posters,/srv/moviemate/data/posters,/srv/moviemate/data/tv_posters
export BACKDROP_DIRS=/srv/moviemate/assets/backdrops
export WATCH_REGION=IN
# Optional: export TMDB_BEARER_TOKEN from your secret manager for live watch lookups.

# Assert artifacts are internally consistent without starting the web server.
python -c "from ml.artifacts import load_artifacts; a=load_artifacts('/srv/moviemate/model'); print(a.info.get('n_titles'), a.info.get('n_series', 0), a.info['vocabulary_size'])"
```

`ProductionConfig` refuses to start when `SECRET_KEY` is the development placeholder, the CSV is missing, or the model directory is missing. It also forces `DEBUG=False` and `REQUIRE_MODEL_AT_STARTUP=True`, so a bad bundle fails fast instead of serving degraded recommendations.

## 3. Running

```bash
gunicorn --workers 2 --threads 4 --timeout 120 --bind 127.0.0.1:8000 wsgi:app
```

`wsgi:app` calls `create_app()`, which loads the dataset and artifacts once **per worker**. For this workload, 2 workers × 4 threads is ample; each worker holds one cleaned catalogue (movies + series) plus the sparse matrix and the cached watch-provider responses. Measure RSS and adjust only if needed.

Never run gunicorn with `--reload` or `FLASK_DEBUG=true` in production.

## 4. Reverse proxy (example nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name movies.example.com;

    ssl_certificate     /etc/ssl/certs/movies.example.com.pem;
    ssl_certificate_key /etc/ssl/private/movies.example.com.key;

    client_max_body_size 2m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /media/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        # Let the app's ETag / Cache-Control headers reach clients and CDNs.
        proxy_buffering on;
    }
}
```

All paths — `/`, `/api`, `/js`, `/media`, `/frontend`, `/title`, `/admin.html` and `/tvshow` — must reach the same origin; the preserved Frontend pages use relative URLs by default.

## 5. Health checks and degraded mode

- Liveness/readiness: `GET /api/health`.
- Healthy when `data.status == "ok"` and `data.model_loaded == true`.
- With the default strict startup, an unhealthy app will not boot, which is the desired production behaviour.
- If someone explicitly sets `REQUIRE_MODEL_AT_STARTUP=false`, the app boots degraded: catalogue, search, filters, posters, and trending keep working, while recommendations return `503 MODEL_UNAVAILABLE` with the validation reason in `error.details.reason`. Never leave this setting on in production without alerting.

Recommended alerts: repeated `MODEL_UNAVAILABLE` responses (model/catalogue mismatch or corrupt bundle), a sudden drop in `titles_loaded` or `series_loaded`, `INTERNAL_ERROR` rate, and disk usage of poster directories.

## 6. Poster strategy

The app looks for `<content-key>.<ext>` in `POSTER_DIRS` in order — numeric `<tmdb_id>` for movies, `tv-<tmdb_id>` for series — then uses the validated TMDB URL, then the fallback SVG. Choose one of these approaches:

1. **Reuse builder output (simplest):** include `Frontend/movie_dataset_builder/output/posters` and `.../output/tv_posters` in `POSTER_DIRS`. No copying; the existing ~4,958 movie posters and 983 downloaded TV posters are served directly.
2. **Curated app-owned directory:** populate `/srv/moviemate/assets/posters` with `scripts/sync_posters.py --source ... --target ...`, optionally `--download-missing --csv ...` for the few missing IDs. Point `POSTER_DIRS` at that directory first.
3. **Remote-only:** set `ALLOW_REMOTE_POSTERS=true` with `REMOTE_IMAGE_HOSTS=image.tmdb.org`; rely on TMDB URLs and cache at the CDN.

Recommended production settings:

```bash
ALLOW_REMOTE_POSTERS=true
REMOTE_IMAGE_HOSTS=image.tmdb.org
AUTO_DOWNLOAD_POSTERS=false
DOWNLOAD_TIMEOUT_SECONDS=10
POSTER_CACHE_TTL_SECONDS=60
MEDIA_CACHE_MAX_AGE=86400
```

`AUTO_DOWNLOAD_POSTERS=true` makes the first request for a missing image perform a network fetch; keep it off in production and prefetch offline instead. Any download is host-allowlisted, size-limited (12 MB), magic-byte validated, written atomically under a per-ID lock, and negatively cached on failure.

For high traffic, put a CDN in front of `/media/*`. The app already sets conditional `ETag`/`Last-Modified` and `Cache-Control`, so cache hits are cheap. Note the licensing caveat in section 9.

## 7. Model updates and rollback

```bash
# Option A: reinstall a Kaggle export (validates the bundle before use)
python scripts/install_kaggle_model.py ~/Downloads/movie_recommender_model.zip

# Option B: retrain in place from the current CSV
python training/train_local.py --report reports/generated/evaluation_report.md --zip-path
```

`install_kaggle_model.py` extracts only the six known filenames, rejects path traversal, and calls `load_artifacts()` before printing success. Because app startup validates artifacts, the safe sequence is:

1. Install/train into a staging directory (e.g. `--output-dir /srv/moviemate/model-next` or `--target ...`).
2. Validate with `python -c "from ml.artifacts import load_artifacts; load_artifacts('<staging>')"`.
3. Swap directories (`mv model model-prev && mv model-next model`) and restart workers.
4. If health is not `ok`, point `MODEL_DIR` back to `model-prev` and restart.

Changing feature weights requires retraining — inference reads the effective weights from `model_config.json`, so a stale `model_info.json` or mismatched mapping is rejected at startup. If the dataset is refreshed, retrain too: otherwise IDs missing from the model return `422 MOVIE_NOT_IN_MODEL`.

## 8. Security checklist

- [ ] `FLASK_ENV=production`, `FLASK_DEBUG=false`, debug page disabled.
- [ ] `SECRET_KEY` is a random secret from the environment, not a file in version control.
- [ ] The TMDB bearer token is supplied only through the environment or the protected builder `.env`; it is never in code, artifacts, logs or browser responses.
- [ ] Dataset/model/poster paths are environment-driven; endpoints expose only `/media/...` URLs.
- [ ] `REMOTE_IMAGE_HOSTS` is an explicit allowlist; `AUTO_DOWNLOAD_POSTERS=false`.
- [ ] Reverse proxy terminates TLS and enforces the 2 MB body limit.
- [ ] Error responses are JSON envelopes with no stack traces or paths; unexpected errors are logged server-side.
- [ ] Default security headers (`X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options`) are present.
- [ ] Static media filenames are restricted to `<digits>.<image-ext>` or `tv-<digits>.<image-ext>`; image bytes are re-validated before serving. Protected Frontend data files, hidden files and traversal paths return 404.
- [ ] Log aggregation captures `MODEL_UNAVAILABLE`, `MOVIE_NOT_IN_MODEL`, and `INTERNAL_ERROR` events.

## 9. Data and licensing

- This product uses the TMDB API but is not endorsed or certified by TMDB. Keep the attribution/notice in the frontend footer.
- Watch-provider data is powered by JustWatch; verify its current attribution requirements.
- TMDB images are copyrighted by their owners. Serving posters from your own disk does not grant redistribution rights — review TMDB's terms before commercial use or bundling images.
- Model artifacts contain text derived from TMDB (titles, overviews, cast); treat them under the same terms and keep tokens out of logs and artifacts.

## 10. Known limits and future work

- The original movie dataset contains only 2023 releases; TV series span multiple years. Extend the movie builder through 2026 and retrain for broader coverage.
- Ranking is content-based only: no user accounts, ratings history, or feedback are collected, so no Precision@K/Recall@K/NDCG@K can be measured. Add explicit feedback with a time-split holdout before claiming accuracy.
- Posters are presentation-only; no visual embeddings are used in retrieval.
- Filtering/search run in memory over the combined catalogue. Revisit indexing if it grows by orders of magnitude.
- `MovieRepository` reads `movie_matrix` and catalogue once per worker; the process is read-only, so horizontal scaling is straightforward, but cache invalidation after a data refresh requires a rolling restart.
