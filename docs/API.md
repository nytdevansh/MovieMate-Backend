# REST API reference

Base URL in development: `http://127.0.0.1:5000`. JSON endpoints use `GET`, except `POST /api/feedback`, and return `application/json; charset=utf-8`. Image routes return image bytes.

## Conventions

Every JSON response — success or failure — uses the same envelope:

```json
{
  "success": true | false,
  "data": <object | array | null>,
  "meta": { "...": "request-specific metadata" },
  "error": null | { "code": "...", "message": "...", "details": {} }
}
```

- `success` is `true` only for `2xx` responses; `error` is `null` then.
- `meta` carries pagination, counts, echoes of filters, or an empty object.
- Unknown query parameters are rejected with `400 INVALID_PARAMETER` so typos cannot silently change results.
- Numeric parameters must be finite; `page` is 1-based.
- Most endpoints are read-only and require no authentication. The only body endpoint is `POST /api/feedback`; `MAX_CONTENT_LENGTH` is 2 MB.

### Shared validation rules

| Parameter | Rule |
|---|---|
| `page` | integer ≥ 1 (default 1) |
| `per_page` | integer 1..`MAX_PER_PAGE` (default 24, max 100) |
| `year` | integer 1900..2100 |
| `min_rating` | finite number 0..10 |
| `limit` | integer ≥ 1; endpoint-specific maximum (see below) |
| `sort` | one of `popularity_desc`, `rating_desc`, `release_date_desc`, `title_asc` |
| `q` | required, 1–100 characters after control characters are stripped and whitespace collapsed |
| `genre`, `language` | trimmed text, max 100 / 20 characters respectively |
| `region` | two-letter uppercase country code, e.g. `IN`, `US` (default `WATCH_REGION`, normally `IN`) |

Generic title routes accept an explicit `content_id` path segment: `movie-<tmdb_id>` or `tv-<tmdb_id>`. Bare numeric IDs continue to mean movies for backwards compatibility with the original UI and API.

---

## `GET /api/health`

Reports readiness without exposing paths or secrets. Useful as a container/load-balancer health check.

**Example**

```bash
curl http://127.0.0.1:5000/api/health
```

```json
{
  "success": true,
  "data": {
    "status": "ok",
    "movies_loaded": 5000,
    "series_loaded": 1000,
    "titles_loaded": 6000,
    "model_loaded": true,
    "model_schema_version": 1,
    "model_error": null,
    "configuration": {
      "environment": "development",
      "dataset_present": true,
      "local_poster_dirs": 2,
      "remote_posters_enabled": true,
      "fallback_poster_enabled": true,
      "watch_provider_lookups_enabled": true
    },
    "timestamp_utc": "2026-11-18T10:15:30+00:00"
  },
  "meta": {},
  "error": null
}
```

`status` is `"degraded"` when the model failed validation and `REQUIRE_MODEL_AT_STARTUP=false`; `model_error` then contains the validation message. An HTTP `200` is still returned so callers can inspect the payload — check `data.status` / `data.model_loaded` for readiness.

---

## `GET /api/movies`

Paginated, filtered catalogue.

| Query param | Default | Notes |
|---|---|---|
| `page` | `1` | ≥ 1 |
| `per_page` | `24` | ≤ 100 |
| `genre` | — | Case-insensitive exact genre match |
| `year` | — | 1900..2100 |
| `language` | — | Case-insensitive original language code |
| `min_rating` | — | 0..10; movies without a rating are excluded |
| `sort` | `popularity_desc` | See shared rules |

**Example**

```bash
curl "http://127.0.0.1:5000/api/movies?genre=Drama&year=2023&min_rating=7&page=1&per_page=2&sort=rating_desc"
```

```json
{
  "success": true,
  "data": [
    {
      "movie_id": 872585,
      "title": "Oppenheimer",
      "release_year": 2023,
      "genres": ["Drama", "History", "War"],
      "rating": 8.023,
      "poster_url": "/media/posters/872585.jpg",
      "poster_source": "local",
      "backdrop_url": "https://image.tmdb.org/t/p/w1280/neeNHeXjMF5fXoCJRsOmkNGC7q.jpg",
      "short_description": "The story of J. Robert Oppenheimer's role in the development of the atomic bomb during World War II.",
      "original_language": "en"
    }
  ],
  "meta": {
    "page": 1,
    "per_page": 2,
    "total": 1234,
    "total_pages": 617,
    "has_next": true,
    "has_previous": false
  },
  "error": null
}
```

`total` above is illustrative; the real count depends on filters. Pages beyond the end return an empty `data` array with unchanged `total`.

---

## `GET /api/movies/<movie_id>`

Full detail for one TMDB ID. Returns the movie-summary fields **plus**:

`imdb_id`, `original_title`, `release_date`, `runtime_minutes`, `vote_count`, `popularity`, `overview`, `production_countries`, `cast`, `director`, `india_streaming`, `india_rent`, `india_buy`, `homepage`, `adult`, `backdrop_source`.

Pipe-delimited dataset fields are returned as clean string arrays. Missing metadata is `null` or `[]` — it is never fabricated.

```bash
curl http://127.0.0.1:5000/api/movies/872585
```

```json
{
  "success": true,
  "data": {
    "movie_id": 872585,
    "title": "Oppenheimer",
    "release_year": 2023,
    "genres": ["Drama", "History", "War"],
    "rating": 8.023,
    "poster_url": "/media/posters/872585.jpg",
    "poster_source": "local",
    "backdrop_url": "https://image.tmdb.org/t/p/w1280/neeNHeXjMF5fXoCJRsOmkNGC7q.jpg",
    "short_description": "The story of J. Robert Oppenheimer's role in the development of the atomic bomb during World War II.",
    "original_language": "en",
    "imdb_id": "tt15398776",
    "original_title": "Oppenheimer",
    "release_date": "2023-07-19",
    "runtime_minutes": 181.0,
    "vote_count": 12928.0,
    "popularity": 34.5184,
    "overview": "The story of J. Robert Oppenheimer's role in the development of the atomic bomb during World War II.",
    "production_countries": ["United Kingdom", "United States of America"],
    "cast": ["Cillian Murphy", "Emily Blunt", "Matt Damon", "Robert Downey Jr.", "Florence Pugh"],
    "director": ["Christopher Nolan"],
    "india_streaming": ["JioHotstar"],
    "india_rent": ["Apple TV Store", "Zee5", "YouTube", "Amazon Video"],
    "india_buy": ["Apple TV Store", "YouTube"],
    "homepage": "https://www.universalpictures.com/movies/oppenheimer",
    "adult": false,
    "backdrop_source": "tmdb"
  },
  "meta": {},
  "error": null
}
```

**Errors:** `404 MOVIE_NOT_FOUND`.

---

## `GET /api/search`

Ranks title (exact > prefix > substring), original title, genres, cast/director, and overview matches. No model required.

| Query param | Default | Notes |
|---|---|---|
| `q` | required | 1–100 characters |
| `limit` | `50` | ≤ `MAX_SEARCH_RESULTS` (50) |

```bash
curl "http://127.0.0.1:5000/api/search?q=nolan&limit=3"
```

```json
{
  "success": true,
  "data": [ { "movie_id": 872585, "title": "Oppenheimer", "...": "movie-summary fields" } ],
  "meta": { "query": "nolan", "count": 3 },
  "error": null
}
```

**Errors:** `400 INVALID_PARAMETER` for a missing/empty/too-long `q` or a bad `limit`.

---

## `GET /api/recommendations/<movie_id>`

Top-K content-based similar movies from the TF-IDF + cosine engine.

| Query param | Default | Notes |
|---|---|---|
| `limit` | `10` (`default_top_k` from `model_config.json`) | integer ≥ 1; the engine caps it at `max_top_k` (50) |

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
      "short_description": "A look at the making of Christopher Nolan's film about J. Robert Oppenheimer.",
      "poster_source": "local"
    },
    {
      "movie_id": 999125,
      "title": "Filip",
      "poster_url": "/media/remote?url=https%3A%2F%2Fimage.tmdb.org%2Ft%2Fp%2Fw500%2F....jpg",
      "genres": ["Drama", "History", "War"],
      "release_year": 2023,
      "rating": 6.9,
      "similarity_score": 0.143599,
      "short_description": "A young Polish activist fights to expose a dark secret from his country's past.",
      "poster_source": "tmdb"
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

Scores are rounded cosine similarities in `[0, 1]`; higher means more similar. The source movie is excluded, near-duplicate `(title, year)` pairs are skipped, and ordering is deterministic (score, then popularity, then TMDB ID) before personalisation. The `short_description` values above are truncated to 180 characters by the server; the exact text comes from the dataset.

Both recommendation routes set an anonymous `mm_session` cookie (HttpOnly, SameSite=Lax; `Secure` in production) on the first response. Each title shown is timestamped; later requests apply a time-decayed penalty so recently repeated titles sink and fresh titles from a wider candidate pool surface. The response sets `meta.personalization` instead of exposing session IDs or Q-values. Impressions are forgotten after `REPETITION_PENALTY_WINDOW_SECONDS`; state is in memory and resets on restart.

**Errors**

| Status | Code | Meaning |
|---|---|---|
| 404 | `MOVIE_NOT_FOUND` | ID absent from the catalogue |
| 422 | `MOVIE_NOT_IN_MODEL` | ID is in the catalogue but not in the trained artifacts (e.g. dataset refreshed without retraining) |
| 503 | `MODEL_UNAVAILABLE` | Model failed to load (degraded startup) |

---

## `GET /api/series`

Paginated TV-series-only catalogue. It accepts the same filters as `/api/movies`.

```bash
curl "http://127.0.0.1:5000/api/series?page=1&per_page=12&sort=popularity_desc"
```

Each item is the same summary shape as a movie, with `content_id: "tv-<id>"`, `media_type: "tv"`, `number_of_seasons`, `number_of_episodes`, `status` and `tv_type` available in detail responses.

---

## `GET /api/titles/<content_id>`

Full detail for either media type. `content_id` must be `movie-<tmdb_id>` or `tv-<tmdb_id>`.

```bash
curl http://127.0.0.1:5000/api/titles/tv-60574
```

The payload matches `/api/movies/<movie_id>` plus the identity fields and TV fields described above. **Errors:** `400 INVALID_CONTENT_ID`, `404 TITLE_NOT_FOUND`.

---

## `GET /api/titles/<content_id>/watch`

Server-side TMDB/JustWatch availability plus clearly labelled cinema links.

| Query param | Default | Notes |
|---|---|---|
| `region` | `WATCH_REGION` | Two-letter country code; current example: `IN` |

```bash
curl "http://127.0.0.1:5000/api/titles/movie-872585/watch?region=IN"
```

```json
{
  "success": true,
  "data": {
    "content_id": "movie-872585",
    "media_type": "movie",
    "region": "IN",
    "source": "tmdb",
    "providers": {
      "flatrate": [{ "name": "JioHotstar", "logo_url": "https://image.tmdb.org/t/p/w92/...", "link": "https://..." }],
      "rent": [{ "name": "Apple TV Store", "logo_url": null, "link": null, "display_priority": null }]
    },
    "provider_page_url": "https://www.justwatch.com/in/...",
    "cinema": {
      "applicable": true,
      "status": "check_live_listings",
      "release_dates": [{ "country": "IN", "date": "2023-07-21", "type": "Theatrical", "certification": "UA13+" }],
      "links": [
        { "label": "BookMyShow — current showtimes", "url": "https://in.bookmyshow.com/search?q=Oppenheimer" },
        { "label": "Google — showtimes near me", "url": "https://www.google.com/search?q=Oppenheimer+movie+showtimes+near+me" }
      ]
    },
    "errors": [],
    "note": "Streaming availability is time-sensitive ... TMDB does not supply live cinema showtimes ..."
  },
  "meta": {},
  "error": null
}
```

`US` requests also return a Fandango search link. TMDB provides provider and release-date metadata, **not live seat inventory**; ticketing links are labelled as live-listing searches and are never presented as guaranteed availability. Network failures degrade to the dataset's real India provider snapshot (when relevant) or a JustWatch search; no providers are invented. The token is never returned.

---

## `GET /api/titles/<content_id>/recommendations`

The same TF-IDF + cosine engine exposed by the legacy route, but accepts both `movie-<id>` and `tv-<id>` and can recommend across both media types. `limit` behaves as on `/api/recommendations/<movie_id>`; `meta.content_id` and `meta.media_type` echo the source title. `meta.personalization` has the same shape as on the legacy route.

```bash
curl "http://127.0.0.1:5000/api/titles/tv-60574/recommendations?limit=6"
```

**Errors:** `400 INVALID_CONTENT_ID`, `404 TITLE_NOT_FOUND`, `422 MOVIE_NOT_IN_MODEL`, `503 MODEL_UNAVAILABLE`.

---

## `POST /api/feedback`

Records a bounded learning signal for the anonymous session. The content score still leads ranking; these rewards only adjust a small title/genre Q-value multiplier. No account, profile or personal data is stored.

| Field | Type | Notes |
|---|---|---|
| `content_id` | string or number | Preferred `movie-<id>` or `tv-<id>`; a bare number means a movie |
| `event` | string | One of `open` (+0.35), `save` (+1.0), `remove` (−0.35), `dismiss` (−1.0) |

```bash
curl -X POST "http://127.0.0.1:5000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{"content_id":"movie-1","event":"save"}'
```

```json
{
  "success": true,
  "data": {
    "content_id": "movie-1",
    "event": "save",
    "reward": 1.0,
    "personalization": "session_only"
  },
  "meta": {"source": "anonymous_session"},
  "error": null
}
```

Send the `mm_session` cookie back on later requests (browsers do this automatically). Unknown fields, an unknown event or an invalid ID produce `400`; a title not in the catalogue produces `404`. `remove` and `dismiss` are negative rewards; the UI's × button sends `dismiss`.

**Errors:** `400 INVALID_BODY`, `400 INVALID_PARAMETER`, `400 INVALID_CONTENT_ID`, `404 TITLE_NOT_FOUND`.

---

## `GET /api/filters`

Values used to populate UI filter controls, derived from the loaded catalogue.

```json
{
  "success": true,
  "data": {
    "genres": ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "..."],
    "years": [2023],
    "languages": ["en", "hi", "ja", "..."],
    "rating_range": { "min": 0.0, "max": 10.0 },
    "sort_options": ["popularity_desc", "rating_desc", "release_date_desc", "title_asc"],
    "total_movies": 5000
  },
  "meta": {},
  "error": null
}
```

Arrays above are abbreviated. `years` is descending; genres and languages are sorted case-insensitively.

---

## `GET /api/posters/<movie_id>`

Resolved presentation URLs for one movie, useful for debugging poster precedence.

```json
{
  "success": true,
  "data": {
    "movie_id": 872585,
    "title": "Oppenheimer",
    "poster_url": "/media/posters/872585.jpg",
    "poster_source": "local",
    "backdrop_url": "https://image.tmdb.org/t/p/w1280/neeNHeXjMF5fXoCJRsOmkNGC7q.jpg",
    "backdrop_source": "tmdb"
  },
  "meta": {},
  "error": null
}
```

`poster_source` is `local`, `tmdb`, or `fallback`; `backdrop_source` is `local`, `tmdb`, or `null`.

---

## `GET /api/trending`

Most popular titles, tie-broken by vote count then canonical ID.

| Query param | Default | Maximum |
|---|---|---|
| `limit` | `12` | `100` |
| `media_type` | `movie` | `movie`, `tv`, or omit for all titles |

```json
{
  "success": true,
  "data": [ { "movie_id": 872585, "title": "Oppenheimer", "...": "movie-summary fields" } ],
  "meta": { "count": 12, "sort": "popularity_desc", "media_type": "movie" },
  "error": null
}
```

---

## Image routes

| Route | Description |
|---|---|
| `GET /media/posters/<id>.<ext>` | Validated local poster; `ext` ∈ `jpg,jpeg,png,webp,gif` |
| `GET /media/backdrops/<id>.<ext>` | Validated local backdrop |
| `GET /media/remote?url=<encoded-url>` | First-party proxy for allow-listed TMDB images; avoids client-side network blocking |
| `GET /media/fallback/poster.svg` | Committed SVG placeholder (the only accepted fallback filename) |

Images are served with correct MIME type, conditional `ETag`/`Last-Modified` support, and `Cache-Control: max-age=<MEDIA_CACHE_MAX_AGE>`. Any other filename or path returns `404 IMAGE_NOT_FOUND`; no arbitrary filesystem path can be requested.

## Frontend routes

| Route | File / behaviour |
|---|---|
| `GET /`, `GET /index.html`, `GET /frontend/` | Original `Frontend/index.html` sign-in page |
| `GET /js/<filename>` | `web/js/<filename>` (`ui-polish.js`, `frontend-enhance.js`) |
| `GET /dashboard.html`, `GET /frontend/dashboard.html` | Original `Frontend/dashboard.html`, byte-for-byte on disk, with enhancement injected into the HTTP response only |
| `GET /title/<content_id>` | Opens the original dashboard with the detail overlay already showing that title |
| `GET /frontend/<asset>`, `GET /<asset>` | Safe Frontend assets only: CSS, JS and image extensions |

Hidden files, `.env`, `.git`, CSV/JSON data files and path traversal are rejected with `404`. The protected Frontend HTML/CSS/images are never modified on disk. `/js/ui-polish.js` adds skip links, focus rings, reduced-motion support, labels and larger hit targets on every page; `/js/frontend-enhance.js` adds live sections, keyboard-accessible clickable cards, detail views, streaming/cinema links, personalised model recommendations and a localStorage My List on the dashboard only.

Frontend assets return `Cache-Control: no-cache, max-age=0`, so a deployment is visible on the next reload. In contrast, `/media/*` images return `public, max-age=<MEDIA_CACHE_MAX_AGE>` (with conditional validation).

## Error reference

| Status | `error.code` | Common cause |
|---|---|---|
| 400 | `INVALID_PARAMETER` | Missing required `q`, bad type/bound/choice, unknown query key or feedback event |
| 400 | `INVALID_BODY` | Missing, malformed or unknown field in `POST /api/feedback` |
| 404 | `NOT_FOUND` | Unknown route |
| 400 | `INVALID_CONTENT_ID` | Content ID is neither `movie-<id>` nor `tv-<id>` |
| 404 | `MOVIE_NOT_FOUND` | Legacy movie ID absent from catalogue |
| 404 | `TITLE_NOT_FOUND` | Generic content ID absent from catalogue |
| 404 | `IMAGE_NOT_FOUND` | Invalid image filename or no valid file on disk |
| 405 | `METHOD_NOT_ALLOWED` | Non-GET request to a GET-only route |
| 413 | `PAYLOAD_TOO_LARGE` | Request body above 2 MB |
| 422 | `MOVIE_NOT_IN_MODEL` | Catalogue/model mismatch after a dataset or model update |
| 503 | `MODEL_UNAVAILABLE` | Degraded model state |
| 500 | `INTERNAL_ERROR` | Unexpected server error (details logged server-side only) |

Example error:

```bash
curl "http://127.0.0.1:5000/api/movies?page=0"
```

```json
{
  "success": false,
  "data": null,
  "meta": {},
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "'page' must be >= 1",
    "details": {}
  }
}
```

## Frontend integration notes

`web/js/frontend-enhance.js` uses these endpoints exclusively; no recommendations are embedded in the client. It calls the generic title/watch/recommendation endpoints so the detail overlay on the preserved `Frontend/dashboard.html` always shows model output. To host the API separately, set before loading `frontend-enhance.js`:

```html
<script>window.MOVIEMATE_FRONTEND_CONFIG = { apiBaseUrl: "https://api.example.com/api", watchRegion: "IN" };</script>
```

The client checks `success`, reads `data` only on success, and surfaces `error.message` in a visible error state. `poster_url` is normally a same-origin `/media/...` path (local image, `/media/remote?url=...` proxy, or `/media/fallback/poster.svg`); all three render identically in an `<img>`. Set `PROXY_REMOTE_IMAGES=false` to emit direct allowlisted TMDB URLs instead.
