"""Server-side TMDB watch-provider and cinema-link enrichment.

TMDB provides streaming/rent/buy providers and theatrical release metadata, but
it does not provide live cinema showtimes. This service therefore returns real
provider deep links plus clearly labelled ticketing search links, and never
invents availability.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping
from urllib.parse import quote_plus

import requests

from ml.ids import content_id_from_record

logger = logging.getLogger(__name__)

_RELEASE_TYPE = {
    1: "Premiere",
    2: "Limited theatrical",
    3: "Theatrical",
    4: "Digital",
    5: "Physical",
    6: "TV",
}


class WatchService:
    """Cached, fail-soft lookup of availability links from TMDB."""

    def __init__(self, config: Any) -> None:
        self.token = str(getattr(config, "TMDB_BEARER_TOKEN", "") or "").strip()
        self.api_root = str(getattr(config, "TMDB_API_ROOT", "https://api.themoviedb.org/3")).rstrip("/")
        self.default_region = str(getattr(config, "WATCH_REGION", "IN") or "IN").upper()
        self.timeout = float(getattr(config, "WATCH_TIMEOUT_SECONDS", 6))
        self.cache_ttl = float(getattr(config, "WATCH_CACHE_TTL_SECONDS", 21600))
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "accept": "application/json",
                "User-Agent": "MovieRecommender/1.0",
            })
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._guard = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def availability(self, record: Mapping[str, Any], region: str | None = None) -> dict[str, Any]:
        """Return provider and cinema-link information for one catalogue record."""
        identity = content_id_from_record(record)
        region_code = (region or self.default_region).strip().upper()[:2] or "IN"
        cache_key = (identity.model_key, region_code)
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        with self._lock_for(cache_key):
            cached = self._cached(cache_key)
            if cached is not None:
                return cached
            result = self._build(record, identity, region_code)
            if self.cache_ttl > 0:
                self._cache[cache_key] = (time.monotonic() + self.cache_ttl, result)
            return result

    # ----------------------------------------------------------------- private
    def _build(self, record: Mapping[str, Any], identity: Any, region: str) -> dict[str, Any]:
        tmdb: dict[str, Any] = {}
        errors: list[str] = []
        if self.enabled:
            try:
                tmdb = self._get(
                    f"/{identity.source_path}/{identity.source_id}/watch/providers"
                )
            except Exception as exc:  # network/API failures must not break the page
                errors.append(f"watch providers unavailable: {type(exc).__name__}")
            if identity.media_type == "movie":
                try:
                    dates = self._get(
                        f"/movie/{identity.source_id}/release_dates"
                    )
                    tmdb["_release_dates"] = dates.get("results") or []
                except Exception as exc:
                    errors.append(f"release dates unavailable: {type(exc).__name__}")

        providers = self._providers(tmdb, region)
        used_tmdb = bool(providers)
        if not any(providers.values()):
            providers = self._dataset_providers(record, region)

        result: dict[str, Any] = {
            "content_id": identity.public,
            "media_type": identity.media_type,
            "region": region,
            "source": "tmdb" if used_tmdb else "dataset_fallback",
            "providers": providers,
            "provider_page_url": ((tmdb.get("results") or {}).get(region) or {}).get("link")
            or self._justwatch_search(str(record.get("title") or ""), region),
            "cinema": self._cinema(identity, str(record.get("title") or ""), region, tmdb),
            "errors": errors,
            "note": (
                "Streaming availability is time-sensitive and comes from TMDB / JustWatch. "
                "TMDB does not supply live cinema showtimes; cinema links open the "
                "ticketing provider's current listings."
            ),
        }
        return result

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.api_root}{path}"
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException:
                if attempt == 2:
                    raise
                continue
            if response.status_code == 429 or response.status_code in {500, 502, 503, 504}:
                if attempt == 2:
                    response.raise_for_status()
                time.sleep(0.5 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        return {}

    @staticmethod
    def _providers(tmdb: Mapping[str, Any], region: str) -> dict[str, list[dict[str, Any]]]:
        results = tmdb.get("results") or {}
        region_data = results.get(region) or {}
        output: dict[str, list[dict[str, Any]]] = {}
        for category in ("flatrate", "free", "ads", "rent", "buy"):
            items: list[dict[str, Any]] = []
            seen: set[str] = set()
            for provider in region_data.get(category) or []:
                if not isinstance(provider, dict):
                    continue
                name = str(provider.get("provider_name") or "").strip()
                if not name or name.casefold() in seen:
                    continue
                seen.add(name.casefold())
                logo = provider.get("logo_path")
                items.append({
                    "name": name,
                    "logo_url": f"https://image.tmdb.org/t/p/w92{logo}" if logo else None,
                    "link": provider.get("link") or None,
                    "display_priority": provider.get("display_priority"),
                })
            if items:
                output[category] = items
        return output

    @staticmethod
    def _dataset_providers(record: Mapping[str, Any], region: str) -> dict[str, list[dict[str, Any]]]:
        """Use the builder's India snapshot only when it is relevant and real."""
        if region != "IN":
            return {}
        mapping = (
            ("flatrate", "india_streaming"),
            ("rent", "india_rent"),
            ("buy", "india_buy"),
        )
        from ml.preprocessing import split_list

        output: dict[str, list[dict[str, Any]]] = {}
        for category, column in mapping:
            names = split_list(record.get(column))
            if names:
                output[category] = [
                    {"name": name, "logo_url": None, "link": None, "display_priority": None}
                    for name in names
                ]
        return output

    @staticmethod
    def _cinema(
        identity: Any, title: str, region: str, tmdb: Mapping[str, Any]
    ) -> dict[str, Any]:
        if identity.media_type != "movie":
            return {
                "applicable": False,
                "status": "not_applicable",
                "release_dates": [],
                "links": [{
                    "label": "Watch provider options",
                    "url": WatchService._justwatch_search(title, region),
                }],
            }

        dates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for country in tmdb.get("_release_dates") or []:
            if not isinstance(country, dict):
                continue
            country_code = str(country.get("iso_3166_1") or "").upper()
            for item in country.get("release_dates") or []:
                if not isinstance(item, dict) or item.get("type") not in {1, 2, 3}:
                    continue
                date = str(item.get("release_date") or "")[:10]
                if not date:
                    continue
                key = (country_code, date)
                if key in seen:
                    continue
                seen.add(key)
                dates.append({
                    "country": country_code,
                    "date": date,
                    "type": _RELEASE_TYPE.get(int(item.get("type")), "Theatrical"),
                    "certification": item.get("certification") or None,
                })
        dates.sort(key=lambda item: (item["date"], item["country"]))
        return {
            "applicable": True,
            "status": "check_live_listings",
            "release_dates": dates[:12],
            "links": WatchService._cinema_links(title, region),
        }

    @staticmethod
    def _cinema_links(title: str, region: str) -> list[dict[str, str]]:
        query = quote_plus(title.strip())
        links: list[dict[str, str]] = []
        if region == "IN":
            links.append({
                "label": "BookMyShow — current showtimes",
                "url": f"https://in.bookmyshow.com/search?q={query}",
            })
        if region == "US":
            links.append({
                "label": "Fandango — current showtimes",
                "url": f"https://www.fandango.com/search?q={query}",
            })
        links.append({
            "label": "Google — showtimes near me",
            "url": f"https://www.google.com/search?q={query}+movie+showtimes+near+me",
        })
        return links

    @staticmethod
    def _justwatch_search(title: str, region: str) -> str:
        locale = region.lower() if region else "in"
        return f"https://www.justwatch.com/{locale}/search?q={quote_plus(title.strip())}"

    def _cached(self, key: tuple[str, str]) -> dict[str, Any] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            self._cache.pop(key, None)
            return None
        return value

    def _lock_for(self, key: tuple[str, str]) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock
