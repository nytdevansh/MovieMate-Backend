"""Poster/backdrop resolution, validation, caching and optional downloading.

Resolution order: validated local image -> validated TMDB URL -> fallback SVG.
Only TMDB-ID-based filenames are exposed to browsers; filesystem paths are
never part of an API response.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import re
import threading
import time
import urllib.request
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.parse import quote, urlparse, urlunparse

from ml.preprocessing import normalize_free_text, valid_http_url

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif")
_FILENAME_RE = re.compile(r"^((?:tv-)?\d+)(\.[a-z0-9]+)$", re.IGNORECASE)
_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MIN_IMAGE_BYTES = 64
_CONTENT_TYPE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_REMOTE_CACHE_MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class PosterResolution:
    poster_url: str
    poster_source: str  # "local" | "tmdb" | "fallback"
    backdrop_url: str | None
    backdrop_source: str | None  # "local" | "tmdb" | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "poster_url": self.poster_url,
            "poster_source": self.poster_source,
            "backdrop_url": self.backdrop_url,
            "backdrop_source": self.backdrop_source,
        }


def is_valid_image_bytes(data: bytes) -> bool:
    """Check image magic bytes; reject HTML error pages and wrong formats."""
    if len(data) < 12:
        return False
    return (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
        or data.startswith((b"GIF87a", b"GIF89a"))
    )


def is_valid_image_file(path: Path) -> bool:
    try:
        if not path.is_file() or not (_MIN_IMAGE_BYTES <= path.stat().st_size <= _MAX_IMAGE_BYTES):
            return False
        with path.open("rb") as handle:
            return is_valid_image_bytes(handle.read(32))
    except OSError:
        return False


class PosterService:
    """Stateless-per-request resolver with a short-lived in-process cache."""

    def __init__(self, config: Any) -> None:
        self.poster_dirs: list[Path] = list(config.POSTER_DIRS)
        self.backdrop_dirs: list[Path] = list(config.BACKDROP_DIRS)
        self.fallback_url = normalize_free_text(config.FALLBACK_POSTER_URL) or (
            "/media/fallback/poster.svg"
        )
        self.remote_hosts = set(config.REMOTE_IMAGE_HOSTS)
        self.allow_remote = bool(config.ALLOW_REMOTE_POSTERS)
        self.auto_download = bool(config.AUTO_DOWNLOAD_POSTERS)
        self.download_timeout = float(config.DOWNLOAD_TIMEOUT_SECONDS)
        self.cache_ttl = float(config.POSTER_CACHE_TTL_SECONDS)
        self.proxy_remote = bool(getattr(config, "PROXY_REMOTE_IMAGES", True))
        self.remote_cache_ttl = float(
            getattr(config, "REMOTE_IMAGE_CACHE_MAX_AGE", 86400)
        )
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._locks: dict[tuple[str, int], threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._remote_cache: OrderedDict[str, tuple[float, bytes, str]] = OrderedDict()
        self._remote_cache_bytes = 0
        self._remote_cache_guard = threading.Lock()

    # ------------------------------------------------------------------ public
    def resolve(self, record: Mapping[str, Any]) -> PosterResolution:
        """Resolve poster/backdrop URLs for one catalogue record."""
        poster_id = self._poster_id(record)
        key = (
            "resolve",
            poster_id,
            normalize_free_text(record.get("poster_url")),
            normalize_free_text(record.get("backdrop_url")),
        )
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        poster_path = self.find_local("posters", poster_id)
        if poster_path is None and self.auto_download:
            poster_path = self.ensure_local_poster(
                poster_id, record.get("poster_url"), "posters"
            )
        if poster_path is not None:
            poster_url = self._media_url("posters", poster_id, poster_path)
            poster_source = "local"
        else:
            remote_poster = self.public_remote_url(record.get("poster_url"))
            if remote_poster:
                poster_url, poster_source = remote_poster, "tmdb"
            else:
                poster_url, poster_source = self.fallback_url, "fallback"

        backdrop_url: str | None = None
        backdrop_source: str | None = None
        backdrop_path = self.find_local("backdrops", poster_id)
        if backdrop_path is not None:
            backdrop_url = self._media_url("backdrops", poster_id, backdrop_path)
            backdrop_source = "local"
        else:
            remote_backdrop = self.public_remote_url(record.get("backdrop_url"))
            if remote_backdrop:
                backdrop_url, backdrop_source = remote_backdrop, "tmdb"

        result = PosterResolution(poster_url, poster_source, backdrop_url, backdrop_source)
        self._set_cached(key, result, self.cache_ttl)
        return result

    def find_local(self, kind: str, poster_id: str | int) -> Path | None:
        """First valid image for the ID across configured directories."""
        identifier = str(poster_id)
        key = ("local", kind, identifier, tuple(str(d) for d in self._dirs(kind)))
        cached = self._get_cached(key)
        if cached is not None:
            return cached or None
        found: Path | None = None
        for directory in self._dirs(kind):
            for suffix in IMAGE_SUFFIXES:
                candidate = directory / f"{identifier}{suffix}"
                if is_valid_image_file(candidate):
                    found = candidate
                    break
            if found is not None:
                break
        self._set_cached(key, found if found else False, self.cache_ttl)
        return found

    def media_path(self, kind: str, filename: str) -> Path | None:
        """Resolve a browser-supplied '<id>.<ext>' filename for static serving."""
        match = _FILENAME_RE.fullmatch(filename)
        if not match or kind not in {"posters", "backdrops"}:
            return None
        suffix = match.group(2).lower()
        if suffix not in IMAGE_SUFFIXES:
            return None
        poster_id = match.group(1).lower()
        for directory in self._dirs(kind):
            candidate = directory / f"{poster_id}{suffix}"
            if is_valid_image_file(candidate):
                return candidate
        return None

    def media_mimetype(self, path: Path) -> str:
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    def fallback_path(self) -> Path | None:
        """Serve the committed fallback SVG from the assets directory."""
        candidate = Path(__file__).resolve().parents[2] / "assets" / "fallback" / "poster.svg"
        return candidate if candidate.is_file() else None

    def ensure_local_poster(
        self,
        poster_id: str | int,
        url: Any,
        kind: str = "posters",
        force: bool = False,
    ) -> Path | None:
        """Download an image once, atomically, only when no valid local copy exists."""
        poster_id = str(poster_id)
        existing = self.find_local(kind, poster_id)
        if existing is not None:
            return existing
        if not force and not self.auto_download:
            return None
        remote = self._validated_remote_url(url)
        if not remote:
            return None

        negative_key = ("download_failed", kind, poster_id, remote)
        if not force and self._get_cached(negative_key) is not None:
            return None

        with self._lock_for(kind, poster_id):
            existing = self.find_local(kind, poster_id)
            if existing is not None:
                return existing
            download_dir = (self.poster_dirs if kind == "posters" else self.backdrop_dirs)
            if not download_dir:
                return None
            target_dir = download_dir[0]
            temporary: Path | None = None
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                request = urllib.request.Request(
                    remote, headers={"User-Agent": "MovieRecommender/1.0"}
                )
                with urllib.request.urlopen(request, timeout=self.download_timeout) as response:
                    content_type = response.headers.get_content_type()
                    if not content_type.startswith("image/"):
                        raise ValueError(f"unexpected content type {content_type!r}")
                    data = response.read(_MAX_IMAGE_BYTES + 1)
                if not _MIN_IMAGE_BYTES <= len(data) <= _MAX_IMAGE_BYTES or not is_valid_image_bytes(data):
                    raise ValueError("downloaded file is not a valid supported image")
                suffix = _CONTENT_TYPE_SUFFIX.get(content_type, ".jpg")
                temporary = target_dir / f".{poster_id}.{uuid.uuid4().hex}.part"
                temporary.write_bytes(data)
                final = target_dir / f"{poster_id}{suffix}"
                os.replace(temporary, final)
                self._cache.clear()
                return final
            except (OSError, URLError, ValueError) as exc:
                if temporary is not None and temporary.exists():
                    temporary.unlink(missing_ok=True)
                logger.warning("Image download failed for %s (id=%s): %s", kind, poster_id, exc)
                self._set_cached(negative_key, True, max(3600.0, self.cache_ttl))
                return None

    # ----------------------------------------------------------------- private
    def _dirs(self, kind: str) -> list[Path]:
        return self.poster_dirs if kind == "posters" else self.backdrop_dirs

    def _validated_remote_url(self, value: Any) -> str | None:
        """Return an allow-listed remote image URL, or None."""
        if not self.allow_remote:
            return None
        return valid_http_url(value, self.remote_hosts) or None

    def public_remote_url(self, value: Any) -> str | None:
        """URL browsers should request: first-party proxy when enabled."""
        remote = self._validated_remote_url(value)
        if not remote:
            return None
        if not self.proxy_remote:
            return remote
        parsed = urlparse(remote)
        if parsed.scheme == "http":
            # Always upgrade proxied image URLs; the page itself is HTTPS.
            remote = urlunparse(parsed._replace(scheme="https"))
        return "/media/remote?url=" + quote(remote, safe="")

    def fetch_remote_image(self, value: Any) -> tuple[bytes, str]:
        """Fetch and cache one validated image for the browser-side proxy.

        Only allow-listed hosts are accepted. This prevents the public endpoint
        from becoming a general-purpose URL fetcher while still letting clients
        load TMDB images from our own origin when their network blocks TMDB.
        """
        remote = self._validated_remote_url(value)
        if not remote:
            raise ValueError("Image host is not allowed")
        parsed = urlparse(remote)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Unsupported image URL scheme")
        if parsed.scheme == "http":
            remote = urlunparse(parsed._replace(scheme="https"))

        now = time.monotonic()
        with self._remote_cache_guard:
            cached = self._remote_cache.get(remote)
            if cached is not None:
                expires_at, data, content_type = cached
                if expires_at > now:
                    self._remote_cache.move_to_end(remote)
                    return data, content_type
                self._remote_cache.pop(remote, None)
                self._remote_cache_bytes -= len(data)

        request = urllib.request.Request(
            remote, headers={"User-Agent": "MovieMate-ImageProxy/1.0"}
        )
        with urllib.request.urlopen(request, timeout=self.download_timeout) as response:
            content_type = response.headers.get_content_type().strip().lower()
            if content_type not in _CONTENT_TYPE_SUFFIX:
                raise ValueError(f"Unsupported image type {content_type!r}")
            data = response.read(_MAX_IMAGE_BYTES + 1)
        if not _MIN_IMAGE_BYTES <= len(data) <= _MAX_IMAGE_BYTES or not is_valid_image_bytes(data):
            raise ValueError("Remote response was not a valid supported image")

        with self._remote_cache_guard:
            while (
                self._remote_cache
                and self._remote_cache_bytes + len(data) > _REMOTE_CACHE_MAX_BYTES
            ):
                _, evicted = self._remote_cache.popitem(last=False)
                self._remote_cache_bytes -= len(evicted[1])
            self._remote_cache[remote] = (
                time.monotonic() + max(0.0, self.remote_cache_ttl),
                data,
                content_type,
            )
            self._remote_cache_bytes += len(data)
        return data, content_type

    @staticmethod
    def _poster_id(record: Mapping[str, Any]) -> str:
        """Poster filenames are keyed by content, never by raw TMDB ID alone."""
        value = record.get("poster_key") or record.get("content_id") or record.get("tmdb_id")
        text = str(value).strip()
        if text.startswith("movie-"):
            return text.split("-", 1)[1]
        return text

    @staticmethod
    def _media_url(kind: str, poster_id: str, path: Path) -> str:
        return f"/media/{kind}/{poster_id}{path.suffix.lower()}"

    def _lock_for(self, kind: str, poster_id: str | int) -> threading.Lock:
        key = (kind, str(poster_id))
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _get_cached(self, key: tuple[Any, ...]) -> Any:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            self._cache.pop(key, None)
            return None
        return value

    def _set_cached(self, key: tuple[Any, ...], value: Any, ttl: float) -> None:
        if ttl > 0:
            self._cache[key] = (time.monotonic() + ttl, value)
