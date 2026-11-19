"""WSGI entry point for production servers (gunicorn wsgi:app)."""
from __future__ import annotations

from backend.app import create_app

app = create_app()
