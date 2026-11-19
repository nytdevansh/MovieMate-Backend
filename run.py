#!/usr/bin/env python3
"""Development entry point: python run.py"""
from __future__ import annotations

from backend.app import create_app
from backend.config import Config

if __name__ == "__main__":
    config = Config.from_env()
    app = create_app(config)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
