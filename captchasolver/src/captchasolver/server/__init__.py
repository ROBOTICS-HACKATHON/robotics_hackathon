"""HTTP API + web UI for captchasolver. Run locally with `captchasolver serve`."""
from .api import app, create_app, main

__all__ = ["app", "create_app", "main"]
