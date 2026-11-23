"""Serve the woven sample desk and its integrated photograph preparation room."""
import os
from pathlib import Path

from waitress import serve

from image_lab import create_app

from .web import install


def application(data=None):
    root = Path(data or os.environ.get("WEAVEDESK_DATA", ".weavedesk"))
    app = create_app(root / "preparation")
    install(app, root / "samples")
    return app


if __name__ == "__main__":
    serve(application(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), threads=4)
