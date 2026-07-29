"""
Server entry point.

Run (production / container):  python run.py
Run (local dev, hot reload):   DEV_RELOAD=1 python run.py

Reload defaults to OFF.  The Dockerfile's CMD is `python run.py`, so a default
of True meant HF Spaces ran uvicorn's file-watching dev reloader in production,
supervising a child process that holds the ~2.2GB BGE-M3 model.
"""
import os

import uvicorn

from app.config import APP_PORT

if __name__ == "__main__":
    dev_reload = os.getenv("DEV_RELOAD", "0") == "1"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=APP_PORT,
        reload=dev_reload,
        reload_dirs=["app"] if dev_reload else None,
    )
