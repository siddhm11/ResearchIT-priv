"""
Dev server entry point.
Run:  python run.py
"""
import uvicorn
from app.config import APP_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=APP_PORT,
        reload=True,
        reload_dirs=["app"],
    )
