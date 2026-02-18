"""
FAIV API entry point.
Run with: uvicorn faiv_app.main:app --host 127.0.0.1 --port 8000 --reload
Or:       uvicorn faiv_app.core:fastapi_app --host 127.0.0.1 --port 8000 --reload
"""
from faiv_app.core import fastapi_app

app = fastapi_app
