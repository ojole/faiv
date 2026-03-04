import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env if present (safe if python-dotenv isn't installed)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'), override=False)
    load_dotenv(os.path.join(BASE_DIR, '.env.local'), override=True)
except Exception:
    pass

from faiv_app.core import fastapi_app
from a2wsgi import ASGIMiddleware

# Passenger looks for a top-level WSGI callable named 'application'
application = ASGIMiddleware(fastapi_app)
