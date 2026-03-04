import base64
import hashlib
import hmac
import logging
import os
import threading
import time
import uuid
from typing import Dict, Optional

logger = logging.getLogger(__name__)

EMBED_COOKIE_NAME = "faiv_embed"
EMBED_TOKEN_VERSION = "v1"
EMBED_NONCE_PREFIX = "faiv:embed:nonce"

EMBED_TOKEN_TTL_SECONDS = int(os.getenv("FAIV_EMBED_TOKEN_TTL_SECONDS", "60"))
EMBED_COOKIE_MAX_AGE_SECONDS = int(os.getenv("FAIV_EMBED_COOKIE_MAX_AGE_SECONDS", "3600"))

_memory_nonce_cache: Dict[str, float] = {}
_memory_nonce_lock = threading.Lock()

_redis_client = None
_redis_available = False
try:
    import redis

    _redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    _redis_client.ping()
    _redis_available = True
except Exception as ex:
    logger.warning("Embed auth Redis unavailable (%s). Using in-memory nonce cache.", ex)
    _redis_available = False


def _embed_secret() -> str:
    return os.getenv("FAIV_EMBED_SECRET", "").strip()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _sign(payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(digest)


def _cleanup_nonce_cache(now_ts: float) -> None:
    expired = [nonce for nonce, expires_at in _memory_nonce_cache.items() if expires_at <= now_ts]
    for nonce in expired:
        _memory_nonce_cache.pop(nonce, None)


def _mark_nonce_once(nonce: str, ttl_seconds: int) -> bool:
    ttl = max(1, int(ttl_seconds))
    if _redis_available and _redis_client is not None:
        try:
            key = f"{EMBED_NONCE_PREFIX}:{nonce}"
            created = _redis_client.set(key, "1", ex=ttl, nx=True)
            return bool(created)
        except Exception as ex:
            logger.warning("Embed nonce Redis write failed, using memory fallback: %s", ex)

    now_ts = time.time()
    expires_at = now_ts + ttl
    with _memory_nonce_lock:
        _cleanup_nonce_cache(now_ts)
        existing = _memory_nonce_cache.get(nonce)
        if existing and existing > now_ts:
            return False
        _memory_nonce_cache[nonce] = expires_at
        return True


def issue_embed_token(secret: Optional[str] = None) -> str:
    secret_value = secret or _embed_secret()
    if not secret_value:
        raise ValueError("FAIV_EMBED_SECRET is not configured.")
    timestamp_ms = int(time.time() * 1000)
    nonce = uuid.uuid4().hex
    payload = f"{timestamp_ms}.{nonce}"
    signature = _sign(payload, secret_value)
    return f"{EMBED_TOKEN_VERSION}.{timestamp_ms}.{nonce}.{signature}"


def verify_token(token: str, secret: Optional[str] = None, ttl_seconds: Optional[int] = None) -> bool:
    if not token:
        return False

    secret_value = secret or _embed_secret()
    if not secret_value:
        return False

    parts = token.split(".")
    if len(parts) != 4:
        return False

    version, timestamp_str, nonce, signature = parts
    if version != EMBED_TOKEN_VERSION:
        return False
    if not nonce:
        return False

    try:
        timestamp_ms = int(timestamp_str)
    except ValueError:
        return False

    ttl = max(1, int(ttl_seconds or EMBED_TOKEN_TTL_SECONDS))
    now_ms = int(time.time() * 1000)
    ttl_ms = ttl * 1000
    max_clock_skew_ms = 5000
    if timestamp_ms > now_ms + max_clock_skew_ms:
        return False
    if now_ms - timestamp_ms > ttl_ms:
        return False

    payload = f"{timestamp_ms}.{nonce}"
    expected_signature = _sign(payload, secret_value)
    if not hmac.compare_digest(signature, expected_signature):
        return False

    return _mark_nonce_once(nonce, ttl)


def create_embed_cookie_value(secret: Optional[str] = None) -> str:
    secret_value = secret or _embed_secret()
    if not secret_value:
        raise ValueError("FAIV_EMBED_SECRET is not configured.")

    issued_at = int(time.time())
    session_id = uuid.uuid4().hex
    payload = f"{issued_at}.{session_id}"
    signature = _sign(payload, secret_value)
    return f"{EMBED_TOKEN_VERSION}.{issued_at}.{session_id}.{signature}"


def verify_embed_cookie(cookie_value: str, secret: Optional[str] = None, max_age_seconds: Optional[int] = None) -> bool:
    if not cookie_value:
        return False

    secret_value = secret or _embed_secret()
    if not secret_value:
        return False

    parts = cookie_value.split(".")
    if len(parts) != 4:
        return False

    version, issued_at_str, session_id, signature = parts
    if version != EMBED_TOKEN_VERSION:
        return False
    if not session_id:
        return False

    try:
        issued_at = int(issued_at_str)
    except ValueError:
        return False

    max_age = max(1, int(max_age_seconds or EMBED_COOKIE_MAX_AGE_SECONDS))
    now_ts = int(time.time())
    if issued_at > now_ts + 5:
        return False
    if now_ts - issued_at > max_age:
        return False

    payload = f"{issued_at}.{session_id}"
    expected_signature = _sign(payload, secret_value)
    return hmac.compare_digest(signature, expected_signature)


def set_embed_cookie(
    response,
    secure: bool,
    secret: Optional[str] = None,
    cookie_value: Optional[str] = None,
    domain: Optional[str] = None,
) -> None:
    value = cookie_value or create_embed_cookie_value(secret=secret)
    response.set_cookie(
        key=EMBED_COOKIE_NAME,
        value=value,
        httponly=True,
        secure=secure,
        samesite="none",
        path="/",
        max_age=EMBED_COOKIE_MAX_AGE_SECONDS,
        domain=domain,
    )
