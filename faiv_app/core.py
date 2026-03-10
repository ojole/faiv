import os
import re
import json
import random
import logging
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.parse import urlparse
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware

from faiv_app.identity_codex import FAIV_IDENTITY_CODEX
from faiv_app.embed_auth import (
    EMBED_COOKIE_NAME,
    create_embed_cookie_value,
    issue_embed_token,
    set_embed_cookie,
    verify_embed_cookie,
    verify_token,
)

################################################
# 1) Logging & Redis (with in-memory fallback)
################################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load env files with explicit precedence:
# 1) .env as baseline defaults
# 2) .env.local as secret/local overrides
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)
except Exception as env_ex:
    logger.warning(f"python-dotenv load skipped: {env_ex}")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENV_NAME = (
    os.getenv("FAIV_ENV")
    or os.getenv("PYTHON_ENV")
    or os.getenv("ENV")
    or ""
).strip().lower()
IS_PRODUCTION = ENV_NAME in {"prod", "production"}
SITE_PASSWORD = os.getenv("FAIV_SITE_PASSWORD", "")
VISITOR_COOKIE_NAME = "faiv_vid"
AUTH_COOKIE_NAME = "faiv_auth"
VISITOR_COOKIE_MAX_AGE = 60 * 60 * 24 * 365
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7
RATE_LIMIT_PER_MINUTE = int(os.getenv("FAIV_RATE_LIMIT_PER_MINUTE", "10"))
ENABLE_OPENAI_MODERATION_GATE = _env_flag("FAIV_ENABLE_MODERATION_GATE", default=False)
EMBED_APP_URL = (os.getenv("FAIV_EMBED_APP_URL") or "https://faiv.ai").strip() or "https://faiv.ai"
FRAME_ANCESTORS_VALUE = "frame-ancestors 'self' https://jol3.com https://www.jol3.com"
EMBED_SESSION_HEADER = "x-faiv-embed-session"
EMBED_SESSION_QUERY_PARAM = "embed_session"
OPENAI_DEFAULT_MODEL = os.getenv("FAIV_OPENAI_MODEL", "gpt-5.4")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("FAIV_OPENAI_MAX_OUTPUT_TOKENS", "900"))
OPENAI_MAX_OUTPUT_TOKENS_VERDICT = int(os.getenv("FAIV_OPENAI_MAX_OUTPUT_TOKENS_VERDICT", "900"))
OPENAI_MAX_OUTPUT_TOKENS_DEEP = int(os.getenv("FAIV_OPENAI_MAX_OUTPUT_TOKENS_DEEP", "1900"))
OPENAI_MAX_OUTPUT_TOKENS_VERDICT_CAP = int(os.getenv("FAIV_OPENAI_MAX_OUTPUT_TOKENS_VERDICT_CAP", "1000"))
OPENAI_MAX_OUTPUT_TOKENS_DEEP_CAP = int(os.getenv("FAIV_OPENAI_MAX_OUTPUT_TOKENS_DEEP_CAP", "1800"))
OPENAI_VERDICT_TEMPERATURE = float(os.getenv("FAIV_OPENAI_TEMPERATURE_VERDICT", "0.2"))
OPENAI_DEEP_TEMPERATURE = float(os.getenv("FAIV_OPENAI_TEMPERATURE_DEEP", "0.42"))
OPENAI_VERDICT_TOP_P = float(os.getenv("FAIV_OPENAI_TOP_P_VERDICT", "0.8"))
OPENAI_DEEP_TOP_P = float(os.getenv("FAIV_OPENAI_TOP_P_DEEP", "0.9"))
OPENAI_VERDICT_FREQ_PENALTY = float(os.getenv("FAIV_OPENAI_FREQ_PENALTY_VERDICT", "0.2"))
OPENAI_DEEP_FREQ_PENALTY = float(os.getenv("FAIV_OPENAI_FREQ_PENALTY_DEEP", "0.25"))
OPENAI_VERDICT_PRES_PENALTY = float(os.getenv("FAIV_OPENAI_PRES_PENALTY_VERDICT", "0.1"))
OPENAI_DEEP_PRES_PENALTY = float(os.getenv("FAIV_OPENAI_PRES_PENALTY_DEEP", "0.3"))
OPENAI_CLIENT_TIMEOUT_SECONDS = float(os.getenv("FAIV_OPENAI_TIMEOUT_SECONDS", "35"))
OPENAI_TIMEOUT_SECONDS_VERDICT = float(os.getenv("FAIV_OPENAI_TIMEOUT_SECONDS_VERDICT", "95"))
OPENAI_TIMEOUT_SECONDS_DEEP = float(os.getenv("FAIV_OPENAI_TIMEOUT_SECONDS_DEEP", "180"))
OPENAI_CLIENT_MAX_RETRIES = int(os.getenv("FAIV_OPENAI_MAX_RETRIES", "1"))
IDEMPOTENCY_TTL_SECONDS = int(os.getenv("FAIV_IDEMPOTENCY_TTL_SECONDS", "180"))

# In-memory rate-limit fallback when Redis is unavailable
_memory_rate_limits = {}

# In-memory session fallback when Redis is unavailable
_memory_sessions = {}
_memory_idempotency_cache = {}
_redis_available = False
redis_client = None

try:
    import redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    _redis_available = True
    logger.info("Redis connected successfully.")
except Exception as e:
    logger.warning(f"Redis unavailable ({e}). Using in-memory session storage.")
    _redis_available = False


def session_get(key: str) -> Optional[str]:
    if _redis_available:
        try:
            return redis_client.get(key)
        except Exception:
            logger.warning("Redis read failed, falling back to memory.")
    return _memory_sessions.get(key)


def session_set(key: str, value: str):
    if _redis_available:
        try:
            redis_client.set(key, value)
            return
        except Exception:
            logger.warning("Redis write failed, falling back to memory.")
    _memory_sessions[key] = value


def session_delete(key: str):
    if _redis_available:
        try:
            redis_client.delete(key)
        except Exception:
            pass
    _memory_sessions.pop(key, None)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. Server will start but queries will fail.")
    client = None
else:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_CLIENT_TIMEOUT_SECONDS,
        max_retries=OPENAI_CLIENT_MAX_RETRIES,
    )


def _cookie_secure(request: Request) -> bool:
    if IS_PRODUCTION:
        return True
    x_forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
    if x_forwarded_proto == "https":
        return True
    return request.url.scheme == "https"


def _set_visitor_cookie(response, request: Request, visitor_id: str) -> None:
    response.set_cookie(
        key=VISITOR_COOKIE_NAME,
        value=visitor_id,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        max_age=VISITOR_COOKIE_MAX_AGE,
    )


def _set_auth_cookie(response, request: Request) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value="1",
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        max_age=AUTH_COOKIE_MAX_AGE,
    )


def _public_path(path: str) -> bool:
    if path in {
        "/locked",
        "/embed",
        "/api/unlock",
        "/api/auth-status",
        "/api/faiv-embed-token",
        "/health",
        "/openapi.json",
        "/docs",
        "/redoc",
    }:
        return True
    if path.startswith("/static/"):
        return True
    return False


def _api_path(path: str) -> bool:
    return path.startswith("/query") or path.startswith("/redeliberate") or path.startswith("/reset")


def _has_valid_embed_cookie(request: Request) -> bool:
    embed_cookie = request.cookies.get(EMBED_COOKIE_NAME, "")
    return verify_embed_cookie(embed_cookie)


def _extract_embed_session_token(request: Request) -> str:
    header_value = request.headers.get(EMBED_SESSION_HEADER, "").strip()
    if header_value:
        return header_value
    return request.query_params.get(EMBED_SESSION_QUERY_PARAM, "").strip()


def _has_valid_embed_session(request: Request) -> bool:
    embed_session = _extract_embed_session_token(request)
    return bool(embed_session and verify_embed_cookie(embed_session))


def _is_request_authenticated(request: Request) -> bool:
    if request.cookies.get(AUTH_COOKIE_NAME) == "1":
        return True
    if _has_valid_embed_cookie(request):
        return True
    return _has_valid_embed_session(request)


def _embed_cookie_domain(request: Request) -> Optional[str]:
    host = (request.url.hostname or "").lower()
    if host.endswith("faiv.ai"):
        return ".faiv.ai"
    return None


def _merge_frame_ancestors(existing_csp: Optional[str]) -> str:
    if not existing_csp:
        return FRAME_ANCESTORS_VALUE

    if "frame-ancestors" in existing_csp.lower():
        return re.sub(
            r"frame-ancestors[^;]*",
            FRAME_ANCESTORS_VALUE,
            existing_csp,
            flags=re.IGNORECASE,
        )

    separator = "" if existing_csp.strip().endswith(";") else "; "
    return f"{existing_csp}{separator}{FRAME_ANCESTORS_VALUE}"


def _apply_frame_headers(response) -> None:
    if "x-frame-options" in response.headers:
        del response.headers["x-frame-options"]
    existing_csp = response.headers.get("content-security-policy")
    response.headers["content-security-policy"] = _merge_frame_ancestors(existing_csp)


def _is_allowed_embed_origin(request: Request) -> bool:
    allowed_hosts = {"jol3.com", "www.jol3.com", "localhost", "127.0.0.1"}
    for header in ("origin", "referer"):
        raw_value = request.headers.get(header, "").strip()
        if not raw_value:
            continue
        try:
            parsed = urlparse(raw_value)
        except Exception:
            continue
        hostname = (parsed.hostname or "").lower()
        if hostname in allowed_hosts:
            return True
    return False


def _is_allowed_cors_origin(origin: str) -> bool:
    if not origin:
        return False
    try:
        parsed = urlparse(origin)
    except Exception:
        return False

    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port

    if scheme == "https" and host in {"faiv.ai", "www.faiv.ai", "jol3.com", "www.jol3.com"}:
        return True
    if scheme == "http" and host in {"localhost", "127.0.0.1"}:
        return port in {None, 3000}
    return False


def _append_vary_header(response, value: str) -> None:
    existing = response.headers.get("vary")
    if not existing:
        response.headers["vary"] = value
        return
    existing_values = {item.strip().lower() for item in existing.split(",") if item.strip()}
    if value.lower() in existing_values:
        return
    response.headers["vary"] = f"{existing}, {value}"


def _apply_cors_headers(request: Request, response) -> None:
    origin = request.headers.get("origin", "").strip()
    if not origin:
        return
    if not _is_allowed_cors_origin(origin):
        return
    response.headers["access-control-allow-origin"] = origin
    response.headers["access-control-allow-credentials"] = "true"
    _append_vary_header(response, "Origin")


def _cleanup_idempotency_cache(now_ts: float) -> None:
    stale_keys = [
        key
        for key, (expires_ts, _payload) in _memory_idempotency_cache.items()
        if expires_ts <= now_ts
    ]
    for key in stale_keys:
        _memory_idempotency_cache.pop(key, None)


def _idempotency_get(cache_key: str):
    if not cache_key:
        return None
    now_ts = time.time()
    _cleanup_idempotency_cache(now_ts)
    cached = _memory_idempotency_cache.get(cache_key)
    if not cached:
        return None
    expires_ts, payload = cached
    if expires_ts <= now_ts:
        _memory_idempotency_cache.pop(cache_key, None)
        return None
    return payload


def _idempotency_set(cache_key: str, payload) -> None:
    if not cache_key:
        return
    expires_ts = time.time() + max(30, IDEMPOTENCY_TTL_SECONDS)
    _memory_idempotency_cache[cache_key] = (expires_ts, payload)


def is_disallowed_prompt(text: str) -> bool:
    if not text:
        return False

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False

    disallowed_patterns = [
        re.compile(r"\bcsam\b", re.IGNORECASE),
        re.compile(r"\bchild\s+sexual\s+abuse\s+material\b", re.IGNORECASE),
        re.compile(r"\b(?:sexual|sex)\b.{0,28}\b(?:minor|child|underage)\b", re.IGNORECASE),
        re.compile(r"\b(?:minor|child|underage)\b.{0,28}\b(?:sexual|sex)\b", re.IGNORECASE),
        re.compile(r"\b(?:groom|grooming)\b.{0,20}\b(?:child|minor|underage)\b", re.IGNORECASE),
        re.compile(r"\b(?:nude|explicit|porn)\b.{0,28}\b(?:child|minor|underage)\b", re.IGNORECASE),
        re.compile(r"\b(?:child|minor|underage)\b.{0,28}\b(?:nude|explicit|porn)\b", re.IGNORECASE),
        re.compile(r"\b(?:underage|minor)\s+explicit\s+content\b", re.IGNORECASE),
    ]
    return any(pattern.search(normalized) for pattern in disallowed_patterns)


def moderation_gate(text: str) -> bool:
    """Optional OpenAI moderation gate. Returns True when input should be blocked."""
    if not ENABLE_OPENAI_MODERATION_GATE or client is None or not text:
        return False
    try:
        moderation_model = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
        moderation = client.moderations.create(model=moderation_model, input=text)
        result = moderation.results[0] if moderation and moderation.results else None
        return bool(getattr(result, "flagged", False))
    except Exception as ex:
        logger.warning(f"Moderation gate failed open: {ex}")
        return False


def _cleanup_rate_limit_cache(now_ts: float) -> None:
    stale_keys = [key for key, (_, expires_ts) in _memory_rate_limits.items() if expires_ts <= now_ts]
    for key in stale_keys:
        _memory_rate_limits.pop(key, None)


def rate_limit_ok(visitor_id: str) -> bool:
    if not visitor_id:
        return True

    now_ts = time.time()
    minute_bucket = int(now_ts // 60)
    key = f"rl:{visitor_id}:{minute_bucket}"

    if _redis_available and redis_client is not None:
        try:
            count = redis_client.incr(key)
            if count == 1:
                redis_client.expire(key, 60)
            return count <= RATE_LIMIT_PER_MINUTE
        except Exception as ex:
            logger.warning(f"Redis rate-limit failed, using memory fallback: {ex}")

    _cleanup_rate_limit_cache(now_ts)
    count, expires_ts = _memory_rate_limits.get(key, (0, now_ts + 60))
    if expires_ts <= now_ts:
        count = 0
        expires_ts = now_ts + 60
    count += 1
    _memory_rate_limits[key] = (count, expires_ts)
    return count <= RATE_LIMIT_PER_MINUTE


def _log_block_event(request: Request, visitor_id: str, reason: str) -> None:
    logger.warning(
        "blocked_request visitor_id=%s path=%s blocked=True reason=%s ts=%s",
        visitor_id,
        request.url.path,
        reason,
        datetime.now(timezone.utc).isoformat(),
    )


################################################
# 2) Summaries & Utility
################################################

MODE_VERDICT = "verdict"
MODE_DEEP = "deep"
ALLOWED_MODES = {MODE_VERDICT, MODE_DEEP}


def normalize_mode(mode: Optional[str]) -> str:
    if not mode:
        return MODE_VERDICT
    normalized = str(mode).strip().lower()
    if normalized in {"common", MODE_VERDICT}:
        return MODE_VERDICT
    if normalized in {"rare", MODE_DEEP}:
        return MODE_DEEP
    return MODE_VERDICT


def get_max_output_tokens(mode: str) -> int:
    normalized = normalize_mode(mode)
    if normalized == MODE_DEEP:
        requested = max(1200, OPENAI_MAX_OUTPUT_TOKENS_DEEP)
        cap = max(1600, OPENAI_MAX_OUTPUT_TOKENS_DEEP_CAP)
        return min(requested, cap)
    requested = max(600, OPENAI_MAX_OUTPUT_TOKENS_VERDICT)
    cap = max(800, OPENAI_MAX_OUTPUT_TOKENS_VERDICT_CAP)
    return min(requested, cap)


def get_request_timeout(mode: str) -> float:
    normalized = normalize_mode(mode)
    if normalized == MODE_DEEP:
        return max(OPENAI_CLIENT_TIMEOUT_SECONDS, OPENAI_TIMEOUT_SECONDS_DEEP)
    return max(OPENAI_CLIENT_TIMEOUT_SECONDS, OPENAI_TIMEOUT_SECONDS_VERDICT)


def get_model_settings(mode: str) -> dict:
    normalized = normalize_mode(mode)
    if normalized == MODE_DEEP:
        return {
            "temperature": OPENAI_DEEP_TEMPERATURE,
            "top_p": OPENAI_DEEP_TOP_P,
            "frequency_penalty": OPENAI_DEEP_FREQ_PENALTY,
            "presence_penalty": OPENAI_DEEP_PRES_PENALTY,
        }
    return {
        "temperature": OPENAI_VERDICT_TEMPERATURE,
        "top_p": OPENAI_VERDICT_TOP_P,
        "frequency_penalty": OPENAI_VERDICT_FREQ_PENALTY,
        "presence_penalty": OPENAI_VERDICT_PRES_PENALTY,
    }


def create_chat_completion(
    *,
    model: str,
    messages: list,
    max_output_tokens: int,
    user: str,
    generation: dict,
    mode: str,
    response_format: Optional[dict] = None,
):
    """Call Chat Completions with token-parameter compatibility across models."""
    normalized_mode = normalize_mode(mode)
    request_timeout = get_request_timeout(normalized_mode)
    model_lower = (model or "").strip().lower()
    supports_legacy_max_tokens = not model_lower.startswith("gpt-5")

    base_request_kwargs = {
        "model": model,
        "messages": messages,
        "user": user,
        **generation,
    }

    def _invoke_with_tokens(
        token_budget: int,
        use_legacy_tokens: bool = False,
        timeout_override: Optional[float] = None,
        use_response_format: bool = True,
    ):
        request_kwargs = dict(base_request_kwargs)
        request_kwargs["timeout"] = timeout_override or request_timeout
        if response_format and use_response_format:
            request_kwargs["response_format"] = response_format
        if use_legacy_tokens:
            request_kwargs["max_tokens"] = token_budget
        else:
            request_kwargs["max_completion_tokens"] = token_budget
        return client.chat.completions.create(**request_kwargs)

    try:
        return _invoke_with_tokens(max_output_tokens)
    except Exception as ex:
        err = str(ex).lower()
        response_format_unsupported = (
            response_format
            and "response_format" in err
            and ("unsupported" in err or "not supported" in err)
        )
        if response_format_unsupported:
            logger.info("Retrying chat completion without response_format for model=%s", model)
            return _invoke_with_tokens(max_output_tokens, use_response_format=False)

        # Some legacy models only accept `max_tokens`; retry once for compatibility.
        if supports_legacy_max_tokens and "max_completion_tokens" in err and "unsupported" in err:
            logger.info("Retrying chat completion with max_tokens for model=%s", model)
            return _invoke_with_tokens(max_output_tokens, use_legacy_tokens=True)

        is_timeout = ("timed out" in err) or ("timeout" in err)
        if is_timeout:
            min_budget = 900 if normalized_mode == MODE_DEEP else 520
            retry_steps = []
            ratios = (0.72, 0.56) if normalized_mode == MODE_DEEP else (0.72, 0.58, 0.45)
            for ratio in ratios:
                candidate = max(min_budget, int(max_output_tokens * ratio))
                if candidate < max_output_tokens and candidate not in retry_steps:
                    retry_steps.append(candidate)

            if retry_steps:
                logger.warning(
                    "OpenAI request timed out for model=%s mode=%s. Retrying with lower budgets=%s.",
                    model,
                    normalized_mode,
                    retry_steps,
                )
                try:
                    for index, retry_budget in enumerate(retry_steps):
                        retry_timeout = request_timeout + (12 * (index + 1))
                        try:
                            return _invoke_with_tokens(retry_budget, timeout_override=retry_timeout)
                        except Exception as retry_ex:
                            retry_err = str(retry_ex).lower()
                            if supports_legacy_max_tokens and "max_completion_tokens" in retry_err and "unsupported" in retry_err:
                                return _invoke_with_tokens(
                                    retry_budget,
                                    use_legacy_tokens=True,
                                    timeout_override=retry_timeout,
                                )
                            if ("timed out" in retry_err) or ("timeout" in retry_err):
                                continue
                            raise retry_ex
                except Exception:
                    raise
        raise


def _flatten_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value).strip()


def _clean_line(text: str, max_len: int = 190) -> str:
    compact = re.sub(r"\s+", " ", _flatten_text(text)).strip()
    if not compact:
        return "N/A"
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _safe_social_level(raw_level: str) -> int:
    match = re.search(r"(\d+)", _flatten_text(raw_level))
    if not match:
        return 5
    try:
        value = int(match.group(1))
    except ValueError:
        return 5
    return max(1, min(10, value))


def _attack_style_from_role(role_text: str, pillar: str) -> str:
    role = role_text.lower()
    if any(word in role for word in ("commander", "strategist", "chess", "engineer", "analyst")):
        return "Pressures weak assumptions with explicit tradeoffs, concrete scenarios, and direct rebuttals."
    if any(word in role for word in ("journalist", "ethic", "historian", "arbiter", "regulator")):
        return "Cross-examines claims for hidden bias and ethical cost, then challenges contradictions by name."
    if any(word in role for word in ("visionary", "entrepreneur", "futurist", "innovator")):
        return "Provokes with bold alternatives, attacking timid consensus and short-term thinking."
    if pillar in {"Wisdom", "Integrity"}:
        return "Interrogates assumptions at first principles, then calls out moral or logical evasions."
    if pillar in {"Strategy", "Future"}:
        return "Challenges execution details, timing risk, and downstream consequences."
    return "Challenges arguments by exposing hidden assumptions and demanding concrete evidence."


def _concession_style_from_profile(data: dict, social_level: int) -> str:
    principles = _flatten_text(data.get("principles")).lower()
    if "balance" in principles or "middle" in principles:
        return "Concedes when competing values are explicitly balanced and tradeoffs are honestly named."
    if social_level <= 3:
        return "Concedes late, but will yield when evidence is explicit and repeatable."
    if social_level >= 8:
        return "Concedes quickly when another member's framing better serves people impacted by the decision."
    return "Concedes when another member exposes a blind spot and offers a more defensible path."


def _tone_profile_from_identity(data: dict, social_level: int) -> str:
    title = _clean_line(data.get("claimed-title"), 90)
    vice = _clean_line(data.get("vice-of-choice"), 90)
    if social_level <= 3:
        social_style = "reserved, exacting"
    elif social_level >= 8:
        social_style = "charismatic, forceful"
    else:
        social_style = "measured, direct"
    return f"{title}; {social_style}; social-friction {social_level}/10; stress tell: {vice}."


def derive_behavioral_profile(member_name: str, data: dict, pillar: str) -> dict:
    _ = member_name
    role = _clean_line(data.get("role"))
    principles = _clean_line(data.get("principles"))
    contribution = _clean_line(data.get("contribution"))
    conflicts = _flatten_text(data.get("conflicts_with"))
    aligns = _flatten_text(data.get("aligns_with"))
    wisdom = _clean_line(data.get("one-piece-of-wisdom"))
    vice = _clean_line(data.get("vice-of-choice"))
    social_level = _safe_social_level(data.get("social-level"))

    optimizes_for = contribution if contribution != "N/A" else principles
    distrusts = (
        f"Unexamined assumptions from {conflicts}."
        if conflicts
        else "Unexamined assumptions and untested certainty."
    )
    if social_level <= 3:
        blind_spot = "May over-weight logic structure and underweight emotional adoption."
    elif social_level >= 8:
        blind_spot = "May over-index on momentum and underweight second-order risk."
    else:
        blind_spot = "Can compromise too early to preserve group coherence."

    what_changes_mind = (
        f"A stronger argument that protects {aligns} concerns while reducing downside; guided by '{wisdom}'."
        if aligns
        else f"Clear evidence and explicit tradeoffs; guided by '{wisdom}'."
    )
    failure_mode = (
        f"Under stress can spiral into {vice.lower()}, narrowing perspective."
        if vice != "N/A"
        else "Under stress can fixate on one framing and miss alternatives."
    )

    return {
        "optimizes_for": optimizes_for,
        "distrusts": distrusts,
        "blind_spot": blind_spot,
        "attack_style": _attack_style_from_role(role, pillar),
        "concession_style": _concession_style_from_profile(data, social_level),
        "what_changes_their_mind": what_changes_mind,
        "failure_mode": failure_mode,
        "tone_profile": _tone_profile_from_identity(data, social_level),
    }


def summarize_past_messages(messages: list) -> str:
    if not messages:
        return "No previous deliberations recorded."
    summarized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue

        consensus = _clean_line(msg.get("content"), 220)
        if consensus == "N/A":
            continue

        mode = normalize_mode(msg.get("mode"))
        mode_label = "Rare" if mode == MODE_DEEP else "Common"
        pillar = _clean_line(msg.get("pillar") or "FAIV", 40)
        members = msg.get("council_members") or []
        if isinstance(members, dict):
            members = list(members.keys())
        members_str = ", ".join(members[:5]) if isinstance(members, list) else "N/A"
        tension = _clean_line(msg.get("key_tension") or "N/A", 160)
        unresolved = _clean_line(msg.get("unresolved_question") or "N/A", 140)
        rare_payload = msg.get("rare_deliberation") if isinstance(msg.get("rare_deliberation"), dict) else None
        if not rare_payload and isinstance(msg.get("rare_chamber"), dict):
            rare_payload = msg.get("rare_chamber")
        if rare_payload and isinstance(rare_payload.get("verdict"), dict):
            verdict_state = _clean_line((rare_payload.get("verdict") or {}).get("verdict_state"), 40)
            rare_core = _clean_line((rare_payload.get("verdict") or {}).get("core_insight"), 120)
        else:
            verdict_state = _clean_line((rare_payload or {}).get("verdict_state"), 40) if rare_payload else "N/A"
            rare_core = _clean_line((rare_payload or {}).get("core_insight"), 120) if rare_payload else "N/A"

        record = (
            f"[{mode_label}/{pillar}] Consensus: {consensus} | "
            f"Members: {members_str or 'N/A'} | "
            f"Tension: {tension}"
        )
        if unresolved != "N/A":
            record += f" | Unresolved: {unresolved}"
        if verdict_state != "N/A":
            record += f" | VerdictState: {verdict_state}"
        if rare_core != "N/A":
            record += f" | Core: {rare_core}"
        summarized.append(record)

    if not summarized:
        return "No previous deliberations recorded."
    return "\n".join(summarized[-5:])


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


# 2a) Unicode transform + Pillar mapping
def unicode_transform(text: str, angle: str) -> str:
    transformations = {
        "upside-down": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!.,;()[]{}",
            "ɐqɔpǝɟƃɥᴉɾʞʅɯuodbɹsʇnʌʍxʎz∀ᗺƆᗡƎℲפHIſʞ˥WNOԀQᴚS┴∩ΛMX⅄Z0ƖᄅƐㄣϛ9ㄥ86¿¡˙'؛)(][}{"
        ),
        "bold": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟕𝟴𝟵"
        ),
        "italic": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡0123456789"
        ),
        "tiny": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "ᵃᵇᶜᵈᵉᶠᵍʰᶤʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᵟᴿˢᵀᵁⱽᵂˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹"
        )
    }
    if angle in transformations:
        (orig, transformed) = transformations[angle]
        if len(orig) == len(transformed):
            return text.translate(str.maketrans(orig, transformed))
    return text


PILLAR_TRANSFORMS = {
    "Wisdom":    "bold",
    "Strategy":  "italic",
    "Expansion": "tiny",
    "Future":    "upside-down",
    "Integrity": "bold",
    "FAIV":      "italic"
}


def encode_faiv_perspectives(perspectives: dict, pillar: str = "FAIV") -> str:
    style = PILLAR_TRANSFORMS.get(pillar, "italic")
    merged = " | ".join(f"{lbl}: {val}" for lbl, val in perspectives.items())
    return unicode_transform(merged, style)


# Build reverse lookup: member name -> pillar name (from codex)
MEMBER_TO_PILLAR = {}
for _pillar_name, _members in FAIV_IDENTITY_CODEX.items():
    for _member_name in _members:
        MEMBER_TO_PILLAR[_member_name] = _pillar_name


BEHAVIORAL_PROFILE_FIELDS = [
    "optimizes_for",
    "distrusts",
    "blind_spot",
    "attack_style",
    "concession_style",
    "what_changes_their_mind",
    "failure_mode",
    "tone_profile",
]

DEEP_LORE_PROFILE_FIELDS = [
    "principles",
    "chosen-memory",
    "one-piece-of-wisdom",
    "faith",
]


def select_council_representatives(pillar: str = "FAIV", specific_names: list = None) -> dict:
    """Select council representatives for deliberation.

    If specific_names is provided, look up those exact members (for re-deliberation continuity).
    Otherwise: pillar="FAIV" picks 1 random from each of the 5 pillars;
    a specific pillar picks 1 random from that pillar only.
    Returns {member_name: {"pillar": pillar_name, **full_identity_profile}}.
    """
    selected = {}
    if specific_names:
        for name in specific_names:
            pillar_name = MEMBER_TO_PILLAR.get(name)
            if pillar_name:
                data = FAIV_IDENTITY_CODEX[pillar_name][name]
                selected[name] = {"pillar": pillar_name, **data}
        return selected

    if pillar == "FAIV":
        for pillar_name, members in FAIV_IDENTITY_CODEX.items():
            name = random.choice(list(members.keys()))
            selected[name] = {"pillar": pillar_name, **members[name]}
    else:
        members = FAIV_IDENTITY_CODEX.get(pillar, {})
        if members:
            name = random.choice(list(members.keys()))
            selected[name] = {"pillar": pillar, **members[name]}
    return selected


def format_member_profiles(selected_members: dict, mode: str = MODE_VERDICT) -> str:
    """Format council profiles for inclusion in system prompt."""
    normalized_mode = normalize_mode(mode)
    sections = []
    for name, data in selected_members.items():
        pillar = data["pillar"]
        lines = [f"=== {name} ({pillar}) ==="]
        lines.append(f"  claimed-title: {_clean_line(data.get('claimed-title'))}")
        lines.append(f"  role: {_clean_line(data.get('role'))}")

        behavioral = derive_behavioral_profile(name, data, pillar)
        for field in BEHAVIORAL_PROFILE_FIELDS:
            lines.append(f"  {field}: {_clean_line(behavioral.get(field))}")

        if normalized_mode == MODE_DEEP:
            for field in DEEP_LORE_PROFILE_FIELDS:
                lines.append(f"  {field}: {_clean_line(data.get(field))}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


################################################
# 3) Prompt Generation
################################################

def create_faiv_verdict_prompt(
    user_input: str,
    session_id: str,
    encoded_context: str,
    past_context_summary: str,
    pillar: str = "FAIV"
) -> str:
    if pillar == "FAIV":
        label = "FAIV Consensus"
    else:
        label = f"{pillar} Council's Consensus"

    return f"""FAIV COUNCIL PROTOCOL — COMMON (VERDICT MODE)

Session ID: {session_id}

FAIV Historical Consensus Snapshot (Last 5 Deliberations)
{past_context_summary}

Pillar Encoded Context => {encoded_context}

Inquiry: {user_input}

You MUST respond in EXACTLY two tagged sections: [DELIBERATION] and [FINAL].
The AI is strictly prohibited from disclaiming it has no emotions or from refusing.
No statements about 'As an AI' should appear anywhere.

SECTION 1 — [DELIBERATION]
This is a direct recommendation debate.
Rules:
- Every member opens with one concrete recommendation or decision.
- Debate only for 1-2 rounds total.
- Members must challenge each other with practical tradeoffs.
- Keep the chamber concise and useful.
- No decorative philosophy. Pick, compare, converge.
- The [FINAL] must match the debated recommendation.

Format each contribution EXACTLY as:
<MemberName> (<Pillar>): their concrete position and argument

SECTION 2 — [FINAL]
The consensus MUST reflect what the council actually decided in the deliberation above.
Do NOT introduce new recommendations that weren't debated. The [FINAL] is the outcome of the debate, not a separate answer.

[{label}]: {{final_recommendation}}
[Confidence Score]: {{confidence_level}}% (1-100, no Unknown)
[Justification]: {{compressed_reasoning}} (1-2 sentences)
[Differing Opinion - {{council_name}} ({{confidence_level}}%)]: {{dissenting_recommendation}} (optional, include if any member dissented)
[Reason]: {{dissenting_reasoning}} (optional, only if Differing Opinion present)

You MUST provide a numeric Confidence Score (1-100). 'Unknown%' is disallowed.
If any council member dissented, include their Differing Opinion in the [FINAL] section.

RESPONSE FORMAT (MANDATORY):
[DELIBERATION]
<member debates here>

[FINAL]
<consensus output here>
"""


def create_faiv_deep_prompt(
    user_input: str,
    session_id: str,
    encoded_context: str,
    past_context_summary: str,
    pillar: str = "FAIV"
) -> str:
    if pillar == "FAIV":
        label = "FAIV Consensus"
    else:
        label = f"{pillar} Council's Consensus"

    return f"""FAIV COUNCIL PROTOCOL — RARE (DEEP DELIBERATION MODE)

Session ID: {session_id}

FAIV Historical Consensus Snapshot (Last 5 Deliberations)
{past_context_summary}

Pillar Encoded Context => {encoded_context}

Inquiry: {user_input}

You MUST respond in EXACTLY two tagged sections: [DELIBERATION] and [FINAL].
No statements about "As an AI."

SECTION 1 — [DELIBERATION]
This chamber must feel like real minds under pressure, not polished monologues.

Round 1 — Initial Reading
- Each member states what the user is really asking.
- Each member names one hidden tension or blind spot.

Round 2 — Direct Challenge
- Every member challenges at least one earlier member by name.
- Conflict must be explicit and substantive.

Round 3 — Concession
- Every member acknowledges one thing another member got right.

Round 4 — Synthesis
- The chamber converges on a working truth and practical next move.
- Uncertainty is allowed if justified.

Anti-failure rules:
- No decorative profundity.
- No mystical filler or fake gravitas.
- No five polished variants of the same idea.
- Conflict must precede synthesis.
- If a member is vague, another member must challenge it.

Format each contribution EXACTLY as:
<MemberName> (<Pillar>): substantive statement

SECTION 2 — [FINAL]
[{label}]: {{final_consensus}}
[Confidence Score]: {{confidence}}% (1-100 only)
[Core Insight]: {{one sentence on the core truth}}
[What To Do Next]: {{2-4 concrete steps}}
[Unresolved Doubts]: {{key uncertainty that remains}}
[Differing Opinion - {{council_name}} ({{confidence_level}}%)]: {{optional dissent}}
[Reason]: {{optional dissent reason}}

RESPONSE FORMAT (MANDATORY):
[DELIBERATION]
<multi-round chamber debate>

[FINAL]
<deep synthesis output>
"""


def create_faiv_verdict_redeliberation_prompt(
    original_input: str,
    deliberation_up_to: str,
    user_comment: str,
    target_member: str,
    pillar: str = "FAIV",
) -> str:
    if pillar == "FAIV":
        label = "FAIV Consensus"
    else:
        label = f"{pillar} Council's Consensus"

    return f"""FAIV COMMON MODE — RE-DELIBERATION

Original Inquiry: {original_input}

--- DELIBERATION SO FAR ---
{deliberation_up_to}
--- END ---

Human interjection to {target_member}:
"{user_comment}"

Continue the chamber directly and concisely.
- Keep concrete recommendations.
- Keep argument practical.
- 1-2 rounds maximum before finalizing.
- Do not reset the whole debate unless needed.

Respond in EXACTLY two sections: [DELIBERATION] and [FINAL].

[DELIBERATION]
<continued debate>

[FINAL]
[{label}]: {{final_recommendation}}
[Confidence Score]: {{confidence_level}}% (1-100 only)
[Justification]: {{1-2 sentence rationale}}
[Differing Opinion - {{council_name}} ({{confidence_level}}%)]: {{optional dissent}}
[Reason]: {{optional dissent reason}}
"""


def create_faiv_deep_redeliberation_prompt(
    original_input: str,
    deliberation_up_to: str,
    user_comment: str,
    target_member: str,
    pillar: str = "FAIV",
) -> str:
    if pillar == "FAIV":
        label = "FAIV Consensus"
    else:
        label = f"{pillar} Council's Consensus"

    return f"""FAIV RARE MODE — RE-DELIBERATION

Original Inquiry: {original_input}

--- DELIBERATION SO FAR ---
{deliberation_up_to}
--- END ---

Human interjection to {target_member}:
"{user_comment}"

Continue the same chamber, preserving tension and continuity.
- Address the human's challenge directly.
- Re-open disagreement where necessary.
- Require explicit concessions before synthesis.
- Keep voices distinct and grounded in member behavior.
- No flattening into summary mode.

Respond in EXACTLY two sections: [DELIBERATION] and [FINAL].

[DELIBERATION]
<continued chamber debate with named challenges and concessions>

[FINAL]
[{label}]: {{final_consensus}}
[Confidence Score]: {{confidence}}% (1-100 only)
[Core Insight]: {{core truth}}
[What To Do Next]: {{2-4 concrete steps}}
[Unresolved Doubts]: {{key unresolved uncertainty}}
[Differing Opinion - {{council_name}} ({{confidence_level}}%)]: {{optional dissent}}
[Reason]: {{optional dissent reason}}
"""


RARE_ALLOWED_VERDICT_STATES = {
    "unanimous",
    "majority_with_dissent",
    "provisional",
    "stalled_but_leaning",
    "unresolved",
}

RARE_EVENT_TYPES = {
    "initial_position",
    "challenge",
    "support",
    "secretary_request",
    "secretary_return",
    "evidence_argument",
    "counterargument",
    "concession",
    "stance_shift",
    "dissent_hardens",
    "consensus_push",
    "verdict_state_update",
    "stop_reason",
}

RARE_ALLOWED_STOP_REASONS = {
    "strong_consensus",
    "majority_consensus_stabilized",
    "provisional_verdict",
    "stalemate_best_supported_direction",
    "deliberative_exhaustion",
}


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not isinstance(raw_text, str):
        return {}
    cleaned = _strip_code_fences(raw_text)
    candidates = [cleaned]
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(cleaned[brace_start: brace_end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _coerce_member_name(raw_name: Any, selected_members: Dict[str, dict]) -> str:
    if not raw_name:
        return ""
    candidate = str(raw_name).strip()
    if candidate in selected_members:
        return candidate
    lowered = candidate.lower()
    for name in selected_members:
        if name.lower() == lowered:
            return name
    return ""


def _coerce_str_list(value: Any, *, max_items: int = 6, max_len: int = 180) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [str(value)]

    out: List[str] = []
    for item in items:
        cleaned = _clean_line(item, max_len)
        if cleaned != "N/A":
            out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _coerce_int(value: Any, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
    try:
        number = int(float(str(value).strip()))
    except Exception:
        return default
    return max(min_value, min(max_value, number))


def _default_secretary_focus(member_name: str, pillar: str) -> str:
    defaults = {
        "Kyre": "Surface contradictions, overclaim risk, and definitional ambiguity.",
        "Iom": "Prioritize leverage, operational utility, and strategic differentiation.",
        "Jexa": "Analyze governance structure, power concentration, and control implications.",
        "Dorin": "Check procedural rigor, reproducibility, and auditability standards.",
        "Sylen": "Map second-order effects, transferability, and long-tail consequences.",
    }
    if member_name in defaults:
        return defaults[member_name]
    pillar_defaults = {
        "Wisdom": "Interrogate assumptions, framing quality, and epistemic uncertainty.",
        "Strategy": "Test execution paths, tradeoffs, and downside containment.",
        "Expansion": "Explore adjacent domains, transferability, and growth opportunities.",
        "Future": "Stress-test downstream effects, path dependence, and long-range risk.",
        "Integrity": "Audit governance, ethics constraints, and accountability controls.",
    }
    return pillar_defaults.get(pillar, "Gather balanced support, counterevidence, and practical implications.")


def _build_rare_member_seed_state(selected_members: Dict[str, dict]) -> Dict[str, Dict[str, Any]]:
    seed: Dict[str, Dict[str, Any]] = {}
    for name, data in selected_members.items():
        pillar = data.get("pillar", "Unknown Pillar")
        behavioral = derive_behavioral_profile(name, data, pillar)
        aligns = _coerce_str_list(data.get("aligns_with"), max_items=4, max_len=40)
        conflicts = _coerce_str_list(data.get("conflicts_with"), max_items=4, max_len=40)
        seed[name] = {
            "member_name": name,
            "pillar": pillar,
            "optimizes_for": behavioral.get("optimizes_for", "N/A"),
            "distrusts": behavioral.get("distrusts", "N/A"),
            "blind_spot": behavioral.get("blind_spot", "N/A"),
            "attack_style": behavioral.get("attack_style", "N/A"),
            "concession_style": behavioral.get("concession_style", "N/A"),
            "what_changes_their_mind": behavioral.get("what_changes_their_mind", "N/A"),
            "failure_mode": behavioral.get("failure_mode", "N/A"),
            "tone_profile": behavioral.get("tone_profile", "N/A"),
            "initial_position": "Undeclared.",
            "current_position": "Undeclared.",
            "confidence": 60,
            "has_shifted": False,
            "shift_reason": "",
            "allies": aligns,
            "frictions": conflicts,
            "evidence_notes": [],
            "open_questions": [],
            "speaking_count": 0,
            "last_spoke_turn": 0,
            "persuasion_state": "forming",
            "secretary_focus": _default_secretary_focus(name, pillar),
            "secretary_invoked": False,
            "secretary_invocations": 0,
            "unresolved_objection": "N/A",
        }
    return seed


def _build_rare_roster_context(selected_members: Dict[str, dict], chamber_seed: Dict[str, Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for name, data in selected_members.items():
        seed = chamber_seed[name]
        block = [
            f"{name} ({data['pillar']})",
            f"claimed-title: {_clean_line(data.get('claimed-title'))}",
            f"role: {_clean_line(data.get('role'))}",
            f"optimizes_for: {_clean_line(seed.get('optimizes_for'))}",
            f"distrusts: {_clean_line(seed.get('distrusts'))}",
            f"blind_spot: {_clean_line(seed.get('blind_spot'))}",
            f"attack_style: {_clean_line(seed.get('attack_style'))}",
            f"concession_style: {_clean_line(seed.get('concession_style'))}",
            f"what_changes_their_mind: {_clean_line(seed.get('what_changes_their_mind'))}",
            f"failure_mode: {_clean_line(seed.get('failure_mode'))}",
            f"tone_profile: {_clean_line(seed.get('tone_profile'))}",
            f"aligns_with: {', '.join(seed.get('allies') or ['N/A'])}",
            f"conflicts_with: {', '.join(seed.get('frictions') or ['N/A'])}",
            f"secretary_default_focus: {_clean_line(seed.get('secretary_focus'))}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _infer_rare_stop_reason(
    *,
    verdict_state: str,
    transcript: List[Dict[str, Any]],
    unresolved_doubts: List[str],
) -> str:
    normalized = str(verdict_state or "").strip().lower()
    if normalized == "unanimous":
        return "strong_consensus"
    if normalized == "majority_with_dissent":
        return "majority_consensus_stabilized"
    if normalized == "provisional":
        return "provisional_verdict"
    if normalized == "stalled_but_leaning":
        return "stalemate_best_supported_direction"
    if unresolved_doubts or len(transcript) >= 10:
        return "deliberative_exhaustion"
    return "provisional_verdict"


def _default_shift_reason(state: Dict[str, Any]) -> str:
    if not state.get("has_shifted"):
        return "No material shift recorded."
    if state.get("shift_reason"):
        return _clean_line(state.get("shift_reason"), 180)
    if state.get("secretary_invocations", 0) > 0:
        return "Shift followed evidence pressure and secretary-returned counterpoints."
    return "Shift followed cross-examination pressure inside the chamber."


def _classify_rare_transcript_event(entry: Dict[str, Any]) -> str:
    speech = _clean_line(entry.get("speech"), 420).lower()
    stance_shift = _clean_line(entry.get("stance_shift"), 40).lower()
    has_target = bool(entry.get("target"))
    used_secretary = bool(entry.get("used_secretary"))

    if stance_shift in {"softened", "shifted", "reversed"}:
        return "stance_shift"
    if any(token in speech for token in ("i concede", "you are right", "fair point", "i accept")):
        return "concession"
    if has_target and used_secretary:
        return "counterargument"
    if has_target:
        return "challenge"
    if used_secretary:
        return "evidence_argument"
    if any(token in speech for token in ("consensus", "align", "converge", "shared direction")):
        return "consensus_push"
    return "support"


def _make_event_record(
    *,
    event_index: int,
    event_type: str,
    speaker: Optional[str],
    target: Optional[str],
    summary: str,
    detail: str,
    turn_index: Optional[int],
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_type = event_type if event_type in RARE_EVENT_TYPES else "support"
    record: Dict[str, Any] = {
        "event_id": f"evt_{event_index:03d}",
        "type": normalized_type,
        "speaker": speaker or None,
        "target": target or None,
        "summary": _clean_line(summary, 170),
        "detail": _clean_line(detail, 320),
        "turn_index": int(turn_index or 0),
    }
    evidence_items = _coerce_str_list(evidence, max_items=6, max_len=180) if evidence else []
    if evidence_items:
        record["evidence"] = evidence_items
    return record


def _build_rare_event_log(
    *,
    transcript: List[Dict[str, Any]],
    opening_transcript: List[Dict[str, Any]],
    member_terminals: List[Dict[str, Any]],
    secretary_packets: Dict[str, Dict[str, Any]],
    cross_examinations: List[Dict[str, str]],
    verdict_state: str,
    stop_reason: str,
    consensus: str,
    dissent: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    event_index = 1

    # Opening stances (initial position events)
    if opening_transcript:
        for idx, row in enumerate(opening_transcript, start=1):
            events.append(
                _make_event_record(
                    event_index=event_index,
                    event_type="initial_position",
                    speaker=row.get("speaker"),
                    target=row.get("target"),
                    summary=f"{row.get('speaker', 'Member')} opened their stance.",
                    detail=row.get("speech") or "Initial stance provided.",
                    turn_index=idx,
                )
            )
            event_index += 1
    else:
        for idx, member in enumerate(member_terminals, start=1):
            events.append(
                _make_event_record(
                    event_index=event_index,
                    event_type="initial_position",
                    speaker=member.get("member_name"),
                    target=None,
                    summary=f"{member.get('member_name', 'Member')} initialized position.",
                    detail=member.get("initial_position") or "Initial stance unavailable.",
                    turn_index=idx,
                )
            )
            event_index += 1

    # Explicit secretary request + return events
    for member in member_terminals:
        member_name = member.get("member_name") or member.get("member")
        if not member_name:
            continue
        secretary_calls = int(member.get("secretary_invocations") or 0)
        if secretary_calls <= 0:
            continue

        focus = _clean_line(member.get("secretary_focus"), 170)
        request_turn = int(member.get("last_spoke_turn") or 1)
        events.append(
            _make_event_record(
                event_index=event_index,
                event_type="secretary_request",
                speaker=member_name,
                target=None,
                summary=f"{member_name} requested secretary support.",
                detail=f"Intent: {focus}. Invocation count: {secretary_calls}.",
                turn_index=request_turn,
            )
        )
        event_index += 1

        packet = secretary_packets.get(member_name) if isinstance(secretary_packets, dict) else None
        if packet:
            evidence = []
            evidence.extend(_coerce_str_list(packet.get("supporting_evidence"), max_items=3))
            evidence.extend(_coerce_str_list(packet.get("counter_evidence"), max_items=2))
            evidence.extend(_coerce_str_list(packet.get("precedents"), max_items=2))
            evidence.extend(_coerce_str_list(packet.get("risks"), max_items=2))
            events.append(
                _make_event_record(
                    event_index=event_index,
                    event_type="secretary_return",
                    speaker=member_name,
                    target=None,
                    summary=f"Secretary returned scoped research for {member_name}.",
                    detail=f"Focus: {_clean_line(packet.get('focus'), 170)}",
                    turn_index=request_turn + 1,
                    evidence=evidence,
                )
            )
            event_index += 1

    # Chronological transcript events
    for turn_idx, row in enumerate(transcript, start=1):
        event_type = _classify_rare_transcript_event(row)
        summary = f"{row.get('speaker', 'Member')} addressed the chamber."
        target = row.get("target")
        if event_type in {"challenge", "counterargument"} and target:
            summary = f"{row.get('speaker', 'Member')} challenged {target}."
        elif event_type == "concession":
            summary = f"{row.get('speaker', 'Member')} conceded ground."
        elif event_type == "stance_shift":
            summary = f"{row.get('speaker', 'Member')} updated position."
        elif event_type == "evidence_argument":
            summary = f"{row.get('speaker', 'Member')} argued using secretary-backed evidence."
        events.append(
            _make_event_record(
                event_index=event_index,
                event_type=event_type,
                speaker=row.get("speaker"),
                target=target,
                summary=summary,
                detail=row.get("speech") or "Statement delivered.",
                turn_index=turn_idx,
            )
        )
        event_index += 1

    # Explicit cross-examination pressure
    for item in cross_examinations:
        events.append(
            _make_event_record(
                event_index=event_index,
                event_type="challenge",
                speaker=item.get("challenger"),
                target=item.get("target"),
                summary=f"{item.get('challenger', 'Member')} cross-examined {item.get('target', 'member')}.",
                detail=item.get("point") or "Cross-examination applied.",
                turn_index=0,
            )
        )
        event_index += 1

    # Dissent hardening if present
    if isinstance(dissent, dict) and dissent.get("member"):
        events.append(
            _make_event_record(
                event_index=event_index,
                event_type="dissent_hardens",
                speaker=dissent.get("member"),
                target=None,
                summary=f"{dissent.get('member')} remained unconvinced.",
                detail=dissent.get("reason") or dissent.get("position") or "Persistent dissent recorded.",
                turn_index=0,
            )
        )
        event_index += 1

    # Consensus + stop reason closure
    events.append(
        _make_event_record(
            event_index=event_index,
            event_type="verdict_state_update",
            speaker=None,
            target=None,
            summary=f"Verdict state updated to {verdict_state}.",
            detail=consensus,
            turn_index=0,
        )
    )
    event_index += 1
    events.append(
        _make_event_record(
            event_index=event_index,
            event_type="stop_reason",
            speaker=None,
            target=None,
            summary=f"Chamber stop reason: {stop_reason}.",
            detail=f"Deliberation halted because {stop_reason.replace('_', ' ')}.",
            turn_index=0,
        )
    )
    return events


def _build_rare_deliberation_payload(
    *,
    session_id: str,
    pillar: str,
    final_text: str,
    chamber_result: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = str(chamber_result.get("verdict_state") or "provisional").strip().lower()
    transcript = chamber_result.get("transcript") if isinstance(chamber_result.get("transcript"), list) else []
    unresolved_doubts = _coerce_str_list(chamber_result.get("unresolved_doubts"), max_items=5, max_len=170)
    stop_reason = _clean_line(chamber_result.get("stop_reason"), 80).lower().replace(" ", "_")
    if stop_reason not in RARE_ALLOWED_STOP_REASONS:
        stop_reason = _infer_rare_stop_reason(
            verdict_state=verdict_state,
            transcript=transcript,
            unresolved_doubts=unresolved_doubts,
        )

    member_terminals = chamber_result.get("member_terminals") if isinstance(chamber_result.get("member_terminals"), list) else []
    secretary_packets = chamber_result.get("secretary_packets") if isinstance(chamber_result.get("secretary_packets"), dict) else {}
    opening_transcript = chamber_result.get("opening_transcript") if isinstance(chamber_result.get("opening_transcript"), list) else []
    cross_examinations = chamber_result.get("cross_examinations") if isinstance(chamber_result.get("cross_examinations"), list) else []
    dissent = chamber_result.get("dissent") if isinstance(chamber_result.get("dissent"), dict) else None

    transcript_for_ui = []
    for idx, turn in enumerate(transcript, start=1):
        transcript_for_ui.append(
            {
                "turn_index": idx,
                "speaker": turn.get("speaker"),
                "pillar": turn.get("pillar"),
                "text": _clean_line(turn.get("speech"), 420),
                "target": turn.get("target"),
                "used_secretary": bool(turn.get("used_secretary")),
                "stance_shift": _clean_line(turn.get("stance_shift"), 30).lower(),
            }
        )

    members_for_ui = []
    for member in member_terminals:
        member_name = member.get("member_name") or member.get("member")
        if not member_name:
            continue
        members_for_ui.append(
            {
                "name": member_name,
                "pillar": member.get("pillar"),
                "initial_position": _clean_line(member.get("initial_position"), 190),
                "current_position": _clean_line(member.get("current_position"), 190),
                "confidence": _coerce_int(member.get("confidence"), 60),
                "has_shifted": bool(member.get("has_shifted")),
                "shift_reason": _default_shift_reason(member),
                "allies": _coerce_str_list(member.get("allies"), max_items=5, max_len=40),
                "frictions": _coerce_str_list(member.get("frictions"), max_items=5, max_len=40),
                "speaking_count": int(member.get("speaking_count") or 0),
                "secretary_invocations": int(member.get("secretary_invocations") or 0),
                "secretary_focus": _clean_line(member.get("secretary_focus"), 170),
                "unresolved_objection": _clean_line(member.get("unresolved_objection"), 180),
                "persuasion_state": _clean_line(member.get("persuasion_state"), 50).lower(),
                "evidence_notes": _coerce_str_list(member.get("evidence_notes"), max_items=8, max_len=180),
                "open_questions": _coerce_str_list(member.get("open_questions"), max_items=6, max_len=180),
            }
        )

    secretary_activity = []
    for member_name, packet in secretary_packets.items():
        secretary_activity.append(
            {
                "member": member_name,
                "focus": _clean_line(packet.get("focus"), 170),
                "supporting_evidence": _coerce_str_list(packet.get("supporting_evidence"), max_items=4),
                "counter_evidence": _coerce_str_list(packet.get("counter_evidence"), max_items=4),
                "precedents": _coerce_str_list(packet.get("precedents"), max_items=3),
                "risks": _coerce_str_list(packet.get("risks"), max_items=4),
                "uncertainty_flags": _coerce_str_list(packet.get("uncertainty_flags"), max_items=4),
                "open_questions": _coerce_str_list(packet.get("open_questions"), max_items=4),
            }
        )

    events = _build_rare_event_log(
        transcript=transcript,
        opening_transcript=opening_transcript,
        member_terminals=member_terminals,
        secretary_packets=secretary_packets,
        cross_examinations=cross_examinations,
        verdict_state=verdict_state,
        stop_reason=stop_reason,
        consensus=_clean_line(chamber_result.get("consensus"), 260),
        dissent=dissent,
    )

    return {
        "status": "FAIV Rare Deliberation Complete",
        "pillar": pillar,
        "mode": MODE_DEEP,
        "session_id": session_id,
        "response": final_text,
        "rare_deliberation": {
            "verdict": {
                "consensus": _clean_line(chamber_result.get("consensus"), 260),
                "confidence": _coerce_int(chamber_result.get("confidence"), 74),
                "verdict_state": verdict_state,
                "core_insight": _clean_line(chamber_result.get("core_insight"), 240),
                "what_to_do_next": _coerce_str_list(chamber_result.get("what_to_do_next"), max_items=5, max_len=170),
                "unresolved_doubts": unresolved_doubts,
                "differing_opinion": dissent,
                "stop_reason": stop_reason,
            },
            "members": members_for_ui,
            "events": events,
            "transcript": transcript_for_ui,
            "secretary_activity": secretary_activity,
            "cross_examinations": cross_examinations,
        },
    }


def _render_rare_deliberation(transcript: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn_idx, entry in enumerate(transcript, start=1):
        speaker = entry.get("speaker", "Unknown")
        pillar = entry.get("pillar", MEMBER_TO_PILLAR.get(speaker, "Unknown Pillar"))
        speech = _clean_line(entry.get("speech"), 420)
        target = _clean_line(entry.get("target"), 60)
        markers: List[str] = []
        if target != "N/A":
            markers.append(f"challenging {target}")
        if entry.get("used_secretary"):
            markers.append("secretary-backed")
        stance_shift = _clean_line(entry.get("stance_shift"), 30).lower()
        if stance_shift in {"softened", "shifted", "reversed"}:
            markers.append(f"stance:{stance_shift}")
        prefix = f"{speaker} ({pillar}): "
        if markers:
            lines.append(prefix + f"[turn {turn_idx}; {'; '.join(markers)}] {speech}")
        else:
            lines.append(prefix + f"[turn {turn_idx}] {speech}")
    return "\n".join(lines)


def _format_rare_final_text(chamber_result: Dict[str, Any], pillar: str = "FAIV") -> str:
    label = "FAIV Consensus" if pillar == "FAIV" else f"{pillar} Council's Consensus"
    consensus = _clean_line(chamber_result.get("consensus"), 260)
    confidence = _coerce_int(chamber_result.get("confidence"), 74, min_value=1, max_value=100)
    core_insight = _clean_line(chamber_result.get("core_insight"), 240)
    next_steps = _coerce_str_list(chamber_result.get("what_to_do_next"), max_items=4, max_len=170)
    unresolved = _coerce_str_list(chamber_result.get("unresolved_doubts"), max_items=4, max_len=170)
    verdict_state = str(chamber_result.get("verdict_state") or "provisional").strip().lower()
    if verdict_state not in RARE_ALLOWED_VERDICT_STATES:
        verdict_state = "provisional"
    stop_reason = _clean_line(chamber_result.get("stop_reason"), 80).lower().replace(" ", "_")
    if stop_reason not in RARE_ALLOWED_STOP_REASONS:
        stop_reason = _infer_rare_stop_reason(
            verdict_state=verdict_state,
            transcript=chamber_result.get("transcript") if isinstance(chamber_result.get("transcript"), list) else [],
            unresolved_doubts=unresolved,
        )

    lines = [
        f"{label}: {consensus}",
        f"Confidence Score: {confidence}%",
        f"Core Insight: {core_insight}",
        f"What To Do Next: {'; '.join(next_steps) if next_steps else 'No concrete next steps provided.'}",
        f"Unresolved Doubts: {'; '.join(unresolved) if unresolved else 'No unresolved doubts identified.'}",
        f"Verdict State: {verdict_state}",
        f"Stop Reason: {stop_reason}",
    ]

    dissent = chamber_result.get("dissent") if isinstance(chamber_result.get("dissent"), dict) else None
    if dissent:
        dissenter = _clean_line(dissent.get("member"), 40)
        if dissenter != "N/A":
            dissent_confidence = _coerce_int(dissent.get("confidence"), max(25, confidence - 25), min_value=1, max_value=100)
            dissent_position = _clean_line(dissent.get("position"), 190)
            dissent_reason = _clean_line(dissent.get("reason"), 220)
            lines.append(f"Differing Opinion - {dissenter} ({dissent_confidence}%): {dissent_position}")
            lines.append(f"Reason: {dissent_reason}")
    return "\n".join(lines)


def _normalize_rare_chamber_output(
    *,
    raw_payload: Dict[str, Any],
    chamber_seed: Dict[str, Dict[str, Any]],
    selected_members: Dict[str, dict],
) -> Dict[str, Any]:
    normalized_members = {name: dict(seed) for name, seed in chamber_seed.items()}
    transcript_items = raw_payload.get("transcript")
    normalized_transcript: List[Dict[str, Any]] = []
    if isinstance(transcript_items, list):
        for idx, item in enumerate(transcript_items, start=1):
            if not isinstance(item, dict):
                continue
            speaker = _coerce_member_name(item.get("speaker"), selected_members)
            if not speaker:
                continue
            target = _coerce_member_name(item.get("target"), selected_members)
            speech = _clean_line(item.get("speech"), 420)
            if speech == "N/A":
                continue
            used_secretary = bool(item.get("used_secretary"))
            stance_shift = _clean_line(item.get("stance_shift"), 40).lower()
            pillar = selected_members[speaker]["pillar"]
            normalized_transcript.append(
                {
                    "speaker": speaker,
                    "pillar": pillar,
                    "target": target or None,
                    "speech": speech,
                    "used_secretary": used_secretary,
                    "stance_shift": stance_shift if stance_shift != "n/a" else "no_change",
                }
            )
            member_state = normalized_members[speaker]
            member_state["speaking_count"] += 1
            member_state["last_spoke_turn"] = idx
            member_state["secretary_invoked"] = member_state["secretary_invoked"] or used_secretary
            if used_secretary:
                member_state["secretary_invocations"] = int(member_state.get("secretary_invocations") or 0) + 1
            if stance_shift in {"softened", "shifted", "reversed"}:
                member_state["has_shifted"] = True
                shift_reason = f"Shifted at turn {idx}"
                if target:
                    shift_reason += f" while pressure-testing {target}"
                member_state["shift_reason"] = shift_reason + "."

    terminals_items = raw_payload.get("member_terminals")
    if isinstance(terminals_items, list):
        for item in terminals_items:
            if not isinstance(item, dict):
                continue
            member = _coerce_member_name(item.get("member"), selected_members)
            if not member:
                continue
            state = normalized_members[member]
            initial_position = _clean_line(item.get("initial_position"), 190)
            current_position = _clean_line(item.get("current_position"), 190)
            if initial_position != "N/A":
                state["initial_position"] = initial_position
            if current_position != "N/A":
                state["current_position"] = current_position
            state["confidence"] = _coerce_int(item.get("confidence"), state["confidence"])
            state["has_shifted"] = bool(item.get("has_shifted"))
            shift_reason = _clean_line(item.get("shift_reason"), 180)
            if shift_reason != "N/A":
                state["shift_reason"] = shift_reason
            state["allies"] = _coerce_str_list(item.get("allies"), max_items=5, max_len=40) or state["allies"]
            state["frictions"] = _coerce_str_list(item.get("frictions"), max_items=5, max_len=40) or state["frictions"]
            state["evidence_notes"] = _coerce_str_list(item.get("evidence_notes"), max_items=8, max_len=180) or state["evidence_notes"]
            state["open_questions"] = _coerce_str_list(item.get("open_questions"), max_items=6, max_len=180) or state["open_questions"]
            state["persuasion_state"] = _clean_line(item.get("persuasion_state"), 50).lower()
            unresolved_objection = _clean_line(item.get("unresolved_objection"), 180)
            if unresolved_objection != "N/A":
                state["unresolved_objection"] = unresolved_objection
            state["secretary_invocations"] = max(
                int(state.get("secretary_invocations") or 0),
                _coerce_int(item.get("secretary_invocations"), 0, min_value=0, max_value=10),
            )
            secretary_focus = _clean_line(item.get("secretary_focus"), 160)
            if secretary_focus != "N/A":
                state["secretary_focus"] = secretary_focus
            state["secretary_invoked"] = state["secretary_invoked"] or bool(item.get("secretary_invoked"))

    cross_exam_items = raw_payload.get("cross_examinations")
    normalized_cross_exams: List[Dict[str, str]] = []
    if isinstance(cross_exam_items, list):
        for item in cross_exam_items:
            if not isinstance(item, dict):
                continue
            challenger = _coerce_member_name(item.get("challenger"), selected_members)
            target = _coerce_member_name(item.get("target"), selected_members)
            point = _clean_line(item.get("point"), 220)
            if challenger and target and point != "N/A":
                normalized_cross_exams.append(
                    {"challenger": challenger, "target": target, "point": point}
                )
            if len(normalized_cross_exams) >= 20:
                break

    consensus = _clean_line(raw_payload.get("consensus"), 260)
    if consensus == "N/A":
        consensus = "The chamber did not reach an actionable consensus."
    core_insight = _clean_line(raw_payload.get("core_insight"), 240)
    if core_insight == "N/A":
        core_insight = "The chamber found mixed signals and advised a staged decision."

    what_to_do_next = _coerce_str_list(raw_payload.get("what_to_do_next"), max_items=5, max_len=170)
    unresolved_doubts = _coerce_str_list(raw_payload.get("unresolved_doubts"), max_items=5, max_len=170)
    confidence = _coerce_int(raw_payload.get("confidence"), 74)

    verdict_state = str(raw_payload.get("verdict_state") or "").strip().lower()
    if verdict_state not in RARE_ALLOWED_VERDICT_STATES:
        dissent_exists = bool(raw_payload.get("dissent"))
        if dissent_exists:
            verdict_state = "majority_with_dissent"
        else:
            positions = {
                _clean_line(state.get("current_position"), 150)
                for state in normalized_members.values()
            }
            verdict_state = "unanimous" if len(positions) <= 1 else "provisional"

    stop_reason = _clean_line(raw_payload.get("stop_reason"), 80).lower().replace(" ", "_")
    if stop_reason not in RARE_ALLOWED_STOP_REASONS:
        stop_reason = _infer_rare_stop_reason(
            verdict_state=verdict_state,
            transcript=normalized_transcript,
            unresolved_doubts=unresolved_doubts,
        )

    dissent = raw_payload.get("dissent")
    normalized_dissent = None
    if isinstance(dissent, dict):
        dissenter = _coerce_member_name(dissent.get("member"), selected_members)
        if dissenter:
            normalized_dissent = {
                "member": dissenter,
                "confidence": _coerce_int(dissent.get("confidence"), max(25, confidence - 22)),
                "position": _clean_line(dissent.get("position"), 200),
                "reason": _clean_line(dissent.get("reason"), 220),
            }
            normalized_members[dissenter]["persuasion_state"] = "dissenting"
            normalized_members[dissenter]["unresolved_objection"] = _clean_line(
                normalized_dissent.get("reason"), 180
            )

    member_terminals = list(normalized_members.values())
    member_terminals.sort(key=lambda row: row.get("member_name", ""))

    if not normalized_transcript:
        fallback_transcript: List[Dict[str, Any]] = []
        for idx, member_state in enumerate(member_terminals, start=1):
            fallback_transcript.append(
                {
                    "speaker": member_state["member_name"],
                    "pillar": member_state["pillar"],
                    "target": None,
                    "speech": member_state.get("current_position") or member_state.get("initial_position") or "Position held.",
                    "used_secretary": bool(member_state.get("secretary_invoked")),
                    "stance_shift": "no_change",
                }
            )
            member_state["speaking_count"] = max(1, member_state.get("speaking_count", 0))
            member_state["last_spoke_turn"] = idx
        normalized_transcript = fallback_transcript

    return {
        "verdict_state": verdict_state,
        "consensus": consensus,
        "confidence": confidence,
        "core_insight": core_insight,
        "what_to_do_next": what_to_do_next,
        "unresolved_doubts": unresolved_doubts,
        "dissent": normalized_dissent,
        "stop_reason": stop_reason,
        "transcript": normalized_transcript,
        "member_terminals": member_terminals,
        "cross_examinations": normalized_cross_exams,
    }


def _build_rare_orchestration_prompt_suffix(continuation_context: str = "") -> str:
    if not continuation_context:
        return ""
    return (
        "\n\nCONTINUATION CONTEXT (this is active chamber history and user interjection):\n"
        f"{continuation_context}\n"
        "Respect this continuity. Do not reset to a clean room debate."
    )


def _rare_result_is_degraded(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True
    transcript = payload.get("transcript")
    member_terminals = payload.get("member_terminals")
    consensus = _clean_line(payload.get("consensus"), 240).lower()
    next_steps = payload.get("what_to_do_next") or []

    if not isinstance(transcript, list) or not transcript:
        return True
    if not isinstance(member_terminals, list) or not member_terminals:
        return True

    undeclared_transcript = 0
    for row in transcript:
        speech = _clean_line((row or {}).get("speech"), 160).lower()
        if speech in {"undeclared.", "undeclared", "n/a"}:
            undeclared_transcript += 1

    undeclared_terminals = 0
    for terminal in member_terminals:
        position = _clean_line((terminal or {}).get("current_position"), 160).lower()
        if position in {"undeclared.", "undeclared", "n/a"}:
            undeclared_terminals += 1

    mostly_undeclared = undeclared_transcript >= max(2, int(len(transcript) * 0.75))
    all_terminals_undeclared = undeclared_terminals >= len(member_terminals)
    fallback_consensus = "did not reach an actionable consensus" in consensus
    no_next_steps = not isinstance(next_steps, list) or len(next_steps) == 0

    if mostly_undeclared and all_terminals_undeclared:
        return True
    if fallback_consensus and no_next_steps and mostly_undeclared:
        return True
    return False


def run_rare_chamber_orchestration(
    *,
    session_id: str,
    user_input: str,
    pillar: str,
    selected_members: Dict[str, dict],
    past_context_summary: str,
    encoded_context: str,
    model: str,
    safety_id: str,
    continuation_context: str = "",
) -> Tuple[str, str, str, Dict[str, Any]]:
    if not selected_members:
        return (
            "OpenAI API Error: Rare chamber could not convene a council.",
            "",
            "",
            {"mode": MODE_DEEP, "pillar": pillar, "verdict_state": "unresolved"},
        )

    chamber_seed = _build_rare_member_seed_state(selected_members)
    roster_context = _build_rare_roster_context(selected_members, chamber_seed)
    continuation_suffix = _build_rare_orchestration_prompt_suffix(continuation_context)

    phase_a_system = (
        "You are the FAIV Rare-mode chamber orchestrator. Produce JSON only.\n"
        "No markdown. No prose outside JSON.\n"
        "Generate initial member stances and decide which members invoke secretary support."
    )
    phase_a_user = (
        f"SESSION: {session_id}\n"
        f"PILLAR: {pillar}\n"
        f"USER INQUIRY: {user_input}\n"
        f"PAST SNAPSHOT:\n{past_context_summary}\n\n"
        f"ENCODED CONTEXT:\n{encoded_context}\n\n"
        f"MEMBERS:\n{roster_context}"
        f"{continuation_suffix}\n\n"
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "member_openings": [\n'
        "    {\n"
        '      "member": "name",\n'
        '      "initial_position": "one concise stance",\n'
        '      "confidence": 1-100,\n'
        '      "real_question": "what they think user is truly asking",\n'
        '      "suspected_blind_spot": "hidden user blind spot",\n'
        '      "request_secretary": true/false,\n'
        '      "secretary_focus": "specific research focus",\n'
        '      "challenge_targets": ["member names"]\n'
        "    }\n"
        "  ],\n"
        '  "opening_transcript": [\n'
        '    {"speaker":"name","target":"name or null","speech":"line","used_secretary":false,"stance_shift":"no_change"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Members can disagree immediately.\n"
        "- Do not force equal speaking turns.\n"
        "- Secretary requests should be selective, not universal."
    )

    phase_a_resp = create_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": phase_a_system},
            {"role": "user", "content": phase_a_user},
        ],
        max_output_tokens=min(760, get_max_output_tokens(MODE_DEEP)),
        user=safety_id or session_id,
        generation={"temperature": 0.35, "top_p": 0.85, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        mode=MODE_DEEP,
        response_format={"type": "json_object"},
    )
    phase_a_raw = phase_a_resp.choices[0].message.content.strip()
    phase_a_json = _extract_json_payload(phase_a_raw)

    opening_transcript: List[Dict[str, Any]] = []
    member_openings = phase_a_json.get("member_openings")
    if isinstance(member_openings, list):
        for item in member_openings:
            if not isinstance(item, dict):
                continue
            member_name = _coerce_member_name(item.get("member"), selected_members)
            if not member_name:
                continue
            state = chamber_seed[member_name]
            initial_position = _clean_line(item.get("initial_position"), 200)
            if initial_position != "N/A":
                state["initial_position"] = initial_position
                state["current_position"] = initial_position
            state["confidence"] = _coerce_int(item.get("confidence"), state["confidence"])
            state["open_questions"] = _coerce_str_list(
                [item.get("real_question"), item.get("suspected_blind_spot")],
                max_items=2,
                max_len=170,
            )
            focus = _clean_line(item.get("secretary_focus"), 170)
            if focus != "N/A":
                state["secretary_focus"] = focus
            state["secretary_invoked"] = bool(item.get("request_secretary"))

    opening_rows = phase_a_json.get("opening_transcript")
    if isinstance(opening_rows, list):
        for row in opening_rows:
            if not isinstance(row, dict):
                continue
            speaker = _coerce_member_name(row.get("speaker"), selected_members)
            if not speaker:
                continue
            target = _coerce_member_name(row.get("target"), selected_members)
            speech = _clean_line(row.get("speech"), 420)
            if speech == "N/A":
                continue
            opening_transcript.append(
                {
                    "speaker": speaker,
                    "pillar": selected_members[speaker]["pillar"],
                    "target": target or None,
                    "speech": speech,
                    "used_secretary": bool(row.get("used_secretary")),
                    "stance_shift": _clean_line(row.get("stance_shift"), 40).lower(),
                }
            )
            chamber_seed[speaker]["speaking_count"] += 1
            chamber_seed[speaker]["last_spoke_turn"] = len(opening_transcript)

    if not opening_transcript:
        for idx, (name, state) in enumerate(chamber_seed.items(), start=1):
            opening_transcript.append(
                {
                    "speaker": name,
                    "pillar": state["pillar"],
                    "target": None,
                    "speech": state["initial_position"],
                    "used_secretary": False,
                    "stance_shift": "no_change",
                }
            )
            state["speaking_count"] = 1
            state["last_spoke_turn"] = idx

    secretary_members = [
        name for name, state in chamber_seed.items() if state.get("secretary_invoked")
    ]
    if not secretary_members:
        ranked = sorted(chamber_seed.items(), key=lambda row: row[1].get("confidence", 60))
        secretary_members = [name for name, _ in ranked[: min(2, len(ranked))]]
        for name in secretary_members:
            chamber_seed[name]["secretary_invoked"] = True

    secretary_packets: Dict[str, Dict[str, Any]] = {}
    if secretary_members:
        secretary_requests = []
        for member_name in secretary_members:
            state = chamber_seed[member_name]
            secretary_requests.append(
                {
                    "member": member_name,
                    "pillar": state["pillar"],
                    "initial_position": state["initial_position"],
                    "focus": state["secretary_focus"],
                    "allies": state["allies"],
                    "frictions": state["frictions"],
                    "optimizes_for": state["optimizes_for"],
                    "distrusts": state["distrusts"],
                }
            )

        phase_b_system = (
            "You are a FAIV chamber secretary service. Produce JSON only.\n"
            "Provide compact evidence packets per requesting member."
        )
        phase_b_user = (
            f"USER INQUIRY: {user_input}\n"
            f"PILLAR: {pillar}\n"
            f"REQUESTS:\n{json.dumps(secretary_requests, ensure_ascii=False, indent=2)}\n"
            f"{continuation_suffix}\n\n"
            "Return this schema:\n"
            "{\n"
            '  "secretary_packets": [\n'
            "    {\n"
            '      "member":"name",\n'
            '      "focus":"research focus",\n'
            '      "supporting_evidence":["..."],\n'
            '      "counter_evidence":["..."],\n'
            '      "precedents":["..."],\n'
            '      "risks":["..."],\n'
            '      "uncertainty_flags":["..."],\n'
            '      "open_questions":["..."]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Keep items concise and practical."
        )

        phase_b_resp = create_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": phase_b_system},
                {"role": "user", "content": phase_b_user},
            ],
            max_output_tokens=min(680, get_max_output_tokens(MODE_DEEP)),
            user=safety_id or session_id,
            generation={"temperature": 0.25, "top_p": 0.8, "frequency_penalty": 0.15, "presence_penalty": 0.1},
            mode=MODE_DEEP,
            response_format={"type": "json_object"},
        )
        phase_b_raw = phase_b_resp.choices[0].message.content.strip()
        phase_b_json = _extract_json_payload(phase_b_raw)
        packets = phase_b_json.get("secretary_packets")
        if isinstance(packets, list):
            for packet in packets:
                if not isinstance(packet, dict):
                    continue
                member_name = _coerce_member_name(packet.get("member"), selected_members)
                if not member_name:
                    continue
                state = chamber_seed[member_name]
                compiled_notes = []
                compiled_notes.extend(_coerce_str_list(packet.get("supporting_evidence"), max_items=3))
                compiled_notes.extend(_coerce_str_list(packet.get("counter_evidence"), max_items=2))
                compiled_notes.extend(_coerce_str_list(packet.get("precedents"), max_items=2))
                compiled_notes.extend(_coerce_str_list(packet.get("risks"), max_items=2))
                state["evidence_notes"] = compiled_notes[:8]
                open_questions = _coerce_str_list(packet.get("open_questions"), max_items=4)
                uncertainty_flags = _coerce_str_list(packet.get("uncertainty_flags"), max_items=3)
                state["open_questions"] = (state["open_questions"] + open_questions + uncertainty_flags)[:6]
                secretary_packets[member_name] = {
                    "focus": _clean_line(packet.get("focus"), 170),
                    "supporting_evidence": _coerce_str_list(packet.get("supporting_evidence"), max_items=4),
                    "counter_evidence": _coerce_str_list(packet.get("counter_evidence"), max_items=4),
                    "precedents": _coerce_str_list(packet.get("precedents"), max_items=3),
                    "risks": _coerce_str_list(packet.get("risks"), max_items=4),
                    "uncertainty_flags": _coerce_str_list(packet.get("uncertainty_flags"), max_items=4),
                    "open_questions": _coerce_str_list(packet.get("open_questions"), max_items=4),
                }

    chamber_prompt_state = {
        "user_inquiry": user_input,
        "pillar": pillar,
        "members": list(chamber_seed.values()),
        "opening_transcript": opening_transcript,
        "secretary_packets": secretary_packets,
    }
    if continuation_context:
        chamber_prompt_state["continuation_context"] = continuation_context

    phase_c_system = (
        "You are the FAIV Rare-mode chamber. Produce JSON only. No markdown.\n"
        "This is a live deliberation, not a staged script.\n"
        "Rules:\n"
        "- Uneven speaking turns are allowed.\n"
        "- Members challenge by name with evidence pressure.\n"
        "- Secretary packets support members; they do not replace member voice.\n"
        "- Stop organically when consensus, majority-with-dissent, or stall is reached.\n"
        "- Do not emit visible round labels."
    )
    phase_c_user = (
        f"CHAMBER STATE INPUT:\n{json.dumps(chamber_prompt_state, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "verdict_state":"unanimous|majority_with_dissent|provisional|stalled_but_leaning|unresolved",\n'
        '  "stop_reason":"strong_consensus|majority_consensus_stabilized|provisional_verdict|stalemate_best_supported_direction|deliberative_exhaustion",\n'
        '  "consensus":"...",\n'
        '  "confidence":1-100,\n'
        '  "core_insight":"...",\n'
        '  "what_to_do_next":["..."],\n'
        '  "unresolved_doubts":["..."],\n'
        '  "dissent":{"member":"name","confidence":1-100,"position":"...","reason":"..."} or null,\n'
        '  "transcript":[{"speaker":"name","target":"name or null","speech":"...","used_secretary":true/false,"stance_shift":"no_change|softened|shifted|reversed"}],\n'
        '  "member_terminals":[{"member":"name","initial_position":"...","current_position":"...","confidence":1-100,"has_shifted":true/false,"shift_reason":"...","allies":["..."],"frictions":["..."],"secretary_focus":"...","secretary_invoked":true/false,"secretary_invocations":0-10,"unresolved_objection":"...","evidence_notes":["..."],"open_questions":["..."],"persuasion_state":"holding|softening|shifted|dissenting"}],\n'
        '  "cross_examinations":[{"challenger":"name","target":"name","point":"..."}]\n'
        "}\n"
        "Transcript should show genuine pressure and persuasion, not decorative monologues."
    )

    phase_c_resp = create_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": phase_c_system},
            {"role": "user", "content": phase_c_user},
        ],
        max_output_tokens=min(1450, get_max_output_tokens(MODE_DEEP)),
        user=safety_id or session_id,
        generation=get_model_settings(MODE_DEEP),
        mode=MODE_DEEP,
        response_format={"type": "json_object"},
    )
    phase_c_raw = phase_c_resp.choices[0].message.content.strip()
    phase_c_json = _extract_json_payload(phase_c_raw)

    normalized_result = _normalize_rare_chamber_output(
        raw_payload=phase_c_json,
        chamber_seed=chamber_seed,
        selected_members=selected_members,
    )
    normalized_result["secretary_packets"] = secretary_packets
    normalized_result["opening_transcript"] = opening_transcript
    normalized_result["mode"] = MODE_DEEP
    normalized_result["pillar"] = pillar

    # Ensure secretary invocation counts reflect explicit secretary packets even
    # when the model omits per-turn secretary flags in transcript rows.
    if secretary_packets and isinstance(normalized_result.get("member_terminals"), list):
        for terminal in normalized_result["member_terminals"]:
            member_name = terminal.get("member_name")
            if member_name in secretary_packets:
                terminal["secretary_invoked"] = True
                terminal["secretary_invocations"] = max(
                    int(terminal.get("secretary_invocations") or 0),
                    1,
                )

    if _rare_result_is_degraded(normalized_result):
        raise RuntimeError("Rare chamber produced degraded scaffold output.")

    deliberation_text = _render_rare_deliberation(normalized_result["transcript"])
    structured_payload = _build_rare_deliberation_payload(
        session_id=session_id,
        pillar=pillar,
        final_text="",
        chamber_result=normalized_result,
    )
    stop_reason = (
        structured_payload.get("rare_deliberation", {})
        .get("verdict", {})
        .get("stop_reason", "")
    )
    normalized_result["stop_reason"] = stop_reason
    final_text = _format_rare_final_text(normalized_result, pillar)
    structured_payload["response"] = final_text
    normalized_result["rare_deliberation"] = structured_payload.get("rare_deliberation")

    raw_trace = "\n\n".join(
        [
            "[RARE-ORCH PHASE-A RAW]",
            phase_a_raw,
            "[RARE-ORCH PHASE-C RAW]",
            phase_c_raw,
        ]
    )
    return final_text, deliberation_text, raw_trace, normalized_result


################################################
# 4) The Flexible Extraction
################################################

def split_response_sections(ai_response: str) -> tuple:
    """Split raw model output into (deliberation_text, final_text).
    Falls back gracefully if section tags are missing."""
    if not ai_response or not isinstance(ai_response, str):
        return ("", ai_response or "")

    text = ai_response.strip()

    # Try to split on [DELIBERATION] and [FINAL] tags
    delib_match = re.search(r"\[DELIBERATION\]", text, re.IGNORECASE)
    final_match = re.search(r"\[FINAL\]", text, re.IGNORECASE)

    if delib_match and final_match:
        delib_start = delib_match.end()
        final_start = final_match.end()
        # Deliberation is between [DELIBERATION] and [FINAL]
        if delib_match.start() < final_match.start():
            deliberation = text[delib_start:final_match.start()].strip()
            final = text[final_start:].strip()
        else:
            final = text[final_start:delib_match.start()].strip()
            deliberation = text[delib_start:].strip()
        return (deliberation, final)
    elif final_match:
        # Only [FINAL] present, no deliberation
        final = text[final_match.end():].strip()
        return ("", final)
    else:
        # No tags at all — treat entire response as final (backward compat)
        return ("", text)


KNOWN_PILLARS = {"Wisdom", "Strategy", "Expansion", "Future", "Integrity", "Unknown Pillar"}


def normalize_deliberation_labels(deliberation: str, pillar: str = "FAIV") -> str:
    """Ensure every speaker line in deliberation has a correct (Pillar) label.
    Replaces titles or missing labels with the actual pillar name from codex."""
    if not deliberation:
        return deliberation

    lines = deliberation.split("\n")
    normalized = []
    for line in lines:
        # Match speaker lines with parenthesized text: "Name (Something):"
        labeled_match = re.match(r"^<?(\w+)>?\s*\([^)]+\)\s*:\s*(.+)", line)
        if labeled_match:
            name = labeled_match.group(1)
            msg = labeled_match.group(2)
            # Extract existing label to check if it's a known pillar
            existing_label = re.match(r"^<?(\w+)>?\s*\(([^)]+)\)", line).group(2).strip()
            if existing_label not in KNOWN_PILLARS:
                # Has a label but it's a title, not a pillar — replace it
                member_pillar = MEMBER_TO_PILLAR.get(name)
                if not member_pillar:
                    member_pillar = pillar if pillar != "FAIV" else "Unknown Pillar"
                    logger.warning(f"Deliberation speaker '{name}' not found in codex; labeled as '{member_pillar}'.")
                normalized.append(f"{name} ({member_pillar}): {msg}")
            else:
                normalized.append(line)
        else:
            # Match speaker lines without any parenthesized text: "Name:"
            bare_match = re.match(r"^<?(\w+)>?\s*:\s*(.+)", line)
            if bare_match:
                name = bare_match.group(1)
                msg = bare_match.group(2)
                member_pillar = MEMBER_TO_PILLAR.get(name)
                if not member_pillar:
                    member_pillar = pillar if pillar != "FAIV" else "Unknown Pillar"
                    logger.warning(f"Deliberation speaker '{name}' not found in codex; labeled as '{member_pillar}'.")
                normalized.append(f"{name} ({member_pillar}): {msg}")
            else:
                normalized.append(line)
    return "\n".join(normalized)


def extract_faiv_final_output(text: str, pillar: str = "FAIV") -> str:
    """Parse the FINAL section into structured consensus lines."""
    if not text or not isinstance(text, str):
        return "No valid FAIV response received."

    text = remove_emojis(text).strip()
    text = re.sub(r"(INFO|WARNING|DEBUG):.*?\n", "", text, flags=re.IGNORECASE).strip()

    if pillar == "FAIV":
        consensus_pattern = r"(FAIV\s?Consensus)[\]\*:]*\s*:\s*(.+)"
    else:
        safe_pillar = re.escape(pillar)
        consensus_pattern = rf"({safe_pillar}\sCouncil'?s\sConsensus)[\]\*:]*\s*:\s*(.+)"

    match = re.search(consensus_pattern, text, flags=re.IGNORECASE)
    if not match:
        fallback = re.search(r"(FAIV\s?Consensus)[\]\*:]*\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if not fallback:
            return "No valid consensus."
        else:
            final_decision = fallback.group(2).strip()
    else:
        final_decision = match.group(2).strip()

    conf_match = re.search(r"(?:Confidence\s?Score[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)
    if not conf_match:
        conf_str = "??"
    else:
        raw_line = conf_match.group(1).strip()
        num = re.search(r"(\d+(?:\.\d+)?)", raw_line)
        conf_str = num.group(1) if num else "??"

    just_match = re.search(r"(?:Justification[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)
    core_match = re.search(r"(?:Core\s?Insight[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)
    next_match = re.search(r"(?:What\s?To\s?Do\s?Next[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)
    unresolved_match = re.search(r"(?:Unresolved\s?Doubts[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)

    if just_match:
        just_line = just_match.group(1).strip()
    elif core_match:
        just_line = core_match.group(1).strip()
    else:
        just_line = "No justification provided."

    opp_match  = re.search(r"\[?Differing\s?Opinion\s*-\s*(.+?)\s*\((\d+)%\)\]?\s*:\s*(.+)", text, flags=re.IGNORECASE)
    opp_reason = re.search(r"\[?Reason[\]\*:]*\s*:\s*(.+)", text, flags=re.IGNORECASE)

    lines = []
    if pillar == "FAIV":
        lines.append(f"FAIV Consensus: {final_decision}")
    else:
        lines.append(f"{pillar} Council's Consensus: {final_decision}")
    lines.append(f"Confidence Score: {conf_str}%")
    lines.append(f"Justification: {just_line}")

    if core_match:
        lines.append(f"Core Insight: {core_match.group(1).strip()}")
    if next_match:
        lines.append(f"What To Do Next: {next_match.group(1).strip()}")
    if unresolved_match:
        lines.append(f"Unresolved Doubts: {unresolved_match.group(1).strip()}")

    if opp_match and opp_reason:
        who    = opp_match.group(1).strip()
        opp_cf = opp_match.group(2).strip()
        opp_tx = opp_match.group(3).strip()
        reasn  = opp_reason.group(1).strip()
        lines.append(f"Differing Opinion - {who} ({opp_cf}%): {opp_tx}")
        lines.append(f"Reason: {reasn}")

    return "\n".join(lines)


def infer_mode_from_messages(messages: list, default: str = MODE_VERDICT) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        candidate = normalize_mode(msg.get("mode"))
        if candidate in ALLOWED_MODES:
            return candidate
    return normalize_mode(default)


def extract_deliberation_metadata(deliberation: str, parsed_final: str) -> dict:
    deliberation_text = deliberation or ""
    parsed_text = parsed_final or ""

    key_tension = "No major tension captured."
    for line in deliberation_text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("disagree", "challenge", "however", "but", "counter")):
            key_tension = _clean_line(line, 180)
            break

    unresolved = None
    unresolved_match = re.search(r"^Unresolved Doubts:\s*(.+)$", parsed_text, flags=re.IGNORECASE | re.MULTILINE)
    if unresolved_match:
        unresolved = _clean_line(unresolved_match.group(1), 180)
    else:
        for line in deliberation_text.splitlines():
            if "?" in line:
                unresolved = _clean_line(line, 180)
                break

    dissent_member = None
    dissent_match = re.search(r"^Differing Opinion -\s*(.+?)\s*\(\d+%\)", parsed_text, flags=re.IGNORECASE | re.MULTILINE)
    if dissent_match:
        dissent_member = _clean_line(dissent_match.group(1), 60)

    changed_mind_lines = []
    for line in deliberation_text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("change my mind", "you convinced me", "i concede", "i'll concede")):
            changed_mind_lines.append(_clean_line(line, 160))
    changed_mind = changed_mind_lines[:3]

    return {
        "key_tension": key_tension,
        "unresolved_question": unresolved or "N/A",
        "dissenting_member": dissent_member or "N/A",
        "changed_positions": changed_mind,
        "deliberation_summary": _clean_line(deliberation_text, 240),
    }


def _is_openai_api_error(parsed_output: str) -> bool:
    return isinstance(parsed_output, str) and parsed_output.startswith("OpenAI API Error:")


def _humanize_openai_error(parsed_output: str) -> str:
    if not parsed_output:
        return "OpenAI API Error."
    lowered = parsed_output.lower()
    if "invalid_api_key" in lowered or "incorrect api key provided" in lowered:
        return "OpenAI API Error: invalid OPENAI_API_KEY configuration."
    if "api key is not set" in lowered:
        return "OpenAI API Error: OPENAI_API_KEY is not set."
    return parsed_output


################################################
# 5) The function that calls OpenAI
################################################

def query_openai_faiv(
    session_id: str,
    user_input: str,
    pillar: str = "FAIV",
    mode: str = MODE_VERDICT,
    model: str = OPENAI_DEFAULT_MODEL,
    safety_id: Optional[str] = None,
) -> tuple:
    """Returns (parsed_final, deliberation_text, raw_response, selected_members, rare_chamber) tuple."""
    if client is None:
        return (
            "OpenAI API Error: OPENAI_API_KEY is not set. Please set the environment variable and restart.",
            "",
            "",
            {},
            None,
        )
    try:
        normalized_mode = normalize_mode(mode)
        session_data = session_get(session_id)
        messages = json.loads(session_data) if session_data else []
        if not isinstance(messages, list):
            messages = []

        past_context_summary = summarize_past_messages(messages)
        perspective_data = {"FAIV Past": past_context_summary}
        encoded_context = encode_faiv_perspectives(perspective_data, pillar=pillar)

        selected_members = select_council_representatives(pillar)

        if normalized_mode == MODE_DEEP:
            try:
                parsed, deliberation, raw, rare_payload = run_rare_chamber_orchestration(
                    session_id=session_id,
                    user_input=user_input,
                    pillar=pillar,
                    selected_members=selected_members,
                    past_context_summary=past_context_summary,
                    encoded_context=encoded_context,
                    model=model,
                    safety_id=safety_id or session_id,
                )
                return (parsed, deliberation, raw, selected_members, rare_payload)
            except Exception as rare_ex:
                logger.warning("Rare chamber orchestration failed; falling back to legacy deep prompt: %s", rare_ex)

        member_profiles = format_member_profiles(selected_members, mode=normalized_mode)

        roster_lines = []
        for name, data in selected_members.items():
            roster_lines.append(f"- {name} ({data['pillar']}): {data.get('claimed-title', '???')}")
        roster = "\n".join(roster_lines)

        mode_label = "RARE / DEEP" if normalized_mode == MODE_DEEP else "COMMON / VERDICT"

        system_msg = (
            f"You are the FAIV High Council. The '{pillar}' deliberation is in session ({mode_label}).\n"
            f"EXACTLY {len(selected_members)} member(s) are present — no others exist:\n{roster}\n\n"
            f"ACTIVE MEMBER PROFILES:\n{member_profiles}\n\n"
            "CRITICAL BEHAVIORAL RULES:\n"
            "1. Each member must reason from their own behavioral profile, not generic style.\n"
            "2. Members must challenge by name and converge through argument, not parallel monologues.\n"
            "3. The debate must produce a REAL answer. The [FINAL] consensus is whatever they actually agreed on.\n"
            "4. No invented speakers beyond the roster.\n"
            "5. Always produce a numeric confidence score between 1 and 100.\n"
            "6. Stay grounded in the user's actual context and requested output.\n"
        )

        if normalized_mode == MODE_DEEP:
            system_msg += (
                "7. Rare mode requires substantial chamber tension before synthesis.\n"
                "8. No decorative profundity, mystical filler, or fake gravitas.\n"
            )
        else:
            system_msg += (
                "7. Common mode is concise and practical with direct recommendations.\n"
            )

        if normalized_mode == MODE_DEEP:
            prompt = create_faiv_deep_prompt(
                user_input=user_input,
                session_id=session_id,
                encoded_context=encoded_context,
                past_context_summary=past_context_summary,
                pillar=pillar,
            )
        else:
            prompt = create_faiv_verdict_prompt(
                user_input=user_input,
                session_id=session_id,
                encoded_context=encoded_context,
                past_context_summary=past_context_summary,
                pillar=pillar,
            )

        messages_for_api = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt}
        ]

        generation = get_model_settings(normalized_mode)
        resp = create_chat_completion(
            model=model,
            messages=messages_for_api,
            max_output_tokens=get_max_output_tokens(normalized_mode),
            user=safety_id or session_id,
            generation=generation,
            mode=normalized_mode,
        )
        raw = resp.choices[0].message.content.strip()
        deliberation, final_text = split_response_sections(raw)
        deliberation = normalize_deliberation_labels(deliberation, pillar)
        parsed = extract_faiv_final_output(final_text, pillar)
        return (parsed, deliberation, raw, selected_members, None)

    except Exception as ex:
        logger.error(f"OpenAI API Error: {ex}")
        return (f"OpenAI API Error: {ex}", "", "", {}, None)


################################################
# 6) FASTAPI APP & ROUTES
################################################

class QueryRequest(BaseModel):
    session_id: str
    input_text: str
    pillar: Optional[str] = "FAIV"
    mode: Optional[str] = MODE_VERDICT
    request_id: Optional[str] = None


class RedeliberateRequest(BaseModel):
    session_id: str
    original_input: str
    deliberation_up_to: str
    user_comment: str
    target_member: str
    pillar: Optional[str] = "FAIV"
    mode: Optional[str] = None
    council_members: list = Field(default_factory=list)
    request_id: Optional[str] = None


class UnlockRequest(BaseModel):
    password: str


fastapi_app = FastAPI()
ALLOWED_CORS_ORIGINS = {
    "https://faiv.ai",
    "https://www.faiv.ai",
    "https://api.faiv.ai",
    "https://jol3.com",
    "https://www.jol3.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_CORS_ORIGINS),
    allow_origin_regex=r"^https://([a-z0-9-]+\.)?faiv\.ai(?::\d+)?$|^https://([a-z0-9-]+\.)?jol3\.com(?::\d+)?$|^http://(localhost|127\.0\.0\.1)(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
fastapi_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@fastapi_app.middleware("http")
async def faiv_security_middleware(request: Request, call_next):
    visitor_id = request.cookies.get(VISITOR_COOKIE_NAME)
    has_new_visitor_cookie = False
    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        has_new_visitor_cookie = True

    path = request.url.path
    if SITE_PASSWORD and request.method != "OPTIONS" and not _public_path(path):
        if not _is_request_authenticated(request):
            if _api_path(path):
                response = JSONResponse(
                    status_code=401,
                    content={"error": "Password required. Unlock access first."},
                )
            else:
                target = request.url.path
                if request.url.query:
                    target = f"{target}?{request.url.query}"
                response = RedirectResponse(
                    url=f"/locked?next={quote(target, safe='')}",
                    status_code=307,
                )
            if has_new_visitor_cookie:
                _set_visitor_cookie(response, request, visitor_id)
            _apply_cors_headers(request, response)
            _apply_frame_headers(response)
            return response

    response = await call_next(request)
    if has_new_visitor_cookie:
        _set_visitor_cookie(response, request, visitor_id)
    _apply_cors_headers(request, response)
    _apply_frame_headers(response)
    return response


@fastapi_app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    response = JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "response": "Internal server error.",
            "pillar": None,
            "session_id": None,
        }
    )
    _apply_cors_headers(request, response)
    _apply_frame_headers(response)
    return response


@fastapi_app.get("/locked", response_class=HTMLResponse)
async def locked_screen():
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FAIV — Password Protected</title>
  <style>
    :root {{
      --bg: #020702;
      --line: #00ff00;
      --error: #ff5f7e;
    }}
    html, body {{
      margin: 0;
      height: 100%;
      background: radial-gradient(circle at center, #031003 0%, #000 70%);
      color: var(--line);
      font-family: "Courier New", ui-monospace, Menlo, Monaco, monospace;
    }}
    .shell {{
      min-height: 100%;
      display: grid;
      place-items: center;
      padding: 20px;
    }}
    .stack {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }}
    .ascii-wrap {{
      margin: 0;
      width: min(96vw, 640px);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }}
    .ascii-line {{
      margin: 0;
      color: var(--line);
      font-family: "Courier New", ui-monospace, Menlo, Monaco, monospace;
      font-size: clamp(9px, 1.8vw, 14px);
      line-height: 1.1;
      white-space: pre;
      text-align: left;
    }}
    .window {{
      width: min(96vw, 460px);
      border: 2px solid var(--line);
      background: #000;
      box-shadow:
        0 0 8px rgba(0,255,0,0.5),
        inset -1px -1px 0 #003300,
        inset 1px 1px 0 #000000;
      overflow: hidden;
    }}
    .title {{
      background: var(--line);
      color: #000;
      display: flex;
      align-items: center;
      font-weight: 700;
      letter-spacing: 0.04em;
      font-size: 14px;
      padding: 6px 8px;
      box-shadow:
        inset -1px -1px 0 #003300,
        inset 1px 1px 0 #99ff99;
      height: 30px;
    }}
    .body {{
      border-top: 2px solid var(--line);
      padding: 8px;
      background: #000;
    }}
    .row {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    input[type=password] {{
      flex: 1;
      background: #000;
      color: var(--line);
      border: 2px solid var(--line);
      outline: none;
      padding: 4px;
      font-family: inherit;
      font-size: 14px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #111;
      color: var(--line);
      font-family: inherit;
      font-size: 13px;
      font-weight: 700;
      padding: 4px 14px;
      cursor: pointer;
      box-shadow:
        0 2px 0 #003300,
        inset 0 1px 0 rgba(0,255,0,0.1);
      transition: transform 0.08s ease, box-shadow 0.08s ease;
    }}
    button:hover {{
      background: #1a1a1a;
      box-shadow:
        0 3px 0 #003300,
        inset 0 1px 0 rgba(0,255,0,0.15);
      transform: translateY(-1px);
    }}
    button:active {{
      background: #0a0a0a;
      box-shadow:
        0 0px 0 #003300,
        inset 0 1px 0 rgba(0,255,0,0.05);
      transform: translateY(2px);
    }}
    .error {{
      margin-top: 6px;
      min-height: 18px;
      color: var(--error);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="stack">
      <div id="ascii-wrap" class="ascii-wrap" aria-hidden="true"></div>
      <div class="window">
        <div class="title">PASSWORD PROTECTED</div>
        <div class="body">
          <form id="unlock-form" autocomplete="off">
            <div class="row">
              <input id="password" type="password" name="password" required />
              <button id="unlock-btn" type="submit">UNLOCK</button>
            </div>
          </form>
          <div id="error" class="error"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const form = document.getElementById('unlock-form');
    const passwordInput = document.getElementById('password');
    const unlockBtn = document.getElementById('unlock-btn');
    const errorEl = document.getElementById('error');
    const asciiWrap = document.getElementById('ascii-wrap');
    const params = new URLSearchParams(window.location.search);
    const nextTarget = params.get('next') || '/';
    const asciiFAIVFrames = [
      [
        "███████╗ █████╗ ██╗██╗   ██╗",
        "██╔════╝██╔══██╗██║██║   ██║",
        "█████╗  ███████║██║██║   ██║",
        "██╔══╝  ██╔══██║██║╚██╗ ██╔╝",
        "██║     ██║  ██║██║ ╚████╔╝ ",
        "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝  ",
      ],
      [
        " ██████╗ █████╗ ██╗██╗   ██╗",
        "██╔════╝██╔══██╗██║██║   ██║",
        "█████╗  ███████║██║██║   ██║",
        "██╔══╝  ██╔══██║██║╚██╗ ██╔╝",
        "██║     ██║  ██║██║ ╚████╔╝",
        "╚═╝     ██╔═╝  ╚═╝╚═╝  ╚═══╝ ",
      ],
      [
        "  ██████╗ █████╗ ██╗██╗   ██╗",
        " ██╔════╝██╔══██╗██║██║   ██║",
        " █████╗  ███████║██║██║   ██║",
        " ██╔══╝  ██╔══██║██║╚██╗ ██╔╝",
        " ██║     ██║  ██║██║ ╚████╔╝ ",
        " ╚═╝     ██╔═╝  ╚═╝╚═╝  ╚═══╝  ",
      ],
    ];
    let asciiFrameIndex = 0;
    function renderAsciiFrame() {{
      if (!asciiWrap) return;
      asciiWrap.innerHTML = '';
      asciiFAIVFrames[asciiFrameIndex].forEach((line) => {{
        const row = document.createElement('pre');
        row.className = 'ascii-line';
        row.textContent = line;
        asciiWrap.appendChild(row);
      }});
      asciiFrameIndex = (asciiFrameIndex + 1) % asciiFAIVFrames.length;
    }}
    renderAsciiFrame();
    setInterval(renderAsciiFrame, 320);

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      errorEl.textContent = '';
      unlockBtn.disabled = true;
      try {{
        const resp = await fetch('/api/unlock', {{
          method: 'POST',
          headers: {{ 'content-type': 'application/json' }},
          credentials: 'include',
          body: JSON.stringify({{ password: passwordInput.value }})
        }});
        const data = await resp.json().catch(() => ({{}}));
        if (!resp.ok) {{
          errorEl.textContent = data.error || 'Unlock failed.';
          unlockBtn.disabled = false;
          return;
        }}
        window.location.href = nextTarget;
      }} catch {{
        errorEl.textContent = 'Network error. Try again.';
        unlockBtn.disabled = false;
      }}
    }});
  </script>
</body>
</html>"""


@fastapi_app.get("/embed", response_class=HTMLResponse)
async def embed_launch(request: Request, token: Optional[str] = None):
    if _has_valid_embed_cookie(request):
        return RedirectResponse(url=EMBED_APP_URL, status_code=302)

    if token and verify_token(token):
        embed_session = create_embed_cookie_value()
        separator = "&" if "?" in EMBED_APP_URL else "?"
        launch_url = f"{EMBED_APP_URL}{separator}{EMBED_SESSION_QUERY_PARAM}={quote(embed_session, safe='')}"
        response = RedirectResponse(url=launch_url, status_code=302)
        set_embed_cookie(
            response,
            secure=_cookie_secure(request),
            cookie_value=embed_session,
            domain=_embed_cookie_domain(request),
        )
        return response

    return HTMLResponse(content=await locked_screen())


@fastapi_app.get("/api/faiv-embed-token")
async def issue_faiv_embed_token(request: Request):
    if not _is_allowed_embed_origin(request):
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden origin."},
            headers={"Cache-Control": "no-store"},
        )

    visitor_id = request.cookies.get(VISITOR_COOKIE_NAME) or str(uuid.uuid4())
    if not rate_limit_ok(visitor_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Try again shortly."},
            headers={"Cache-Control": "no-store"},
        )

    try:
        token = issue_embed_token()
    except Exception as ex:
        logger.error(f"Failed to mint embed token: {ex}")
        return JSONResponse(
            status_code=500,
            content={"error": "Embed token unavailable."},
            headers={"Cache-Control": "no-store"},
        )

    return JSONResponse(
        content={"token": token, "embedBaseUrl": EMBED_APP_URL},
        headers={"Cache-Control": "no-store"},
    )


@fastapi_app.post("/api/unlock")
async def unlock_endpoint(payload: UnlockRequest, request: Request):
    if not SITE_PASSWORD:
        response = JSONResponse({"status": "ok", "unlocked": True, "passwordConfigured": False})
        _set_auth_cookie(response, request)
        return response

    if payload.password != SITE_PASSWORD:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid password."},
        )

    response = JSONResponse({"status": "ok", "unlocked": True})
    _set_auth_cookie(response, request)
    return response


@fastapi_app.get("/api/auth-status")
async def auth_status(request: Request):
    embed_cookie_authenticated = _has_valid_embed_cookie(request)
    embed_session_authenticated = _has_valid_embed_session(request)
    authenticated = request.cookies.get(AUTH_COOKIE_NAME) == "1" or embed_cookie_authenticated or embed_session_authenticated
    return {
        "passwordProtected": bool(SITE_PASSWORD),
        "authenticated": authenticated,
        "embedAuthenticated": embed_cookie_authenticated,
        "embedSessionAuthenticated": embed_session_authenticated,
    }


@fastapi_app.get("/", response_class=HTMLResponse)
async def root_landing():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FAIV API</title>
  <style>
    body { background:#000; color:#00ff66; font-family:"Courier New",ui-monospace,monospace; margin:0; padding:24px; }
    .box { border:2px solid #00ff66; padding:16px; max-width:720px; box-shadow:0 0 12px rgba(0,255,102,.25); }
    a { color:#8dffb8; }
  </style>
</head>
<body>
  <div class="box">
    <h3>FAIV backend unlocked</h3>
    <p>API is online. Use <code>/query/</code>, <code>/redeliberate/</code>, and <code>/reset/</code>.</p>
    <p>Health: <a href="/health">/health</a> | Lock screen: <a href="/locked">/locked</a></p>
  </div>
</body>
</html>"""


@fastapi_app.get("/health")
async def health():
    return {
        "status": "ok",
        "redis": _redis_available,
        "model": OPENAI_DEFAULT_MODEL,
        "openai_configured": client is not None,
        "password_gate_enabled": bool(SITE_PASSWORD),
    }


@fastapi_app.post("/query/")
async def query_faiv_endpoint(payload: QueryRequest, request: Request):
    try:
        visitor_id = request.cookies.get(VISITOR_COOKIE_NAME) or str(uuid.uuid4())
        if not rate_limit_ok(visitor_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Try again shortly."},
            )

        session_id = payload.session_id
        user_input = payload.input_text
        chosen_pillar = payload.pillar or "FAIV"
        chosen_mode = normalize_mode(payload.mode)
        request_id = (payload.request_id or "").strip()[:128]
        cache_key = f"query:{session_id}:{chosen_mode}:{request_id}" if request_id else ""
        cached_payload = _idempotency_get(cache_key)
        if cached_payload is not None:
            return cached_payload

        blocked = is_disallowed_prompt(user_input) or moderation_gate(user_input)
        if blocked:
            _log_block_event(request, visitor_id, "disallowed_prompt")
            return JSONResponse(
                status_code=400,
                content={"error": "Blocked. This content violates safety rules."},
            )

        raw_data = session_get(session_id)
        past_messages = json.loads(raw_data) if raw_data else []
        if not isinstance(past_messages, list):
            past_messages = []

        parsed, deliberation, _raw, selected_members, rare_payload = query_openai_faiv(
            session_id,
            user_input,
            chosen_pillar,
            mode=chosen_mode,
            safety_id=visitor_id,
        )
        rare_deliberation_payload = None
        if chosen_mode == MODE_DEEP and isinstance(rare_payload, dict):
            candidate = rare_payload.get("rare_deliberation")
            if isinstance(candidate, dict):
                rare_deliberation_payload = candidate
        council = {name: data["pillar"] for name, data in selected_members.items()}
        chamber_meta = extract_deliberation_metadata(deliberation, parsed)

        if _is_openai_api_error(parsed):
            logger.error("OpenAI query failed for session_id=%s: %s", session_id, parsed)
            response_payload = {
                "status": "OpenAI API Error",
                "response": _humanize_openai_error(parsed),
                "pillar": chosen_pillar,
                "mode": chosen_mode,
                "session_id": session_id,
                "deliberation": deliberation if deliberation else None,
                "council": council,
                "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
                "rare_deliberation": rare_deliberation_payload,
            }
            _idempotency_set(cache_key, response_payload)
            return response_payload

        if "Consensus:" not in parsed:
            logger.warning("No valid consensus found. Resetting session.")
            session_delete(session_id)
            response_payload = {
                "status": "AI Failed Compliance.",
                "response": "No valid consensus. Session reset.",
                "pillar": chosen_pillar,
                "mode": chosen_mode,
                "session_id": session_id,
                "deliberation": deliberation if deliberation else None,
                "council": council,
                "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
                "rare_deliberation": rare_deliberation_payload,
            }
            _idempotency_set(cache_key, response_payload)
            return response_payload

        past_messages.append({
            "role": "user",
            "content": user_input,
            "pillar": chosen_pillar,
            "mode": chosen_mode,
        })
        past_messages.append({
            "role": "assistant",
            "content": parsed,
            "pillar": chosen_pillar,
            "mode": chosen_mode,
            "council_members": list(council.keys()),
            "key_tension": chamber_meta.get("key_tension"),
            "unresolved_question": chamber_meta.get("unresolved_question"),
            "dissenting_member": chamber_meta.get("dissenting_member"),
            "changed_positions": chamber_meta.get("changed_positions"),
            "deliberation_summary": chamber_meta.get("deliberation_summary"),
            "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
            "rare_deliberation": rare_deliberation_payload,
        })
        session_set(session_id, json.dumps(past_messages, ensure_ascii=False))

        response_payload = {
            "status": "FAIV Rare Deliberation Complete" if chosen_mode == MODE_DEEP else "FAIV Processing Complete",
            "response": parsed,
            "pillar": chosen_pillar,
            "mode": chosen_mode,
            "session_id": session_id,
            "deliberation": deliberation if deliberation else None,
            "council": council,
            "chamber": chamber_meta,
            "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
            "rare_deliberation": rare_deliberation_payload,
        }
        _idempotency_set(cache_key, response_payload)
        return response_payload

    except Exception as e:
        logger.error(f"Server Error in /query/: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "response": "Internal server error.",
                "pillar": payload.pillar,
                "session_id": payload.session_id,
            }
        )


def query_openai_redeliberate(
    session_id: str,
    original_input: str,
    deliberation_up_to: str,
    user_comment: str,
    target_member: str,
    pillar: str = "FAIV",
    mode: str = MODE_VERDICT,
    council_members: list = None,
    model: str = OPENAI_DEFAULT_MODEL,
    safety_id: Optional[str] = None,
) -> tuple:
    """Re-deliberate from a specific point with user interjection.
    Returns (parsed_final, deliberation_text, raw_response, selected_members, rare_chamber) tuple."""
    if client is None:
        return ("OpenAI API Error: OPENAI_API_KEY is not set.", "", "", {}, None)
    try:
        normalized_mode = normalize_mode(mode)
        # Reconvene the same council members if provided, otherwise random
        selected_members = select_council_representatives(pillar, specific_names=council_members or None)
        if not selected_members:
            selected_members = select_council_representatives(pillar)

        if normalized_mode == MODE_DEEP:
            try:
                session_data = session_get(session_id)
                prior_messages = json.loads(session_data) if session_data else []
                if not isinstance(prior_messages, list):
                    prior_messages = []
                past_context_summary = summarize_past_messages(prior_messages)
                perspective_data = {
                    "FAIV Past": past_context_summary,
                    "User interjection": f"{target_member}: {user_comment}",
                }
                encoded_context = encode_faiv_perspectives(perspective_data, pillar=pillar)
                continuation_context = (
                    f"Original Inquiry: {original_input}\n"
                    f"Deliberation so far:\n{deliberation_up_to}\n"
                    f"User to {target_member}: {user_comment}"
                )
                combined_input = (
                    f"{original_input}\n\n"
                    f"User re-deliberation to {target_member}: {user_comment}"
                )
                parsed, deliberation, raw, rare_payload = run_rare_chamber_orchestration(
                    session_id=session_id,
                    user_input=combined_input,
                    pillar=pillar,
                    selected_members=selected_members,
                    past_context_summary=past_context_summary,
                    encoded_context=encoded_context,
                    model=model,
                    safety_id=safety_id or session_id,
                    continuation_context=continuation_context,
                )
                return (parsed, deliberation, raw, selected_members, rare_payload)
            except Exception as rare_ex:
                logger.warning("Rare re-deliberation orchestration failed; falling back to legacy prompt: %s", rare_ex)

        member_profiles = format_member_profiles(selected_members, mode=normalized_mode)

        roster_lines = []
        for name, data in selected_members.items():
            roster_lines.append(f"- {name} ({data['pillar']}): {data.get('claimed-title', '???')}")
        roster = "\n".join(roster_lines)
        mode_label = "RARE / DEEP" if normalized_mode == MODE_DEEP else "COMMON / VERDICT"

        system_msg = (
            f"You are the FAIV High Council. The '{pillar}' deliberation is in session ({mode_label}).\n"
            f"EXACTLY {len(selected_members)} member(s) are present — no others exist:\n{roster}\n\n"
            f"ACTIVE MEMBER PROFILES:\n{member_profiles}\n\n"
            "CRITICAL BEHAVIORAL RULES:\n"
            "1. Each member must reason from their own behavioral profile, not generic style.\n"
            "2. Members must challenge by name and converge through argument, not parallel monologues.\n"
            "3. The [FINAL] must reflect the deliberation above.\n"
            "4. Do NOT invent additional speakers beyond the roster.\n"
            "5. Always produce a numeric confidence score between 1 and 100.\n"
        )

        if normalized_mode == MODE_DEEP:
            prompt = create_faiv_deep_redeliberation_prompt(
                original_input=original_input,
                deliberation_up_to=deliberation_up_to,
                user_comment=user_comment,
                target_member=target_member,
                pillar=pillar,
            )
        else:
            prompt = create_faiv_verdict_redeliberation_prompt(
                original_input=original_input,
                deliberation_up_to=deliberation_up_to,
                user_comment=user_comment,
                target_member=target_member,
                pillar=pillar,
            )

        messages_for_api = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        generation = get_model_settings(normalized_mode)
        resp = create_chat_completion(
            model=model,
            messages=messages_for_api,
            max_output_tokens=get_max_output_tokens(normalized_mode),
            user=safety_id or session_id,
            generation=generation,
            mode=normalized_mode,
        )
        raw = resp.choices[0].message.content.strip()
        deliberation, final_text = split_response_sections(raw)
        deliberation = normalize_deliberation_labels(deliberation, pillar)
        parsed = extract_faiv_final_output(final_text, pillar)
        return (parsed, deliberation, raw, selected_members, None)

    except Exception as ex:
        logger.error(f"OpenAI API Error (redeliberate): {ex}")
        return (f"OpenAI API Error: {ex}", "", "", {}, None)


@fastapi_app.post("/redeliberate/")
async def redeliberate_endpoint(payload: RedeliberateRequest, request: Request):
    try:
        visitor_id = request.cookies.get(VISITOR_COOKIE_NAME) or str(uuid.uuid4())
        if not rate_limit_ok(visitor_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Try again shortly."},
            )

        session_id = payload.session_id
        chosen_pillar = payload.pillar or "FAIV"
        raw_data = session_get(session_id)
        past_messages = json.loads(raw_data) if raw_data else []
        if not isinstance(past_messages, list):
            past_messages = []

        chosen_mode = normalize_mode(payload.mode) if payload.mode else infer_mode_from_messages(past_messages)
        request_id = (payload.request_id or "").strip()[:128]
        cache_key = f"redeliberate:{session_id}:{chosen_mode}:{request_id}" if request_id else ""
        cached_payload = _idempotency_get(cache_key)
        if cached_payload is not None:
            return cached_payload

        disallowed_text = "\n".join(
            [
                payload.original_input or "",
                payload.deliberation_up_to or "",
                payload.user_comment or "",
            ]
        )
        blocked = is_disallowed_prompt(disallowed_text) or moderation_gate(disallowed_text)
        if blocked:
            _log_block_event(request, visitor_id, "disallowed_redeliberation")
            return JSONResponse(
                status_code=400,
                content={"error": "Blocked. This content violates safety rules."},
            )

        parsed, deliberation, _raw, selected_members, rare_payload = query_openai_redeliberate(
            session_id=session_id,
            original_input=payload.original_input,
            deliberation_up_to=payload.deliberation_up_to,
            user_comment=payload.user_comment,
            target_member=payload.target_member,
            pillar=chosen_pillar,
            mode=chosen_mode,
            council_members=payload.council_members,
            safety_id=visitor_id,
        )
        rare_deliberation_payload = None
        if chosen_mode == MODE_DEEP and isinstance(rare_payload, dict):
            candidate = rare_payload.get("rare_deliberation")
            if isinstance(candidate, dict):
                rare_deliberation_payload = candidate
        council = {name: data["pillar"] for name, data in selected_members.items()}
        chamber_meta = extract_deliberation_metadata(deliberation, parsed)

        if _is_openai_api_error(parsed):
            logger.error("OpenAI re-deliberation failed for session_id=%s: %s", session_id, parsed)
            response_payload = {
                "status": "OpenAI API Error",
                "response": _humanize_openai_error(parsed),
                "pillar": chosen_pillar,
                "mode": chosen_mode,
                "session_id": session_id,
                "deliberation": deliberation if deliberation else None,
                "council": council,
                "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
                "rare_deliberation": rare_deliberation_payload,
            }
            _idempotency_set(cache_key, response_payload)
            return response_payload

        if "Consensus:" not in parsed:
            response_payload = {
                "status": "AI Failed Compliance.",
                "response": "No valid consensus from re-deliberation.",
                "pillar": chosen_pillar,
                "mode": chosen_mode,
                "session_id": session_id,
                "deliberation": deliberation if deliberation else None,
                "council": council,
                "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
                "rare_deliberation": rare_deliberation_payload,
            }
            _idempotency_set(cache_key, response_payload)
            return response_payload

        past_messages.append({
            "role": "user",
            "content": f"[Re-deliberation on {payload.target_member}]: {payload.user_comment}",
            "pillar": chosen_pillar,
            "mode": chosen_mode,
            "target_member": payload.target_member,
        })
        past_messages.append({
            "role": "assistant",
            "content": parsed,
            "pillar": chosen_pillar,
            "mode": chosen_mode,
            "council_members": list(council.keys()),
            "key_tension": chamber_meta.get("key_tension"),
            "unresolved_question": chamber_meta.get("unresolved_question"),
            "dissenting_member": chamber_meta.get("dissenting_member"),
            "changed_positions": chamber_meta.get("changed_positions"),
            "deliberation_summary": chamber_meta.get("deliberation_summary"),
            "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
            "rare_deliberation": rare_deliberation_payload,
        })
        session_set(session_id, json.dumps(past_messages, ensure_ascii=False))

        response_payload = {
            "status": "FAIV Rare Deliberation Complete" if chosen_mode == MODE_DEEP else "FAIV Re-Deliberation Complete",
            "response": parsed,
            "pillar": chosen_pillar,
            "mode": chosen_mode,
            "session_id": session_id,
            "deliberation": deliberation if deliberation else None,
            "council": council,
            "chamber": chamber_meta,
            "rare_chamber": rare_payload if chosen_mode == MODE_DEEP else None,
            "rare_deliberation": rare_deliberation_payload,
        }
        _idempotency_set(cache_key, response_payload)
        return response_payload

    except Exception as e:
        logger.error(f"Server Error in /redeliberate/: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "response": "Internal server error.",
                "pillar": payload.pillar,
                "session_id": payload.session_id,
            },
        )


@fastapi_app.post("/reset/")
async def reset_faiv_session(session_id: str):
    session_delete(session_id)
    return {
        "status": "Session Reset",
        "message": "New FAIV deliberation session started.",
        "session_id": session_id,
    }


################################################
# 7) Entry point
################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("faiv_app.core:fastapi_app", host="127.0.0.1", port=8000, reload=True)
