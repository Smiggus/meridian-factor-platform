import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt

import config

logger = logging.getLogger(__name__)

_bearer = HTTPBearer()


def create_token(subject: str = "local", scope: str | None = None) -> str:
    """
    Generates a signed JWT. Run once and store the output  -- pass it as
    the Bearer token in every API request.

    `scope` (optional, space-separated, additive) is stamped as the `scope`
    claim. A token carrying the `admin` scope may call the cache-mutating
    admin routes (see require_admin).

    Usage:
        python -c "from api.auth import create_token; print(create_token())"
        python -c "from api.auth import create_token; print(create_token('ops', scope='admin'))"
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    if scope:
        payload["scope"] = scope
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def verify_token(
    credentials=Depends(_bearer),
) -> dict:
    """Verify the Bearer token and return its payload. Raises 401 on any failure."""
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials,
            config.JWT_SECRET,
            algorithms=[config.JWT_ALGORITHM],
        )
        if payload.get("sub") is None:
            raise exc
        return payload
    except (JWTError, AttributeError):
        raise exc


# Admin authorization (API-004)
ADMIN_SCOPE = "admin"
ADMIN_SUBJECTS_ENV = "MERIDIAN_ADMIN_SUBJECTS"

_open_admin_warned = False
_open_admin_lock = threading.Lock()


def _admin_subjects() -> frozenset[str]:
    """Subjects named in MERIDIAN_ADMIN_SUBJECTS (comma list), read per call."""
    raw = os.environ.get(ADMIN_SUBJECTS_ENV, "")
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def _has_admin_claim(payload) -> bool:
    """True when the JWT carries an admin scope or role.

    Reads `scope` (a space-separated string or a list), `scopes`, `role` and `roles`. Only
    `scope` is split on whitespace. The other claims are compared whole against "admin".
    """
    if not isinstance(payload, dict):
        return False

    def _items(value, split: bool) -> set[str]:
        if isinstance(value, str):
            return set(value.split()) if split else {value}
        if isinstance(value, (list, tuple, set)):
            return {v for v in value if isinstance(v, str)}
        return set()

    claimed = (
        _items(payload.get("scope"), split=True)
        | _items(payload.get("scopes"), split=False)
        | _items(payload.get("role"), split=False)
        | _items(payload.get("roles"), split=False)
    )
    return ADMIN_SCOPE in claimed


def require_admin(payload=Depends(verify_token)) -> dict:
    """
    FastAPI dependency for cache-mutating admin routes (POST /cache/invalidate,
    POST /admin/cache/warm). Runs verify_token first (401 on a bad token).

    A caller is admin when its JWT carries an admin scope/role claim or its
    subject is listed in MERIDIAN_ADMIN_SUBJECTS. Enforcement starts the
    moment MERIDIAN_ADMIN_SUBJECTS is set: a non-admin caller then gets 403.

    Backward compatibility: while MERIDIAN_ADMIN_SUBJECTS is unset every
    authenticated caller is still allowed (existing consumers and the Dagster
    composite_cache_warm_sensor hold ordinary tokens), and a WARNING is logged
    once per process so the open state is visible.
    """
    global _open_admin_warned

    subject = payload.get("sub") if isinstance(payload, dict) else None
    if _has_admin_claim(payload):
        logger.info("admin route allowed for subject %r (admin claim)", subject)
        return payload

    subjects = _admin_subjects()
    if subjects:
        if subject in subjects:
            logger.info(
                "admin route allowed for subject %r (listed in %s)", subject, ADMIN_SUBJECTS_ENV
            )
            return payload
        logger.warning("admin route denied for subject %r (not admin)", subject)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin scope required",
        )

    with _open_admin_lock:
        first = not _open_admin_warned
        _open_admin_warned = True
    if first:
        logger.warning(
            "admin routes are open to every authenticated caller: %s is unset and "
            "subject %r carries no admin claim. Set %s (comma list) to enforce.",
            ADMIN_SUBJECTS_ENV,
            subject,
            ADMIN_SUBJECTS_ENV,
        )
    return payload
