"""Shared YYYY-MM-DD query-parameter validation for the API routers (API-005).

Every route that documents a YYYY-MM-DD parameter parses it here, so a
malformed value is rejected with the same 400 before any lookup happens
instead of being truncated, coerced by pandas, or turned into a "no data" 404.
"""

from __future__ import annotations

import re
from datetime import date as date_cls
from datetime import datetime

from fastapi import HTTPException, status

# `YYYY-MM-DD`, optionally followed by an ISO time suffix after `T` or a space
# (the old `[:10]` truncation accepted these), or the compact `YYYYMMDD` form
# (accepted by `/scores` via pandas and by `/universe`). The suffix is
# discarded: only the calendar date is used.
_ISO_DATE_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:[T ](?:[01]\d|2[0-3])(?::?[0-5]\d(?::?[0-5]\d(?:[.,]\d+)?)?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?)?"
)
_COMPACT_DATE_RE = re.compile(r"\d{8}")


def parse_iso_date(raw: str, param_name: str = "date") -> date_cls:
    """Parse a calendar date parameter; raise HTTP 400 otherwise.

    Accepted (the API contract is frozen, so what used to work keeps working):
    `YYYY-MM-DD`, `YYYY-MM-DD` followed by an ISO time suffix (`T12:34:56`,
    ` 12:34:56`, with optional fraction and `Z`/offset), and compact
    `YYYYMMDD`. The time suffix is ignored. Trailing garbage, other
    separators and impossible dates ("2026-02-30") are rejected. Surrounding
    whitespace is ignored. Callers that default an omitted parameter do so
    before calling this.
    """
    text = raw.strip() if isinstance(raw, str) else raw
    if isinstance(text, str):
        try:
            m = _ISO_DATE_RE.fullmatch(text)
            if m:
                return date_cls.fromisoformat(m.group("date"))
            if _COMPACT_DATE_RE.fullmatch(text):
                return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            pass
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid {param_name} {raw!r}; expected YYYY-MM-DD",
    )
