"""YYYY-MM-DD query-parameter parsing shared by the routers (API-005).

Every route with a date parameter calls parse_iso_date. A bad value gets a 400 up front, where
it used to be truncated, passed to pandas or reported as a 404 for missing data.
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
    """Parse a calendar date parameter, or raise HTTP 400.

    The contract is frozen, so every form that used to work still does: `YYYY-MM-DD` with
    or without an ISO time suffix (which is dropped), and compact `YYYYMMDD`. Surrounding
    whitespace is stripped. Anything else is a 400, impossible dates included.
    Callers fill in a default for an omitted parameter before calling.
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
