"""Exact, order-independent per-date content evidence for PG/Iceberg score mirrors.

Use the same ordered column tuple on both sides. The digest includes every field,
including nullable floats and timestamps; never replace this with sums of values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from contextvars import ContextVar
from datetime import date, datetime, timezone
from typing import Iterable, Sequence

evidence_scratch: ContextVar[str | None] = ContextVar("mirror_evidence_scratch", default=None)


def _canonical(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return ["timestamp", value.astimezone(timezone.utc).isoformat(timespec="microseconds")]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, float):
        if not math.isfinite(value):
            # Iceberg stores null for non-finite PG values in the existing mirror.
            return None
        return ["float", value.hex()]
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    raise TypeError(f"unsupported mirror value type: {type(value).__name__}")


def content_evidence(
    rows: Iterable[Sequence], columns: Sequence[str], identity_columns: Sequence[str]
) -> dict:
    """Order-independent evidence with disk-backed uniqueness and hash ordering.

    SQLite's bounded page cache and file-backed sort avoid row-count-sized Python
    collections. Scratch follows MERIDIAN_MIRROR_SCRATCH_DIR, then TMPDIR.

    Returns {"rows", "identities", "digest"}:

    - rows: number of rows seen.
    - identities: number of DISTINCT factor_name (or group_name) values, not
      the number of row identities. It is 1 for a single-factor date, and 0
      when the columns carry neither label. Row identities are unique by
      construction (a duplicate raises), so their count is always `rows`. The
      key name is historical and is kept because both mirror sides emit it and
      the evidence dicts are compared and reported with it.
    - digest: sha256 over the sorted per-row hashes.
    """
    identity_indexes = tuple(columns.index(name) for name in identity_columns)
    label = next((columns.index(c) for c in ("factor_name", "group_name") if c in columns), None)
    scratch_root = evidence_scratch.get() or os.getenv("MERIDIAN_MIRROR_SCRATCH_DIR")
    with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
        db = sqlite3.connect(os.path.join(scratch, "evidence.sqlite"))
        try:
            db.execute("PRAGMA cache_size=-2048")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("CREATE TABLE rows (identity BLOB PRIMARY KEY, hash BLOB, label TEXT)")
            # Maintain order on disk rather than a potentially large temporary sort.
            db.execute("CREATE INDEX hash_order ON rows(hash)")
            db.execute("CREATE INDEX label_order ON rows(label)")
            count = 0
            for row in rows:
                if len(row) != len(columns):
                    raise ValueError("mirror row does not match column contract")
                identity = json.dumps(
                    [_canonical(row[i]) for i in identity_indexes], separators=(",", ":")
                )
                payload = json.dumps(
                    [_canonical(v) for v in row], separators=(",", ":"), ensure_ascii=False
                )
                label_value = json.dumps(_canonical(row[label])) if label is not None else None
                try:
                    db.execute(
                        "INSERT INTO rows VALUES (?, ?, ?)",
                        (identity, hashlib.sha256(payload.encode("utf-8")).digest(), label_value),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate mirror identity") from exc
                count += 1
            digest = hashlib.sha256()
            for (row_hash,) in db.execute("SELECT hash FROM rows ORDER BY hash"):
                digest.update(row_hash)
            label_count = db.execute("SELECT COUNT(DISTINCT label) FROM rows").fetchone()[0]
            return {"rows": count, "identities": label_count, "digest": digest.hexdigest()}
        finally:
            db.close()
