"""Test helper: capture and decode the stage-table COPY of the #432 write path.

upsert_daily_scores / upsert_group_scores load their payload into a temp stage
table with one `COPY _m432_stage (...) FROM STDIN` (text format) issued on
`conn.connection.cursor()`. Unit tests have no database, so they swap that
cursor for a recorder and read the rows back out of the COPY text.
"""

from __future__ import annotations

import re

_FLOAT_COLS = {"raw_value", "z_score", "group_score"}
_INT_COLS = {"direction", "campaign_id"}


def _unescape(field: str) -> "str | None":
    if field == "\\N":
        return None
    out, i = [], 0
    while i < len(field):
        ch = field[i]
        if ch == "\\" and i + 1 < len(field):
            nxt = field[i + 1]
            out.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def decode_copy_text(sql: str, buf) -> "list[dict]":
    """Rows of a recorded COPY ... FROM STDIN (text format) as dicts.

    Numeric columns are converted (floats, ints); everything else stays a str and
    \\N is None. A row is split on literal tabs only, so an escaped tab inside a
    value survives as a tab character in the decoded string.
    """
    cols = [c.strip() for c in re.search(r"\(([^)]*)\)\s+FROM STDIN", sql).group(1).split(",")]
    text = buf.getvalue() if hasattr(buf, "getvalue") else buf.read()
    rows = []
    for line in text.split("\n"):
        if not line:
            continue
        row = {}
        for col, raw in zip(cols, line.split("\t"), strict=True):
            val = _unescape(raw)
            if val is not None and col in _FLOAT_COLS:
                val = float(val)
            elif val is not None and col in _INT_COLS:
                val = int(val)
            row[col] = val
        rows.append(row)
    return rows


def record_stage_copy(conn, sink) -> None:
    """Wire a MagicMock `conn` so a stage COPY is appended to `sink` as
    {"sql": "COPY _m432_stage (...) FROM STDIN", "params": [row dicts]}."""

    def copy_expert(sql, buf):
        sink.append({"sql": sql, "params": decode_copy_text(sql, buf)})

    conn.connection.cursor.return_value.copy_expert.side_effect = copy_expert
