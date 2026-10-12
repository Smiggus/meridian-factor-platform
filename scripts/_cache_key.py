"""_cache_key.py - filename-keyed research caches that cannot answer the wrong question.

The bug this exists to prevent (found in review, and present in several call sites
across the research scripts at the time):

    def _fetch(con, tickers, start, end, cache: Path):
        if cache is not None and cache.exists():
            return pl.read_parquet(cache)      # tickers/start/end IGNORED
        ...

A cache keyed on a FIXED filename while its function takes varying parameters returns
whatever was cached last, regardless of what was asked for. The failure is silent and
the output is entirely plausible, because it IS a real result - just the answer to a
different question. In practice this survives review: the run completes, the numbers
look reasonable, and the only tell is a shape or row count that does not match the
window that was requested.

A `--force` flag does not fix this. It relies on someone remembering, and the failure
mode is precisely that nothing prompts you to remember.

The fix is to put every parameter that determines the CONTENT into the filename, so a
parameter change is a cache MISS rather than a wrong answer.

    path = keyed_path(base_dir, "z_anchors", source="pg", anchors=anchors)
    # -> <base_dir>/z_anchors__source-pg__anchors-94-a3f19c2b.parquet
"""

import hashlib
import json
from pathlib import Path


def _fingerprint(value) -> str:
    """Stable short digest of an arbitrary parameter value.

    Collections are summarised as <count>-<digest> so the filename stays readable and
    a wrong count is visible without decoding the hash.
    """
    if isinstance(value, (list, tuple, set)):
        items = sorted(str(v) for v in value)
        digest = hashlib.sha256("\x00".join(items).encode()).hexdigest()[:8]
        return f"{len(items)}-{digest}"
    if isinstance(value, dict):
        blob = json.dumps({str(k): str(v) for k, v in sorted(value.items())})
        return hashlib.sha256(blob.encode()).hexdigest()[:8]
    s = str(value)
    if len(s) <= 24 and "/" not in s and " " not in s:
        return s
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def keyed_path(base_dir, stem: str, suffix: str = ".parquet", **params) -> Path:
    """Cache path whose NAME encodes every content-determining parameter.

    Pass every argument the fetch depends on. Anything omitted is, by construction, a
    parameter the cache cannot distinguish - which is the bug this module exists to
    prevent, so omit nothing that changes the result.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    parts = [f"{k}-{_fingerprint(v)}" for k, v in sorted(params.items())]
    return base / (stem + ("__" + "__".join(parts) if parts else "") + suffix)
