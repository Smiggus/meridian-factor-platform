"""Cache file names that encode the parameters of the fetch behind them.

Several research scripts had a cache like this:

    def _fetch(con, tickers, start, end, cache: Path):
        if cache is not None and cache.exists():
            return pl.read_parquet(cache)      # tickers/start/end IGNORED
        ...

The file name is fixed but the function takes parameters, so a second call with a different
window gets the first call's data back. Nothing fails. The numbers are a real result for
a different question, and the only sign is a row count that does not fit the window you
asked for. It got through review more than once. A `--force` flag does not help, since
nobody thinks to pass it.

keyed_path puts every parameter that decides the content into the file name, so changing
one is a cache miss.

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
    """Cache path whose name encodes the content-determining parameters.

    Pass every argument the fetch depends on. The cache cannot tell apart two calls that
    differ only in a parameter you left out.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    parts = [f"{k}-{_fingerprint(v)}" for k, v in sorted(params.items())]
    return base / (stem + ("__" + "__".join(parts) if parts else "") + suffix)
