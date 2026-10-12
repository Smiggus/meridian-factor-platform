#!/usr/bin/env python3
"""
parity_rescore.py - parity gate for the ret_1d/close bulk rescore (Meridian #275).

Proves the burst-box compute (scripts/burst/rescore_ret1d_close.py) reproduces
what the in-cluster daily pipeline produces, by correlating a sample of staged
score dates against the corresponding rows already in factors.daily_scores.

Use RECENT dates whose gold rows were written by the current daily pipeline,
so a faithful burst compute lands corr ~1.0 per factor. (Do NOT
parity-check against older gold written before the inputs changed: those rows
are exactly what the rerun replaces, so a low corr there is expected, not a
failure.)

Reads staged scores from the run's MinIO prefix and gold from PostgreSQL, joins
on (ticker, factor_name), and reports per-factor Pearson corr on BOTH raw_value
and z_score, plus membership overlap. Read-only: writes nothing.

WHY BOTH COLUMNS (#294): z_score is a rank transform (OLS sector-neutralisation
then rank-normalise), so a strictly monotone error in the raw construction - a
units change, a scale factor, a monotone formula drift - reproduces gold's
z_scores exactly and sails through a z-only correlation gate. raw_value is the
pre-transform economic quantity, where that class of drift is visible. A factor
fails if EITHER column correlates below the threshold.

    cd ~/Meridian && set -a && . .env && set +a
    python scripts/burst/parity_rescore.py --run-id <parity-run-id>
"""

import argparse
import io
import os
import sys
from urllib.parse import quote_plus

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

_REPO = os.environ.get("MERIDIAN_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO, "dagster"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "burst"))

from burst_common import BurstConfig  # noqa: E402
from meridian_pipeline.pg_identity import BURST_APPLICATION_NAME, pg_connect_args  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--min-corr",
        type=float,
        default=0.98,
        help="minimum per-factor Pearson correlation, applied to raw_value AND z_score",
    )
    args = ap.parse_args()
    min_corr = args.min_corr

    cfg = BurstConfig.from_env(args.run_id)
    s3 = cfg.s3_client()
    url = (
        f"postgresql+psycopg2://{cfg.pg_user}:{quote_plus(cfg.pg_password)}"
        f"@{cfg.pg_host}:{cfg.pg_port}/{cfg.gold_db}"
    )
    eng = create_engine(url, connect_args=pg_connect_args(BURST_APPLICATION_NAME))

    prefix = f"{cfg.run_root}/scores/"
    keys = [
        o["Key"]
        for pg in s3.get_paginator("list_objects_v2").paginate(
            Bucket=cfg.minio_bucket, Prefix=prefix
        )
        for o in pg.get("Contents", [])
        if o["Key"].endswith("scores.parquet")
    ]
    if not keys:
        print(f"no staged scores under {prefix}")
        return 2

    overall_ok = True
    for k in sorted(keys):
        d = k.split("score_date=")[1].split("/")[0]
        body = s3.get_object(Bucket=cfg.minio_bucket, Key=k)["Body"].read()
        staged = pq.read_table(io.BytesIO(body)).to_pandas()[
            ["ticker", "factor_name", "raw_value", "z_score"]
        ]
        factors = sorted(staged["factor_name"].unique().tolist())
        with eng.connect() as c:
            gold = pd.read_sql(
                text(
                    "SELECT ticker, factor_name, raw_value, z_score "
                    "FROM factors.daily_scores "
                    "WHERE score_date = :d AND factor_name = ANY(:f)"
                ),
                c,
                params={"d": d, "f": factors},
            )
        print(f"\n=== {d}: staged {len(staged)} rows, gold {len(gold)} rows ===")
        for fac in factors:
            sf = staged[staged["factor_name"] == fac].set_index("ticker")
            gf = gold[gold["factor_name"] == fac].set_index("ticker")
            line = f"  {fac:26s}"
            worst_n = 0
            low = False
            for col in ("raw_value", "z_score"):
                j = pd.concat({"st": sf[col], "gd": gf[col]}, axis=1).dropna()
                worst_n = max(worst_n, len(j))
                if len(j) < 5:
                    line += f"  {col}: n={len(j):5d} (too few)"
                    continue
                # A constant column (every ticker identical) has undefined
                # correlation - report it rather than silently passing NaN.
                if j["st"].nunique() < 2 or j["gd"].nunique() < 2:
                    same = bool((j["st"] - j["gd"]).abs().max() < 1e-9)
                    line += f"  {col}: constant {'match' if same else 'MISMATCH'}"
                    low = low or not same
                    continue
                corr = j["st"].corr(j["gd"])
                bad = corr is None or not (corr > min_corr)
                low = low or bad
                line += f"  {col}: n={len(j):5d} corr={corr:.4f}"
            if low:
                overall_ok = False
                line += "  <-- LOW"
            print(line)

    print("\nPARITY", "PASS" if overall_ok else f"REVIEW (some factors < {min_corr})")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
