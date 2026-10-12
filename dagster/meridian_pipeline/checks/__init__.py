"""Asset-check registry for the showcase code location.

ALL_FACTOR_CHECKS is the path by which standalone `@asset_check` objects reach
a `Definitions` object. The private repository registers a dozen more checks
here (factor coverage, group presence and several model-specific checks);
the showcase ships the three that guard the data the whole pipeline stands on:

  check_ohlcv_read_contract     BLOCKING read-boundary contract on the OHLCV tape
  check_ret1d_input_quality     tripwire for upstream ret_1d corruption
  check_trading_day_invariant   detective control for the trading-day-only rule
"""

from meridian_pipeline.checks.ohlcv_contract_checks import (
    check_ohlcv_read_contract,
)
from meridian_pipeline.checks.ret1d_checks import (
    check_ret1d_input_quality,
)
from meridian_pipeline.checks.trading_day_checks import (
    check_trading_day_invariant,
)

ALL_FACTOR_CHECKS = [
    check_ohlcv_read_contract,
    check_ret1d_input_quality,
    check_trading_day_invariant,
]
