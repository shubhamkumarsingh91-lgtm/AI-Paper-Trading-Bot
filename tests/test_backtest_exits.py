"""
Hand-checked tests for backtest_exits.py (middle-ground "carry winners" exit).
Run: python -m pytest -q tests/test_backtest_exits.py

Independent personal project, not tied to any employer or agency. Synthetic data only.
"""
import os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STATE_DIR', tempfile.mkdtemp())
import meta_filter as mf          # noqa: E402
import backtest_exits as bx       # noqa: E402

ET = 'America/New_York'


def two_days(day1, day2, open2=None):
    """Regular-hours 15-min bars 9:30..15:45 for 2 days. dayN = list of 26 closes; opens = previous
    close except the first bar of each day (open2 = day-2 gap open)."""
    idx = []
    for d in ('2026-10-21', '2026-10-22'):
        idx += list(pd.date_range(f'{d} 09:30', periods=26, freq='15min', tz=ET))
    idx = pd.DatetimeIndex(idx)
    c = np.array(day1 + day2, float)
    o = np.r_[c[0], c[:-1]].copy()
    if open2 is not None:
        o[26] = open2
    return o, c, (idx.hour * 60 + idx.minute).to_numpy(), idx.normalize().to_numpy()


def test_no_carry_is_identical_to_live_exit():
    rng = np.random.default_rng(1)
    for _ in range(300):
        day1 = list(100 * np.exp(np.cumsum(rng.normal(0, 0.006, 26))))
        day2 = list(day1[-1] * np.exp(np.cumsum(rng.normal(0, 0.006, 26))))
        o, c, m, d = two_days(day1, day2)
        i = int(rng.integers(1, 24))
        assert bx.make_exit(None)(o, c, m, d, i) == mf.simulate_exit(o, c, m, d, i)


def test_winner_carried_then_gap_down_sold_at_open():
    # buy at bar 1 open = 100. Closes drift to 101.2 (+1.2%: below the +1.5% breakeven lock).
    # 15:45 bar (index 25) opens at 101.2 -> +1.2% >= +1% -> carried overnight.
    # Day 2 opens at 99.0 (below the 100 breakeven stop) -> GAP_STOP at 99.0 on bar 26.
    day1 = [100.0] * 2 + [101.2] * 24
    o, c, m, d = two_days(day1, [99.0] * 26, open2=99.0)
    assert bx.make_exit(0.01)(o, c, m, d, 1) == (99.0, 'GAP_STOP', 26)
    # live rule: sold at the 15:45 open = 101.2, end of day
    assert bx.make_exit(None)(o, c, m, d, 1) == (101.2, 'EOD', 25)


def test_winner_carried_then_falls_back_to_breakeven():
    # same day 1 (+1.2% at 15:45 -> carried). Day 2 opens 101.5 (above stop), first close 101.0,
    # second close 99.9 -> pct -0.1% <= 0 breakeven floor -> CARRY_STOP at 99.9 on bar 27
    day2 = [101.0, 99.9] + [99.9] * 24
    o, c, m, d = two_days([100.0] * 2 + [101.2] * 24, day2, open2=101.5)
    assert bx.make_exit(0.01)(o, c, m, d, 1) == (99.9, 'CARRY_STOP', 27)


def test_small_winner_not_carried_at_one_percent_but_carried_at_half():
    # +0.8% at 15:45: not enough for carry >= +1% (sold EOD at 100.8), enough for +0.5%
    day1 = [100.0] * 2 + [100.8] * 24
    o, c, m, d = two_days(day1, [102.0] * 26, open2=101.0)
    assert bx.make_exit(0.01)(o, c, m, d, 1) == (100.8, 'EOD', 25)
    # carried; day 2 never falls to the stop; up +2% < +4% so only the breakeven lock (+1.5%) is
    # armed and never hit; 15:45 day 2 is day_ix 1 < 2 and +2% >= +0.5% -> carried again, but the
    # data ends -> sold at that 15:45 open (102.0) as EOD
    assert bx.make_exit(0.005)(o, c, m, d, 1) == (102.0, 'EOD', 51)


def test_loser_never_carried():
    # -1% at 15:45 -> sold at the close like live, even with carry on
    day1 = [100.0] * 2 + [99.0] * 24
    o, c, m, d = two_days(day1, [105.0] * 26, open2=105.0)
    assert bx.make_exit(0.005)(o, c, m, d, 1) == (99.0, 'EOD', 25)
