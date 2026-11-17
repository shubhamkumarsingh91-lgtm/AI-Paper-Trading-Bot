"""
Checks for backtest_bear.py: "apply the bear bar afterwards" must give EXACTLY the calls the
live rule (bear bar applied inside candidates_for_symbol) gives.
Run: python -m pytest -q tests/test_backtest_bear.py

Independent personal project, not tied to any employer or agency. Synthetic data only.
"""
import os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STATE_DIR', tempfile.mkdtemp())
import meta_filter as mf          # noqa: E402
import backtest_meta as bm        # noqa: E402
import backtest_bear as bb        # noqa: E402

ORIG = mf.candidates_for_symbol    # keep the real function (backtest_bear patches it only inside main)


def synthetic(sym='AAA', seed=3):
    raw = bm.fake_bars(sym, 120, seed_base=seed)
    spy = bm.fake_bars('SPY', 120)
    df = bm.prep(sym, raw, bm.spy_context(spy), 0)
    rng = np.random.default_rng(seed)
    df['p'] = rng.uniform(0.55, 0.75, len(df))              # made-up main-AI confidence
    days = sorted(df.index.normalize().unique())
    bear = {pd.Timestamp(d): int(i % 2 == 0) for i, d in enumerate(days)}   # every other day is a bear day
    df['rsi'] = 50.0; df['above_ema50'] = 1.0; df['bb_pct'] = 0.5         # let the confidence rule decide
    return df, bear


def keys(rows):
    return sorted((r['sym'], r['t']) for r in rows)


def test_subset_equals_live_rule():
    df, bear = synthetic()
    thr = 0.60
    everything = pd.DataFrame(ORIG('AAA', df, thr, bear, bear_min_conf=0.0))
    assert (everything['bear'] == 1).any() and (everything['bear'] == 0).any()
    for bar in (0.68, 0.66, 0.64, 0.62, 0.0, 1.01):
        live = ORIG('AAA', df, thr, bear, bear_min_conf=bar)
        mine = bb.apply_bar(everything, bar)
        assert keys(live) == sorted(zip(mine['sym'], mine['t'])), bar


def test_bar_below_cutoff_changes_nothing():
    # if the bear bar is below the normal cutoff, max(thr, bar) = thr -> same calls as no bar at all
    df, bear = synthetic(seed=5)
    a = ORIG('AAA', df, 0.66, bear, bear_min_conf=0.62)
    b = ORIG('AAA', df, 0.66, bear, bear_min_conf=0.0)
    assert keys(a) == keys(b)


def test_bear_size_used_only_on_bear_days():
    # two calls, same bar time, each +1%: bear one sized 6%, normal one 12%
    t = pd.Timestamp('2026-10-21 10:00', tz='America/New_York')
    base = dict(entry_t=t + pd.Timedelta(minutes=15), exit_t=t + pd.Timedelta(hours=1),
                ret=0.01, reason='EOD', entry=100.0, exit=101.0)
    c = pd.DataFrame([dict(sym='BEAR', t=t, p=0.70, bear=1, **base),
                      dict(sym='BULL', t=t + pd.Timedelta(minutes=15), p=0.70, bear=0,
                           **{**base, 'entry_t': t + pd.Timedelta(minutes=30)})])
    tr = bm.portfolio(c, bear_cap=0.06)
    pnl = {x['sym']: round(x['pnl'], 2) for x in tr}
    # BEAR: $100k * 6% * 1% = $60.  BULL buys while BEAR is still open, so equity is still $100k:
    # $100k * 12% * 1% = $120
    assert pnl == {'BEAR': 60.0, 'BULL': 120.0}


def test_no_bear_buys_means_zero_bear_calls():
    df, bear = synthetic(seed=7)
    everything = pd.DataFrame(ORIG('AAA', df, 0.60, bear, bear_min_conf=0.0))
    kept = bb.apply_bar(everything, 1.01)
    assert (kept['bear'] == 0).all() and len(kept) == (everything['bear'] == 0).sum()
