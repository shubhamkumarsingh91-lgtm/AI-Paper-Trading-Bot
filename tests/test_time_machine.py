"""
Hand-checked tests for time_machine.py and sample_method_ab_testing.py.
Every expected number is worked out by hand in the comment next to it.
Run: python -m pytest -q tests/test_time_machine.py

Independent personal project, not tied to any employer or agency. Synthetic data only.
"""
import os, sys, tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STATE_DIR', tempfile.mkdtemp())
import meta_filter as mf                    # noqa: E402
import time_machine as tm                   # noqa: E402
import sample_method_ab_testing as ab       # noqa: E402

ET = 'America/New_York'


def one_day(date, closes, opens=None):
    """26 regular-hours 15-min bars on `date`."""
    idx = pd.date_range(f'{date} 09:30', periods=len(closes), freq='15min', tz=ET)
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else np.r_[c[0], c[:-1]], float)
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c), low=np.minimum(o, c), close=c, volume=1.0), index=idx)


# ── DayTable ─────────────────────────────────────────────────────────────────

def test_day_table_returns_and_skips_incomplete_days():
    closes = 100 * 1.01 ** np.arange(1, 27)          # each close is +1% on the one before
    opens = np.r_[100.0, closes[:-1]]                 # no gaps: open = previous close
    good = one_day('2026-10-19', closes, opens)
    half = one_day('2026-10-20', closes[:13])         # half day (13 bars) -> skipped
    t = tm.DayTable(pd.concat([good, half]))
    assert list(t.days) == [pd.Timestamp('2026-10-19', tz=ET)]
    # every bar moves +1%: log(1.01) = 0.00995033 for both open->close and close->close
    assert np.allclose(t.oc[0], np.log(1.01)) and np.allclose(t.cc[0], np.log(1.01))


def test_pool_uses_only_days_before_the_decision_day():
    closes = np.full(26, 100.0)
    bars = pd.concat([one_day(d, closes) for d in ['2026-10-19', '2026-10-20', '2026-10-21', '2026-10-22']])
    t = tm.DayTable(bars)
    # deciding on 10-21: pool = rows of 10-19 and 10-20 (rows 0, 1); never 10-21 or 10-22
    assert list(t.pool_before(pd.Timestamp('2026-10-21', tz=ET))) == [0, 1]
    # lookback 1 -> only the most recent day before 10-22, which is 10-21 (row 2)
    assert list(t.pool_before(pd.Timestamp('2026-10-22', tz=ET), lookback=1)) == [2]


# ── volatility factor ────────────────────────────────────────────────────────

def test_vol_factor_by_hand_and_clipped():
    pool = np.array([[0.01, -0.01], [0.01, -0.01]])   # std = 0.01
    # today +-0.02 -> std 0.02 -> factor 0.02 / 0.01 = 2.0
    assert tm.vol_factor([0.02, -0.02, 0.02, -0.02], pool) == pytest.approx(2.0)
    # today +-0.05 -> 5.0, clipped to 2.0;  today +-0.001 -> 0.1, clipped to 0.5
    assert tm.vol_factor([0.05, -0.05, 0.05, -0.05], pool) == pytest.approx(2.0)
    assert tm.vol_factor([0.001, -0.001, 0.001, -0.001], pool) == pytest.approx(0.5)
    # fewer than 3 bars today -> no adjustment
    assert tm.vol_factor([0.05, -0.05], pool) == 1.0


# ── path simulation ──────────────────────────────────────────────────────────

def test_day_method_replays_one_whole_past_day():
    oc = np.zeros((1, 26)); cc = np.zeros((1, 26))
    oc[0, 20] = np.log(1.02)                           # entry bar 20 open->close: +2%
    cc[0, 21:25] = np.log([0.99, 1.0, 1.01, 1.0])
    p = tm.simulate_paths(oc, cc, 20, 'day', 3, np.random.default_rng(0))
    # 5 bars (20..24): 1.02, 1.02*0.99=1.0098, 1.0098, 1.0098*1.01=1.019898, 1.019898
    assert p.shape == (3, 5)
    assert np.allclose(p[0], [1.02, 1.0098, 1.0098, 1.019898, 1.019898])


def test_slot_method_mixes_days_and_day_method_does_not():
    # day 0 moves +1% every bar, day 1 moves -1% every bar
    oc = np.vstack([np.full(26, np.log(1.01)), np.full(26, np.log(0.99))])
    cc = oc.copy()
    rng = np.random.default_rng(1)
    day = tm.simulate_paths(oc, cc, 10, 'day', 500, rng)
    step = day[:, 1:] / day[:, :-1]
    # a whole-day path never changes direction mid-path
    assert np.all(np.isclose(step, 1.01).all(1) | np.isclose(step, 0.99).all(1))
    slot = tm.simulate_paths(oc, cc, 10, 'slot', 500, np.random.default_rng(1))
    s = slot[:, 1:] / slot[:, :-1]
    mixed = ~(np.isclose(s, 1.01).all(1) | np.isclose(s, 0.99).all(1))
    assert mixed.mean() > 0.99                         # 14 independent coin flips: almost every path mixes


def test_volatility_factor_scales_moves():
    oc = np.full((1, 26), np.log(1.01)); cc = oc.copy()
    p = tm.simulate_paths(oc, cc, 23, 'day', 1, np.random.default_rng(0), factor=2.0)
    # 2 bars, each log(1.01)*2 -> 1.01^2 = 1.0201, then 1.0201^2 = 1.04060401
    assert np.allclose(p[0], [1.0201, 1.04060401])


# ── exit rules ───────────────────────────────────────────────────────────────

def test_exits_hand_cases():
    paths = np.array([
        [0.995, 0.985, 0.979, 1.05],     # -2.1% at bar 3 -> STOP_LOSS at 0.979, 3 bars
        [1.016, 1.005, 0.999, 1.0],      # peak +1.6% -> breakeven lock; 0.999 <= 1.0 -> PROFIT_LOCK, 3 bars
        [1.045, 1.03, 1.019, 1.0],       # peak +4.5% -> +2% floor; 1.019 <= 1.02 -> PROFIT_LOCK, 3 bars
        [1.07, 1.10, 1.044, 1.2],        # peak +10% -> trail max(1.10*0.95=1.045, 1.03); 1.044 -> TRAILING_STOP
        [1.001, 1.002, 1.003, 1.004],    # nothing fires -> EOD at the last close 1.004, 4 bars
    ])
    px, why, held = tm.exits_vectorized(paths)
    assert np.allclose(px, [0.979, 0.999, 1.019, 1.044, 1.004])
    assert [tm.REASONS[w] for w in why] == ['STOP_LOSS', 'PROFIT_LOCK', 'PROFIT_LOCK', 'TRAILING_STOP', 'EOD']
    assert list(held) == [3, 3, 3, 3, 4]


def test_exits_match_the_live_rules_on_random_paths():
    # 3,000 random paths through the vectorized rules must give the SAME exit price and
    # reason as meta_filter.simulate_exit (the function the live backtests use)
    rng = np.random.default_rng(42)
    for _ in range(3000):
        e = int(rng.integers(1, 25))                   # entry slot 1..24
        path = np.exp(np.cumsum(rng.normal(0, 0.012, 25 - e)))
        px, why, held = tm.exits_vectorized(path[None, :])
        c = np.ones(26); c[e:25] = path; c[25] = path[-1]
        o = np.r_[1.0, c[:-1]]; o[e] = 1.0             # bought at 1.0 at the open of slot e
        idx = pd.date_range('2026-10-21 09:30', periods=26, freq='15min', tz=ET)
        ex, reason, j = mf.simulate_exit(o, c, mf.minutes_of(idx), idx.normalize().to_numpy(), e)
        assert px[0] == pytest.approx(ex) and tm.REASONS[why[0]] == reason
        if reason != 'EOD':
            assert held[0] == j - e + 1


def test_summarize_by_hand():
    px = np.array([1.01, 0.98, 1.0, 1.02])
    why = np.array([tm.REASONS.index(r) for r in ('EOD', 'STOP_LOSS', 'PROFIT_LOCK', 'EOD')])
    held = np.array([10, 2, 5, 10])
    s = tm.summarize(px, why, held, cost=0.0)
    # wins: 1.01 and 1.02 (1.0 is not > 0) -> 2/4; one stop -> 1/4; two EOD -> 2/4
    # mean return (0.01 - 0.02 + 0 + 0.02)/4 = 0.0025; median of 2,5,10,10 = 7.5
    assert s == dict(p_win=0.5, p_stop=0.25, p_eod=0.5, exp_ret=pytest.approx(0.0025), med_bars=7.5)


# ── no peeking at the future ─────────────────────────────────────────────────

def _history(n_days=30, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2026-09-01', periods=n_days)
    return pd.concat([one_day(d.date(), 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 26)))) for d in days])


def test_features_ignore_bars_after_the_decision():
    bars = _history()
    t = bars.index[bars.index.normalize() == bars.index.normalize()[-1]][8]   # last day, 11:30 bar
    cands = pd.DataFrame(dict(sym=['X'], t=[t]))
    f1 = tm.add_features(cands, {'X': bars}, n_paths=400)
    later = bars.copy()
    later.loc[later.index > t, ['open', 'high', 'low', 'close']] *= 1.5       # rewrite the future
    f2 = tm.add_features(cands, {'X': later}, n_paths=400)
    cols = tm.feature_names('slot') + tm.feature_names('day')
    assert f1[cols].equals(f2[cols])                                          # identical: no leak


def test_features_are_repeatable_and_nan_without_history():
    bars = _history()
    t = bars.index[-10]
    c = pd.DataFrame(dict(sym=['X'], t=[t]))
    a = tm.add_features(c, {'X': bars}, n_paths=300)
    b = tm.add_features(c, {'X': bars}, n_paths=300)
    assert a.equals(b)                                                        # fixed seed per call
    early = pd.DataFrame(dict(sym=['X'], t=[bars.index[30]]))                 # day 2: pool has 1 day < 15
    assert tm.add_features(early, {'X': bars}, n_paths=300)['tms_p_win'].isna().all()


# ── A/B helpers ──────────────────────────────────────────────────────────────

def test_call_stats_and_window():
    c = pd.DataFrame(dict(ret=[0.02, -0.01, 0.0, 0.01],
                          t=pd.to_datetime(['2026-10-21 09:45', '2026-10-21 10:00',
                                            '2026-10-21 13:45', '2026-10-21 14:00']).tz_localize(ET)))
    s = ab.call_stats(c)
    # winners 0.02 and 0.01 -> precision 2/4 = 0.5; avg 0.02/4 = 0.005; PF 0.03 / 0.01 = 3.0
    assert s['n'] == 4 and s['precision'] == 0.5
    assert s['avg_ret'] == pytest.approx(0.005) and s['profit_factor'] == pytest.approx(3.0)
    # window 10-14 keeps 10:00 and 13:45; drops 09:45 and 14:00 (end is exclusive)
    assert list(ab.in_window(c, (10, 14))['ret']) == [-0.01, 0.0]


def test_arms_differ_only_by_time_machine_features():
    assert ab.ARMS['base'] == mf.META_FEATURES
    assert ab.ARMS['slot'] == mf.META_FEATURES + tm.feature_names('slot')
    assert ab.ARMS['day'] == mf.META_FEATURES + tm.feature_names('day')
    assert 'atr_pct' in ab.ARMS['base']          # the base arm already is the plain-ATR control


def test_luck_check_observed_gap_by_hand():
    # 2 days. Base keeps all 4 calls (2 winners) -> 50%. Arm keeps the 2 winners + 1 loser -> 66.7%.
    # observed gain = 2/3 - 1/2 = +16.7 points
    T = pd.DataFrame(dict(
        t=pd.to_datetime(['2026-10-20 10:00', '2026-10-20 11:00', '2026-10-21 10:00', '2026-10-21 11:00']).tz_localize(ET),
        ret=[0.01, -0.01, 0.01, -0.01], keep_base=[True] * 4, keep_slot=[True, True, True, False]))
    b = ab.bootstrap_vs_base(T, 'slot', n_boot=500)
    assert b['precision_gain'] == pytest.approx(2 / 3 - 1 / 2) and b['test_days'] == 2
    # every resample is ahead or tied (the arm only ever drops a loser), never behind
    assert b['range_90'][0] >= 0
