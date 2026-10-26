"""
Hand-checked tests for meta_filter.py and backtest_meta.py.
Every expected number below is worked out by hand in the comment next to it.
Run: python -m pytest -q tests/test_meta_filter.py

Independent personal project, not tied to any employer or agency. Synthetic data only.
"""
import os, sys, tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STATE_DIR', tempfile.mkdtemp())
import meta_filter as mf          # noqa: E402
import ai_trading_bot as bot      # noqa: E402
import backtest_meta as bm        # noqa: E402

ET = 'America/New_York'


def day_bars(closes, start='2026-10-21 09:30', opens=None):
    idx = pd.date_range(start, periods=len(closes), freq='15min', tz=ET)
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else closes, float)
    return o, c, (idx.hour * 60 + idx.minute).to_numpy(), idx.normalize().to_numpy(), idx


def test_rules_match_live_bot():
    # the filter must replay the SAME exit rules the live bot uses
    for k, v in mf.RULES.items():
        assert getattr(bot, k) == v, k
    assert bot.NO_NEW_BUYS_AFTER == (15, 45)


def test_stop_loss():
    # entry 100 at bar 1 open; closes 99.5, 98.5, 97.9 -> 97.9 is -2.1% <= -2% -> STOP_LOSS at bar 3
    o, c, m, d, _ = day_bars([100, 99.5, 98.5, 97.9, 101], opens=[100, 100, 99, 98, 98])
    assert mf.simulate_exit(o, c, m, d, 1) == (97.9, 'STOP_LOSS', 3)


def test_breakeven_floor():
    # peak 101.6 (+1.6% >= +1.5%) then close 99.9 <= entry 100 -> exit 99.9 (floor checked on close)
    o, c, m, d, _ = day_bars([100, 101.6, 100.5, 99.9], opens=[100, 100, 101, 100])
    assert mf.simulate_exit(o, c, m, d, 1) == (99.9, 'PROFIT_LOCK', 3)


def test_plus2_floor():
    # peak 104.5 (+4.5%) -> floor +2% = 102; close 101.8 <= 102 -> exit 101.8
    o, c, m, d, _ = day_bars([100, 104.5, 103, 101.8], opens=[100, 100, 104, 103])
    assert mf.simulate_exit(o, c, m, d, 1) == (101.8, 'PROFIT_LOCK', 3)


def test_trailing_stop():
    # peak 110 (+10%): trail = max(110*0.95=104.5, 100*1.03=103) = 104.5; close 104.4 <= 104.5 -> exit
    o, c, m, d, _ = day_bars([100, 106, 110, 107, 104.4], opens=[100, 100, 106, 110, 107])
    assert mf.simulate_exit(o, c, m, d, 1) == (104.4, 'TRAILING_STOP', 4)


def test_eod_exit_at_1545_open():
    # bought 15:15 bar open; nothing triggers; 15:45 bar -> exit at its OPEN (100.7)
    o, c, m, d, idx = day_bars([100, 100.2, 100.4, 100.6, 100.8], start='2026-10-21 14:45',
                               opens=[100, 100.1, 100.3, 100.5, 100.7])
    assert str(idx[4].time()) == '15:45:00'
    assert mf.simulate_exit(o, c, m, d, 2) == (100.7, 'EOD', 4)


def test_net_return_costs():
    # buy 100 * 1.0003 = 100.03 ; sell 102 * 0.9997 = 101.9694 ; 101.9694/100.03 - 1 = 0.019388...
    assert mf.net_return(100, 102) == pytest.approx(101.9694 / 100.03 - 1, abs=1e-12)
    assert mf.net_return(100, 100) < 0          # flat trade loses the costs


def _cand_frame():
    # one day, 9:30..15:45; probability high at 10:00 and 15:30 bars only
    idx = pd.date_range('2026-10-21 09:30', '2026-10-21 15:45', freq='15min', tz=ET)
    n = len(idx)
    df = pd.DataFrame(dict(open=100.0, high=100.5, low=99.5, close=100.0, p=0.1, rsi=50.0,
                           above_ema50=1.0, bb_pct=0.5, vol_ratio=1.0, atr_pct=0.01, adx=20.0,
                           ret_5=0.0, ret_30=0.0, macd_hist=0.0, spy_ret_5=0.0, spy_above_ema20=1.0), index=idx)
    df.loc[idx[2], 'p'] = 0.70      # 10:00 bar -> BUY at 10:15 open
    df.loc[idx[24], 'p'] = 0.70     # 15:30 bar -> too late (after 15:15 decision bar)
    return df, idx


def test_candidates_rule_and_cutoffs():
    df, idx = _cand_frame()
    c = mf.candidates_for_symbol('X', df, 0.62, {})
    assert len(c) == 1 and c[0]['t'] == idx[2] and c[0]['entry_t'] == idx[3]
    assert c[0]['reason'] == 'EOD' and c[0]['exit_t'] == idx[25]           # flat price -> held to 15:45
    assert c[0]['p_margin'] == pytest.approx(0.70 - 0.62)
    assert c[0]['minute'] == 30                                           # 10:00 is 30 min after 9:30


@pytest.mark.parametrize('col,val', [('rsi', 79.0), ('above_ema50', 0.0), ('bb_pct', 0.96)])
def test_candidates_vetoes(col, val):
    df, idx = _cand_frame()
    df.loc[idx[2], col] = val
    assert mf.candidates_for_symbol('X', df, 0.62, {}) == []


def test_candidates_bear_needs_68():
    bear = {pd.Timestamp('2026-10-21', tz=ET): 1}
    df, idx = _cand_frame()
    assert len(mf.candidates_for_symbol('X', df, 0.62, bear)) == 1      # 0.70 >= 0.68 -> buy
    df.loc[idx[2], 'p'] = 0.66
    assert mf.candidates_for_symbol('X', df, 0.62, bear) == []          # 0.66 < 0.68 -> skip in bear
    assert len(mf.candidates_for_symbol('X', df, 0.62, {})) == 1        # same call in a bull market -> buy


def _c(sym, t, exit_t, ret, p, bear=0):
    t = pd.Timestamp(t, tz=ET)
    return dict(sym=sym, t=t, entry_t=t + pd.Timedelta(minutes=15), exit_t=pd.Timestamp(exit_t, tz=ET),
                ret=ret, p=p, bear=bear, reason='EOD')


def test_portfolio_rules():
    C = pd.DataFrame([
        _c('A', '2026-10-21 10:00', '2026-10-21 15:45', 0.01, 0.70),
        _c('B', '2026-10-21 10:00', '2026-10-21 15:45', 0.02, 0.80),   # same bar: only the highest p (B) is bought
        _c('B', '2026-10-21 10:15', '2026-10-21 15:45', 0.05, 0.90),   # already holding B -> skipped
        _c('C', '2026-10-21 10:15', '2026-10-21 15:45', -0.01, 0.65, bear=1),  # bear -> 6% size
    ])
    T = bm.portfolio(C)
    assert [t['sym'] for t in T] == ['B', 'C']
    b, c = T
    assert b['size'] == pytest.approx(12_000) and b['pnl'] == pytest.approx(240)     # 12% of 100k * 2%
    assert c['size'] == pytest.approx(6_000) and c['pnl'] == pytest.approx(-60)      # 6% of 100k * -1%


def test_portfolio_max_five_and_reuse_after_exit():
    # one buy per bar: A..F at 10:00..11:15; all still open at 15:00 -> F would be the 6th position
    rows = [_c(s, f'2026-10-21 {10 + i // 4}:{(i % 4) * 15:02d}', '2026-10-21 15:00', 0.01, 0.7) for i, s in enumerate('ABCDEF')]
    T = bm.portfolio(pd.DataFrame(rows))
    assert [t['sym'] for t in T] == list('ABCDE')            # 6th blocked: 5 still open
    # sizes compound only on realised P&L (none realised before entries) -> all 12,000
    assert all(t['size'] == pytest.approx(12_000) for t in T)


def test_stats_arithmetic():
    T = [dict(pnl=200.0, exit_t=1, reason='EOD'), dict(pnl=-100.0, exit_t=2, reason='STOP_LOSS'),
         dict(pnl=100.0, exit_t=3, reason='EOD'), dict(pnl=-300.0, exit_t=4, reason='STOP_LOSS')]
    s = bm.stats(T)
    assert s['trades'] == 4 and s['wins'] == 2 and s['hit_rate'] == 0.5
    assert s['avg_win'] == 150 and s['avg_loss'] == -200
    assert s['break_even_hit'] == pytest.approx(200 / 350)            # |loss| / (win + |loss|)
    assert s['profit_per_trade'] == -25 and s['total_pnl'] == -100
    assert s['profit_factor'] == pytest.approx(300 / 400)
    # equity 100200 -> 100100 -> 100200 -> 99900 ; peak 100200 ; worst = 99900/100200 - 1
    assert s['max_drawdown'] == pytest.approx(99_900 / 100_200 - 1)
    assert s['exits'] == {'EOD': 2, 'STOP_LOSS': 2}


def test_pick_cut():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    r = np.array([.02, .01, .01, -.01, -.02, .00, -.01, -.02, .01, -.01])
    cut, keep, table = mf.pick_cut(p, r)
    # top 30% = .02,.01,.01 -> avg .01333 (best); top 40% adds -.01 -> .0075 ...
    assert keep == 0.3 and cut == 0.7
    assert table[0]['avg_ret'] == pytest.approx(0.04 / 3)


def test_live_wiring_train_save_score(monkeypatch, tmp_path):
    """Nightly retrain path -> saved filter -> live score, on random data (no network)."""
    syms = ['AAA', 'BBB', 'CCC', 'SPY']
    monkeypatch.setattr(bot, 'fetch_bars', lambda s, days=None, bars=550, complete_only=False: bm.fake_bars(s, days or 60))
    monkeypatch.setattr(bot, 'TRAINING_UNIVERSE', syms)
    monkeypatch.setattr(bot, 'META_FILE', tmp_path / 'meta_filter.pkl')
    monkeypatch.setattr(bot, 'ML_DECISION_THRESHOLD', 0.6)
    spy = bm.fake_bars('SPY', 200)
    ctx = bm.spy_context(spy)
    va = pd.concat([bm.prep(s, bm.fake_bars(s, 200), ctx, i) for i, s in enumerate(syms)]).sort_index()
    probs = np.random.default_rng(1).uniform(0.4, 0.9, len(va))
    bot.train_meta_filter(va, probs)
    assert (tmp_path / 'meta_filter.pkl').exists() and bot.meta_bundle is not None
    b = bot.meta_bundle
    assert b['calls'] >= 60 and 0 < b['keep'] <= 1 and {r['keep'] for r in b['table']} >= {0.3, 1.0}
    # live call
    ml = dict(confidence=0.7, rsi=55, vol_ratio=1.2, bb_pct=0.5, atr_pct=0.01, adx=20, ret_5=0.001,
              ret_30=0.002, macd_hist=0.01, price=100, stock_day_ret=0.003, spy_ret_5=0.001,
              spy_above_ema20=1, minute=60, dow=2)
    p, keep = bot.meta_check('AAA', ml, 0.2, 0.001)
    assert 0 <= p <= 1 and keep == (p >= b['cut'])
    bot.meta_bundle = None
    assert bot.meta_check('AAA', ml, 0.2, 0.001) == (None, True)     # no filter -> never blocks


def test_trade_labels_match_hand_replay():
    # day 14:45..15:45. Bar 0 (14:45) decides -> buy 15:00 open 100 -> nothing triggers -> exit 15:45 open 100.7
    o, c, m, d, idx = day_bars([100, 100.2, 100.4, 100.6, 100.8], start='2026-10-21 14:45',
                               opens=[99.9, 100, 100.3, 100.5, 100.7])
    df = pd.DataFrame(dict(open=o, close=c), index=idx)
    r = mf.trade_labels(df)
    assert r.iloc[0] == pytest.approx(100.7 * 0.9997 / (100 * 1.0003) - 1)
    assert r.iloc[1] == pytest.approx(100.7 * 0.9997 / (100.3 * 1.0003) - 1)   # 15:00 decides, buys 15:15 open


def test_trade_labels_last_decision_bar():
    o, c, m, d, idx = day_bars([100, 100.2, 100.4, 100.6, 100.8], start='2026-10-21 14:45',
                               opens=[99.9, 100, 100.3, 100.5, 100.7])
    r = mf.trade_labels(pd.DataFrame(dict(open=o, close=c), index=idx))
    # bar 2 = 15:15 -> buy 15:30 open 100.5 -> exit 15:45 open 100.7 ; bars 3 (15:30) and 4 (15:45): no buy
    assert r.iloc[2] == pytest.approx(100.7 * 0.9997 / (100.5 * 1.0003) - 1)
    assert r.iloc[3:].isna().all()


def test_calibrate_expectancy():
    p = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    r = np.array([-.01, .002, -.004, .006, .004, .002])
    ok = np.array([1, 1, 1, 1, 1, 0], bool)            # last call vetoed
    th, avg, n, hit = mf.calibrate_expectancy(p, r, ok, min_signals=2)
    # grid steps 0.02: th 0.66, 0.68, 0.70 all select the same calls .006,.004 (0.80 vetoed) -> avg .005 best;
    # ties keep the first (lowest) cutoff -> 0.66. th >= 0.72 -> 1 call (< 2) skipped
    assert th == 0.66 and avg == pytest.approx(0.005) and n == 2 and hit == 1.0


def test_time_exit_after_two_hours():
    # 10:00..13:00 bars. Buy at bar 1 open (10:15, $100). Drifts up but never hits +1.5%.
    # max_hold=8 -> sell at the OPEN of bar 9 (12:15, exactly 2h later) = 100.9
    closes = [100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2]
    opens = [100] * 2 + [c - 0.0 for c in closes[1:-1]]
    opens = [100, 100] + closes[1:11]      # open of bar k = close of bar k-1 (bar 9 open = 100.8)
    o, c, m, d, idx = day_bars(closes, start='2026-10-21 10:00', opens=opens)
    ex, reason, j = mf.simulate_exit(o, c, m, d, 1, max_hold=8)
    assert (reason, j, str(idx[j].time())) == ('TIME_EXIT', 9, '12:15:00') and ex == o[9] == 100.8
    # without the time exit the same trade is still open at bar 12
    assert mf.simulate_exit(o, c, m, d, 1)[1] == 'END_OF_DATA'


def test_time_exit_does_not_override_stop():
    # stop hits at bar 3 (-2.1%) before the 2h mark -> STOP_LOSS, not TIME_EXIT
    o, c, m, d, _ = day_bars([100, 99.5, 98.5, 97.9, 101], opens=[100, 100, 99, 98, 98])
    assert mf.simulate_exit(o, c, m, d, 1, max_hold=8)[1:] == ('STOP_LOSS', 3)


def test_calibrate_rate():
    # 2 days, 4 eligible calls a day at p = .55/.60/.65/.70 ; per_day=1 -> need <= 2 calls/day
    p = np.array([.55, .60, .65, .70] * 2)
    mins = np.array([600] * 8)                                  # 10:00, eligible
    dates = np.array(['d1'] * 4 + ['d2'] * 4)
    th, per = mf.calibrate_rate(p, mins, dates, np.ones(8, bool), per_day=1)
    # th .50-.55 -> 4/day ; .56-.60 -> 3/day ; .61 -> 2/day (first <= 2)
    assert th == 0.61 and per == 2.0
    # calls after 15:15 don't count
    th2, _ = mf.calibrate_rate(p, np.array([960] * 8), dates, np.ones(8, bool), per_day=1)
    assert th2 == 0.50


def test_portfolio_per_day_cap():
    rows = [_c(s, f'2026-10-21 {10 + i // 4}:{(i % 4) * 15:02d}', '2026-10-21 10:50', 0.01, 0.7) for i, s in enumerate('ABCD')]
    rows += [_c('E', '2026-10-22 10:00', '2026-10-22 15:45', 0.01, 0.7)]
    T = bm.portfolio(pd.DataFrame(rows), per_day=2)
    assert [t['sym'] for t in T] == ['A', 'B', 'E']        # 2 on day 1 (first come), then the next day resets


def _cx(sym, t, exit_t, ret, p, entry, exit_, reason):
    d = _c(sym, t, exit_t, ret, p); d.update(entry=entry, exit=exit_, reason=reason); return d


def test_reentry_rules_users_example():
    # Your example: trailing stop sells at $103 (+3%), then the bot re-buys dips that keep falling.
    C = pd.DataFrame([
        _cx('X', '2026-10-21 10:00', '2026-10-21 11:00', 0.03, 0.7, 100.0, 103.0, 'TRAILING_STOP'),
        _cx('X', '2026-10-21 11:30', '2026-10-21 12:00', -0.02, 0.7, 102.0, 99.96, 'STOP_LOSS'),   # dip re-buy below $103
        _cx('X', '2026-10-21 12:30', '2026-10-21 13:00', -0.02, 0.7, 99.0, 97.02, 'STOP_LOSS'),    # second dip re-buy
    ])
    free = bm.portfolio(C)
    # 12% of equity, compounding: 12% of 100,000 x 3% = +360 ; 12% of 100,360 x -2% = -240.86 ; 12% of 100,119.14 x -2% = -240.29
    assert [round(t['pnl']) for t in free] == [360, -241, -240]
    assert round(sum(t['pnl'] for t in free)) == round(360 - 0.02 * 0.12 * 100360 - 0.02 * 0.12 * (100360 - 0.02 * 0.12 * 100360))
    guarded = bm.portfolio(C, reentry_rules=True)
    assert [t['sym'] for t in guarded] == ['X'] and round(guarded[0]['pnl']) == 360   # both dip re-buys blocked (below $103)


def test_reentry_rules_allow_strength_once_half_size():
    C = pd.DataFrame([
        _cx('X', '2026-10-21 10:00', '2026-10-21 11:00', 0.03, 0.7, 100.0, 103.0, 'TRAILING_STOP'),
        _cx('X', '2026-10-21 11:30', '2026-10-21 12:00', 0.01, 0.7, 103.5, 104.5, 'EOD'),          # above $103 -> allowed, half size
        _cx('X', '2026-10-21 12:30', '2026-10-21 13:00', 0.01, 0.7, 105.0, 106.0, 'EOD'),          # 2nd re-entry -> blocked
    ])
    T = bm.portfolio(C, reentry_rules=True)
    assert len(T) == 2 and T[1]['size'] == pytest.approx(0.5 * 0.12 * 100_360)


def test_reentry_rules_one_strike_and_late():
    C = pd.DataFrame([
        _cx('X', '2026-10-21 10:00', '2026-10-21 10:30', -0.02, 0.7, 100.0, 98.0, 'STOP_LOSS'),
        _cx('X', '2026-10-21 11:00', '2026-10-21 12:00', 0.02, 0.7, 101.0, 103.0, 'EOD'),          # struck -> blocked
        _cx('Y', '2026-10-21 10:15', '2026-10-21 10:45', 0.02, 0.6, 50.0, 51.0, 'PROFIT_LOCK'),
        _cx('Y', '2026-10-21 14:45', '2026-10-21 15:45', 0.01, 0.7, 52.0, 52.5, 'EOD'),            # after 14:30 -> blocked
        _cx('X', '2026-10-22 10:00', '2026-10-22 15:45', 0.01, 0.7, 101.0, 102.0, 'EOD'),          # new day -> fine
    ])
    T = bm.portfolio(C, reentry_rules=True)
    assert [(t['sym'], str(pd.Timestamp(t['t']).date())) for t in T] == [('X', '2026-10-21'), ('Y', '2026-10-21'), ('X', '2026-10-22')]


def test_classic_day_label_never_crosses_the_close():
    raw = bm.fake_bars('AAA', 30)
    ctx = bm.spy_context(bm.fake_bars('SPY', 30))
    df = bm.prep('AAA', raw, ctx, 0, label='classic_day')
    m = df.index.hour * 60 + df.index.minute
    lab = df['target'].notna()
    assert lab.any() and m[lab].max() <= 13 * 60 + 45 and m[lab].min() >= 9 * 60 + 30
    live = bm.prep('AAA', raw, ctx, 0, label='classic')      # today's live label DOES label 14:00-15:45 bars
    m2 = live.index.hour * 60 + live.index.minute
    assert (live['target'].notna() & (m2 >= 14 * 60)).any()


def test_buy_window_10_to_2():
    # high-probability calls at every bar 9:30..15:15; window 10:00-14:00 is about the BUY time (decision + 15 min)
    idx = pd.date_range('2026-10-21 09:30', '2026-10-21 15:45', freq='15min', tz=ET)
    df = pd.DataFrame(dict(open=100.0, high=100.5, low=99.5, close=100.0, p=0.9, rsi=50.0, above_ema50=1.0,
                           bb_pct=0.5, vol_ratio=1.0, atr_pct=0.01, adx=20.0, ret_5=0.0, ret_30=0.0, macd_hist=0.0,
                           spy_ret_5=0.0, spy_above_ema20=1.0), index=idx)
    c = mf.candidates_for_symbol('X', df, 0.62, {}, buy_window=(600, 840))
    buys = [str(x['entry_t'].time())[:5] for x in c]
    assert buys[0] == '10:00' and buys[-1] == '14:00' and len(buys) == 17          # 10:00, 10:15 ... 14:00
    c11 = mf.candidates_for_symbol('X', df, 0.62, {}, buy_window=(660, 840))
    assert [str(x['entry_t'].time())[:5] for x in c11][0] == '11:00' and len(c11) == 13
    assert len(mf.candidates_for_symbol('X', df, 0.62, {})) == 24                   # no window: 9:45 .. 15:30 buys


def test_same_seed_same_model():
    spy = bm.fake_bars('SPY', 120); ctx = bm.spy_context(spy)
    train = pd.concat([bm.prep(s, bm.fake_bars(s, 120), ctx, i) for i, s in enumerate(['AAA', 'BBB'])])
    b1, i1 = bm.train_primary(train, seed=7); b2, i2 = bm.train_primary(train, seed=7)
    X = train[bot.FEATURES + ['sym_id']].iloc[:500]
    assert np.array_equal(b1(X), b2(X)) and i1['threshold'] == i2['threshold']


def test_no_buy_across_a_data_gap():
    # bars 9:30..11:15, then nothing until 14:30 (thin IEX data), then to 15:45. A call at 11:15 must NOT buy at 14:30.
    idx = list(pd.date_range('2026-10-21 09:30', '2026-10-21 11:15', freq='15min', tz=ET)) + \
          list(pd.date_range('2026-10-21 14:30', '2026-10-21 15:45', freq='15min', tz=ET))
    df = pd.DataFrame(dict(open=100.0, high=100.5, low=99.5, close=100.0, p=0.1, rsi=50.0, above_ema50=1.0, bb_pct=0.5,
                           vol_ratio=1.0, atr_pct=0.01, adx=20.0, ret_5=0.0, ret_30=0.0, macd_hist=0.0, spy_ret_5=0.0,
                           spy_above_ema20=1.0), index=pd.DatetimeIndex(idx))
    df.loc[pd.Timestamp('2026-10-21 11:15', tz=ET), 'p'] = 0.9
    assert mf.candidates_for_symbol('X', df, 0.62, {}) == []
    assert mf.trade_labels(df).loc[pd.Timestamp('2026-10-21 11:15', tz=ET)] != mf.trade_labels(df).loc[pd.Timestamp('2026-10-21 11:15', tz=ET)]  # NaN


def test_portfolio_now_is_bar_time_plus_15():
    # A gapped row's late entry_t must not release open positions early (the TSLA double-hold case).
    C = pd.DataFrame([
        _c('T', '2026-10-21 10:45', '2026-10-21 14:00', -0.02, 0.9),
        dict(_c('B', '2026-10-21 11:15', '2026-10-21 15:15', 0.0, 0.8), entry_t=pd.Timestamp('2026-10-21 14:45', tz=ET)),
        _c('T', '2026-10-21 13:30', '2026-10-21 14:15', -0.02, 0.9),      # T still held until 14:00 -> must be skipped
    ])
    assert [t['sym'] for t in bm.portfolio(C)] == ['T', 'B']


def test_drawdown_uses_end_of_day_balances():
    T = [dict(pnl=500.0, exit_t='2026-10-21 15:45', reason='EOD'), dict(pnl=-800.0, exit_t='2026-10-21 15:45', reason='EOD'),
         dict(pnl=-200.0, exit_t='2026-10-22 15:45', reason='EOD')]
    # day balances 99,700 then 99,500 -> worst 99,500/100,000 - 1, whatever the order within the first day
    assert bm.stats(T)['max_drawdown'] == pytest.approx(-0.005)
    assert bm.stats([T[1], T[0], T[2]])['max_drawdown'] == pytest.approx(-0.005)
