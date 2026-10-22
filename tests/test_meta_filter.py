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
