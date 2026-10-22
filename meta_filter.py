#!/usr/bin/env python3
"""
META FILTER ("meta-labeling") — a second, small AI that checks the main AI's BUY calls.

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Public market data and Alpaca PAPER (fake money) accounts only.
No internal, personal or protected data (no SSNs/PHI).

THE IDEA (from Marcos López de Prado, "Advances in Financial Machine Learning", ch. 3)
  The main model answers "is this a buy?". The filter answers a different question:
  "of the buys the main model makes, which ones actually end up making money?"
  It sees things the main model doesn't weigh directly: how sure the main model was,
  the time of day, how the whole market is doing today, how many other stocks are
  flashing BUY at the same moment, and so on. It throws out the weaker calls, so the
  bot makes fewer trades but a larger share of them win (higher precision).

HOW A CANDIDATE TRADE IS JUDGED
  Every BUY call is replayed with the bot's real exit rules (same as
  manage_open_positions in ai_trading_bot.py, checked on each 15-minute bar close):
    - stop-loss at -2% while the peak stayed under +1.5%
    - breakeven floor once the peak reached +1.5%
    - +2% floor once the peak reached +4%
    - trailing stop 5% under the peak (never below +3%) once the peak reached +6%
    - everything closed at the 15:45 bar (the bot closes at 15:50)
  Win = the trade made money after trading costs. That is the filter's label.

This file only holds pure functions. backtest_meta.py uses them to test the idea on
history; ai_trading_bot.py can use them to score live BUY calls.
"""
import numpy as np
import pandas as pd

# Exit rules — copied from ai_trading_bot.py (kept in one dict so a test can check them)
RULES = dict(
    STOP_PCT=0.02, LOCK_BREAKEVEN_AT=0.015, LOCK_PROFIT_AT=0.04, LOCK_PROFIT_FLOOR=0.02,
    TRAIL_ACTIVATE_PCT=0.06, TRAIL_PCT=0.05, TRAIL_FLOOR_PCT=0.03,
)
COST_PER_SIDE = 0.0003      # 0.03% per buy and per sell: worse real fills than paper (~$7 per $12k round trip)
EOD_MINUTE = 15 * 60 + 45   # exit at the open of the 15:45 bar
LAST_DECISION_MINUTE = 15 * 60 + 15   # last bar whose close (15:30) can still lead to a buy before 15:45
OPEN_MINUTE = 9 * 60 + 30

META_FEATURES = [
    'p', 'p_margin',            # how sure the main model was, and by how much it cleared its cutoff
    'minute', 'dow',            # minutes since 9:30, day of week
    'rsi', 'vol_ratio', 'bb_pct', 'atr_pct', 'adx',
    'ret_5', 'ret_30', 'macd_rel',
    'stock_day_ret',            # stock's move since today's open
    'spy_day_ret',              # market's move since today's open
    'spy_ret_5', 'spy_above_ema20',
    'breadth',                  # share of the universe flashing BUY in the same bar
    'bear',                     # 1 = SPY below its 50-day average
]


def minutes_of(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 60 + index.minute).to_numpy()


def simulate_exit(o, c, mins, days, i_entry, rules=RULES):
    """Replay one trade bought at the OPEN of bar i_entry. Returns (exit_price, reason, exit_bar)."""
    entry = o[i_entry]
    peak = entry
    d0 = days[i_entry]
    n = len(c)
    j = i_entry
    while j < n:
        if days[j] != d0:                       # day ended without a 15:45 bar (half day / data gap)
            return c[j - 1], 'EOD', j - 1
        if mins[j] >= EOD_MINUTE:
            return o[j], 'EOD', j
        price = c[j]
        peak = max(peak, price)
        pct, peak_pct = price / entry - 1, peak / entry - 1
        if peak_pct >= rules['TRAIL_ACTIVATE_PCT']:
            trail = max(peak * (1 - rules['TRAIL_PCT']), entry * (1 + rules['TRAIL_FLOOR_PCT']))
            if price <= trail:
                return price, 'TRAILING_STOP', j
        elif peak_pct >= rules['LOCK_PROFIT_AT']:
            if price <= entry * (1 + rules['LOCK_PROFIT_FLOOR']):
                return price, 'PROFIT_LOCK', j
        elif peak_pct >= rules['LOCK_BREAKEVEN_AT']:
            if price <= entry:
                return price, 'PROFIT_LOCK', j
        elif pct <= -rules['STOP_PCT']:
            return price, 'STOP_LOSS', j
        j += 1
    return c[n - 1], 'END_OF_DATA', n - 1


def net_return(entry, exit_, cost=COST_PER_SIDE):
    return exit_ * (1 - cost) / (entry * (1 + cost)) - 1


def candidates_for_symbol(sym, df, thr, bear_by_day, bear_min_conf=0.68):
    """
    BUY calls the live precision-first rule would make on this symbol's bars, each replayed
    as an independent trade. df needs open/high/low/close, 'p' (main-model probability) and
    the bot's feature columns. Only regular-hours bars are used for decisions and exits.
    """
    mins_all = minutes_of(df.index)
    rth = (mins_all >= OPEN_MINUTE) & (mins_all < 16 * 60)
    d = df[rth]
    if len(d) < 3:
        return []
    o, c = d['open'].to_numpy(float), d['close'].to_numpy(float)
    mins = minutes_of(d.index)
    days = d.index.normalize().to_numpy()
    day_open = d.groupby(d.index.normalize())['open'].transform('first').to_numpy(float)
    p = d['p'].to_numpy(float)
    rsi, ema50, bb = d['rsi'].to_numpy(float), d['above_ema50'].to_numpy(float), d['bb_pct'].to_numpy(float)
    bear = np.array([bear_by_day.get(pd.Timestamp(x), 0) for x in days])
    need = np.where(bear == 1, max(thr, bear_min_conf), thr)
    ok = ((p >= need) & (rsi <= 78) & (ema50 == 1) & (bb <= 0.95)
          & (mins <= LAST_DECISION_MINUTE) & ~np.isnan(p))
    out = []
    n = len(d)
    for i in np.flatnonzero(ok):
        if i + 1 >= n or days[i + 1] != days[i]:
            continue
        ex, reason, j = simulate_exit(o, c, mins, days, i + 1)
        r = net_return(o[i + 1], ex)
        row = d.iloc[i]
        out.append(dict(
            sym=sym, t=d.index[i], day=pd.Timestamp(days[i]), entry_t=d.index[i + 1], exit_t=d.index[j],
            entry=o[i + 1], exit=ex, reason=reason, ret=r, win=int(r > 0),
            p=p[i], p_margin=p[i] - need[i], minute=mins[i] - OPEN_MINUTE, dow=d.index[i].dayofweek,
            rsi=rsi[i], vol_ratio=row['vol_ratio'], bb_pct=bb[i], atr_pct=row['atr_pct'], adx=row['adx'],
            ret_5=row['ret_5'], ret_30=row['ret_30'], macd_rel=row['macd_hist'] / (c[i] + 1e-9),
            stock_day_ret=c[i] / day_open[i] - 1, spy_ret_5=row['spy_ret_5'],
            spy_above_ema20=row['spy_above_ema20'], bear=bear[i],
        ))
    return out


def add_market_context(cands: pd.DataFrame, spy: pd.DataFrame, n_symbols: int) -> pd.DataFrame:
    """Adds spy_day_ret (SPY since today's open) and breadth (share of symbols flashing BUY this bar)."""
    if cands.empty:
        return cands
    s = spy[(minutes_of(spy.index) >= OPEN_MINUTE) & (minutes_of(spy.index) < 16 * 60)]
    s_open = s.groupby(s.index.normalize())['open'].transform('first')
    spy_day = (s['close'] / s_open - 1).rename('spy_day_ret')
    cands = cands.join(spy_day, on='t')
    cands['spy_day_ret'] = cands['spy_day_ret'].fillna(0.0)
    cands['breadth'] = cands.groupby('t')['sym'].transform('count') / max(1, n_symbols)
    return cands


def train_meta(X: pd.DataFrame, y: pd.Series):
    import xgboost as xgb
    pos = max(1, int(y.sum()))
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, scale_pos_weight=(len(y) - pos) / pos,
        eval_metric='aucpr', verbosity=0, n_jobs=-1, random_state=42,
    )
    m.fit(X[META_FEATURES].astype(float), y)
    return m


def pick_cut(meta_p: np.ndarray, ret: np.ndarray, keeps=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """
    Choose how strict the filter is, on data the filter did NOT train on.
    Tries keeping the top 30%..100% of calls by filter score and picks the share with the
    best average return per trade. keep=1.0 means "filter off" (it found nothing useful).
    Returns (cut_probability, keep_share, table).
    """
    order = np.argsort(-meta_p)
    table = []
    best = (0.0, 1.0, -np.inf)
    for k in keeps:
        n = max(1, int(round(k * len(order))))
        sel = order[:n]
        avg = float(np.mean(ret[sel]))
        cut = float(meta_p[order[n - 1]]) if k < 1.0 else 0.0
        table.append(dict(keep=k, n=n, avg_ret=avg, hit=float(np.mean(ret[sel] > 0)), cut=cut))
        if avg > best[2] + 1e-12:
            best = (cut, k, avg)
    return best[0], best[1], table
