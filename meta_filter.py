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


def simulate_exit(o, c, mins, days, i_entry, rules=RULES, max_hold=None):
    """Replay one trade bought at the OPEN of bar i_entry. Returns (exit_price, reason, exit_bar).
    max_hold = optional time exit: sell at the open of the bar max_hold bars after entry
    (8 bars = 2 hours), if no other rule has sold it first."""
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
        if max_hold and j - i_entry >= max_hold:
            return o[j], 'TIME_EXIT', j
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


def candidates_for_symbol(sym, df, thr, bear_by_day, bear_min_conf=0.68, max_hold=None):
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
        ex, reason, j = simulate_exit(o, c, mins, days, i + 1, max_hold=max_hold)
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


# ── shared by the backtest and the live bot ──────────────────────────────────

def bear_days(spy: pd.DataFrame) -> dict:
    """bear[day] = 1 if the PREVIOUS day's SPY close was below the 50-day average of closes up to then."""
    daily = spy['close'].resample('1D').last().dropna()
    sma = daily.rolling(50).mean()
    flag = (daily < sma).astype(int).shift(1).fillna(0).astype(int)
    return {pd.Timestamp(d).normalize(): int(v) for d, v in flag.items()}


def day_return(df: pd.DataFrame) -> float:
    """Move since today's 9:30 open, using the latest bar's close (live bot)."""
    last_day = df.index[-1].normalize()
    today = df[(df.index.normalize() == last_day) & (minutes_of(df.index) >= OPEN_MINUTE)]
    return float(df['close'].iloc[-1] / today['open'].iloc[0] - 1) if len(today) else 0.0


def train_bundle(cands: pd.DataFrame, min_calls: int = 60):
    """
    Train the filter on replayed BUY calls: fit on the older 2/3, choose strictness on the
    newest 1/3 (unseen). Returns a dict to save with joblib, or None if there are too few calls.
    """
    if cands is None or len(cands) < min_calls:
        return None
    c = cands.sort_values('t')
    cut_at = c['t'].iloc[int(len(c) * 2 / 3)]
    fit, pick = c[c['t'] < cut_at], c[c['t'] >= cut_at]
    if fit['win'].nunique() < 2 or len(pick) < 10:
        return None
    m = train_meta(fit, fit['win'])
    mp = m.predict_proba(pick[META_FEATURES].astype(float))[:, 1]
    cut, keep, table = pick_cut(mp, pick['ret'].to_numpy())
    return dict(model=m, cut=cut, keep=keep, table=table, calls=int(len(c)),
                hit_all=float(c['win'].mean()), avg_ret_all=float(c['ret'].mean()),
                period=[str(c['t'].iloc[0]), str(c['t'].iloc[-1])],
                importance=dict(sorted(zip(META_FEATURES, map(float, m.feature_importances_)), key=lambda x: -x[1])[:6]))


def score(bundle, feats: dict):
    """Filter probability that this BUY call makes money, and whether it clears the chosen cut."""
    x = pd.DataFrame([{k: float(feats.get(k, 0.0)) for k in META_FEATURES}])
    p = float(bundle['model'].predict_proba(x)[:, 1][0])
    return p, p >= bundle['cut']


# ── trade-matched labels (López de Prado ch. 3, "triple-barrier" idea) ─────────
# The classic label asks "+0.5% within 2 hours without a -1% dip?", but the bot
# holds most trades to 15:45. This label asks the question the bot actually
# trades: "if I bought at the next bar's open and used my real exits (stop,
# profit locks, trailing stop, 15:45 close), did I make money after costs?"

def trade_labels(df: pd.DataFrame, max_hold=None) -> pd.Series:
    """Net return of a trade decided at each regular-hours bar (NaN where the bot can't buy)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    mins_all = minutes_of(df.index)
    rth = (mins_all >= OPEN_MINUTE) & (mins_all < 16 * 60)
    d = df[rth]
    if len(d) < 3:
        return out
    o, c = d['open'].to_numpy(float), d['close'].to_numpy(float)
    mins = minutes_of(d.index)
    days = d.index.normalize().to_numpy()
    n = len(d)
    ret = np.full(n, np.nan)
    for i in range(n - 1):
        if mins[i] > LAST_DECISION_MINUTE or days[i + 1] != days[i]:
            continue
        ex, _, _ = simulate_exit(o, c, mins, days, i + 1, max_hold=max_hold)
        ret[i] = net_return(o[i + 1], ex)
    out.loc[d.index] = ret
    return out


def calibrate_expectancy(probs, rets, ok_mask, min_signals: int):
    """
    Pick the cutoff whose BUY calls (that also pass the safety vetoes) made the most money
    per call on validation data, instead of the best precision. Needs >= min_signals calls
    so a handful of lucky calls can't win. Returns (cutoff, avg_ret, n_calls, hit_rate).
    """
    probs, rets, ok_mask = np.asarray(probs), np.asarray(rets), np.asarray(ok_mask, bool)
    best = (0.5, -np.inf, 0, 0.0)
    for th in np.round(np.arange(0.50, 0.91, 0.02), 2):
        sel = ok_mask & (probs >= th) & ~np.isnan(rets)
        n = int(sel.sum())
        if n < min_signals:
            continue
        avg = float(rets[sel].mean())
        if avg > best[1]:
            best = (round(float(th), 2), avg, n, float((rets[sel] > 0).mean()))
    return best


def calibrate_rate(probs, minutes, dates, ok_mask, per_day: float):
    """
    Steady cutoff: the lowest cutoff (0.50-0.95, 1-pt steps) at which the BUY calls that pass the
    vetoes and come before the 15:15 decision bar average no more than 2 x per_day calls a day on
    validation data. The account then takes at most per_day buys a day. Returns (cutoff, calls_per_day).
    """
    probs, minutes, ok_mask = np.asarray(probs), np.asarray(minutes), np.asarray(ok_mask, bool)
    elig = ok_mask & (minutes >= OPEN_MINUTE) & (minutes <= LAST_DECISION_MINUTE)
    n_days = max(1, len(set(np.asarray(dates)[elig])))
    for th in np.round(np.arange(0.50, 0.951, 0.01), 2):       # round: arange gives 0.6000000000000001
        per = float((elig & (probs >= th)).sum()) / n_days
        if per <= 2 * per_day:
            return round(float(th), 2), per
    return 0.95, float((elig & (probs >= 0.95)).sum()) / n_days
