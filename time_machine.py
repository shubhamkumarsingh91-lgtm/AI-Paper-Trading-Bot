#!/usr/bin/env python3
"""
TIME MACHINE: "if I buy now, how does the rest of today usually play out?"

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Public market data (Alpaca IEX bars) and paper money only. No internal, personal or
protected data (no SSNs/PHI). Research tool, not financial advice.

Inspired by the "Time Machine" idea in the open-source day-trader-skill pack, written
from scratch for this bot. Nothing here places orders; the live bots do not import it.

WHAT IT DOES, FOR ONE BUY CALL (stock S, decided at the close of 15-min bar s today)
  1. POOL: the last LOOKBACK complete trading days of S before today (no future data).
     Each day is stored as 26 regular-hours 15-minute bar returns.
  2. VOLATILITY ADJUST: today's bar-to-bar volatility so far, divided by the pool's
     volatility over the same bars, clipped to 0.5x-2x. A wild morning makes the
     simulated afternoon wilder; a quiet one makes it calmer.
  3. SIMULATE the rest of the day N_PATHS times, two different ways (the A/B test):
       'slot' = every future bar is drawn from a RANDOM past day, independently.
                Keeps each time-of-day's typical move, loses any trend inside a day.
       'day'  = every path replays ONE random past day's remaining bars in order.
                Keeps intraday momentum and reversals, but has fewer distinct paths.
  4. REPLAY each path with the bot's REAL exit rules (meta_filter.RULES: -2% stop,
     breakeven lock at +1.5%, +2% floor at +4%, 5% trailing stop from +6%, close at
     15:45), checked on bar closes, after trading costs.
  5. FEATURES (5 per method; prefix tms_ = slot, tmd_ = day):
       p_win     share of paths that made money after costs
       p_stop    share of paths that hit the -2% stop loss
       p_eod     share of paths still open at the 15:45 close
       exp_ret   average net return across paths
       med_bars  median number of 15-min bars held

KNOWN SIMPLIFICATIONS
  - entry price = the decision bar's close (the backtest buys at the next bar's open);
    the first simulated bar uses a past day's open->close move for that slot
  - the 15:45 exit uses the 15:30 bar's close (the live rule uses the 15:45 bar's open)
  - only complete 26-bar days go into the pool (half days and gappy IEX days are skipped)
"""
import zlib

import numpy as np
import pandas as pd

import meta_filter as mf

SLOTS = 26                    # 9:30 ... 15:45 regular-hours 15-min bars
EOD_SLOT = 25                 # the 15:45 bar: exit happens at its open (we use slot 24's close)
LOOKBACK = 40                 # past trading days in the pool
MIN_POOL = 15                 # fewer complete days than this -> features are NaN (unknown)
N_PATHS = 2000
VOL_CLIP = (0.5, 2.0)
METHODS = ('slot', 'day')
PREFIX = {'slot': 'tms_', 'day': 'tmd_'}
FEATS = ['p_win', 'p_stop', 'p_eod', 'exp_ret', 'med_bars']
REASONS = ('STOP_LOSS', 'PROFIT_LOCK', 'TRAILING_STOP', 'EOD')


def feature_names(method):
    return [PREFIX[method] + f for f in FEATS]


# ── 1. day table ─────────────────────────────────────────────────────────────

class DayTable:
    """Complete regular-hours days of one symbol as log-return arrays.
    oc[d, k] = log(close/open) of slot k   (used for the first bar after entry)
    cc[d, k] = log(close_k / close_{k-1})   (used for every later bar; cc[d, 0] = oc[d, 0])"""

    def __init__(self, bars: pd.DataFrame):
        mins = mf.minutes_of(bars.index)
        rth = bars[(mins >= mf.OPEN_MINUTE) & (mins < 16 * 60)]
        slot = (mf.minutes_of(rth.index) - mf.OPEN_MINUTE) // 15
        rth = rth.assign(_slot=slot, _day=rth.index.normalize())
        days, oc, cc = [], [], []
        for day, g in rth.groupby('_day', sort=True):
            if len(g) != SLOTS or not np.array_equal(g['_slot'].to_numpy(), np.arange(SLOTS)):
                continue                                             # half day or missing bars
            o, c = g['open'].to_numpy(float), g['close'].to_numpy(float)
            if (o <= 0).any() or (c <= 0).any():
                continue
            days.append(pd.Timestamp(day))
            oc.append(np.log(c / o))
            cc.append(np.r_[np.log(c[0] / o[0]), np.log(c[1:] / c[:-1])])
        self.days = pd.DatetimeIndex(days)
        self.oc = np.array(oc).reshape(-1, SLOTS)
        self.cc = np.array(cc).reshape(-1, SLOTS)

    def pool_before(self, day, lookback=LOOKBACK):
        """Row numbers of the last `lookback` complete days strictly before `day`."""
        end = int(self.days.searchsorted(pd.Timestamp(day), side='left'))
        return np.arange(max(0, end - lookback), end)


# ── 2. volatility adjustment ─────────────────────────────────────────────────

def vol_factor(today_cc, pool_cc, clip=VOL_CLIP):
    """today_cc = today's close-to-close log returns for slots 1..s (known at decision time).
    pool_cc = the pool's returns for the same slots (days x slots). Needs >= 3 bars today."""
    today_cc = np.asarray(today_cc, float)
    if today_cc.size < 3 or pool_cc.size == 0:
        return 1.0
    hist = float(np.std(pool_cc))
    if hist <= 0:
        return 1.0
    return float(np.clip(np.std(today_cc) / hist, *clip))


# ── 3. path simulation ───────────────────────────────────────────────────────

def simulate_paths(oc, cc, entry_slot, method, n_paths, rng, factor=1.0):
    """Prices relative to entry (=1.0) at the close of slots entry_slot..24.
    oc, cc = pool arrays (days x 26). Returns array (n_paths, 25 - entry_slot)."""
    n_days = oc.shape[0]
    steps = EOD_SLOT - entry_slot
    cols = np.arange(entry_slot, EOD_SLOT)
    if method == 'slot':
        pick = rng.integers(0, n_days, size=(n_paths, steps))         # a different past day per bar
    elif method == 'day':
        pick = np.repeat(rng.integers(0, n_days, size=(n_paths, 1)), steps, axis=1)   # one past day per path
    else:
        raise ValueError(method)
    logr = cc[pick, cols]
    logr[:, 0] = oc[pick[:, 0], entry_slot]                          # bought at this bar's open
    return np.exp(np.cumsum(logr * factor, axis=1))


# ── 4. the bot's exit rules, on many paths at once ───────────────────────────

def exits_vectorized(paths, rules=mf.RULES):
    """Same rules and order as meta_filter.simulate_exit, checked on each bar close.
    Returns (exit_price_rel, reason_code, bars_held); reason_code indexes REASONS."""
    n, steps = paths.shape
    alive = np.ones(n, bool)
    peak = np.ones(n)
    px = np.full(n, np.nan)
    why = np.full(n, REASONS.index('EOD'))
    held = np.full(n, steps)
    for j in range(steps):
        price = paths[:, j]
        peak = np.maximum(peak, price)
        pk = peak - 1.0
        trail_on = pk >= rules['TRAIL_ACTIVATE_PCT']
        lock4_on = ~trail_on & (pk >= rules['LOCK_PROFIT_AT'])
        lock15_on = ~trail_on & ~lock4_on & (pk >= rules['LOCK_BREAKEVEN_AT'])
        stop_on = ~trail_on & ~lock4_on & ~lock15_on
        trail = np.maximum(peak * (1 - rules['TRAIL_PCT']), 1 + rules['TRAIL_FLOOR_PCT'])
        hit_trail = alive & trail_on & (price <= trail)
        hit_lock = alive & ((lock4_on & (price <= 1 + rules['LOCK_PROFIT_FLOOR'])) | (lock15_on & (price <= 1.0)))
        hit_stop = alive & stop_on & (price - 1.0 <= -rules['STOP_PCT'])
        for mask, name in ((hit_trail, 'TRAILING_STOP'), (hit_lock, 'PROFIT_LOCK'), (hit_stop, 'STOP_LOSS')):
            px[mask], why[mask], held[mask] = price[mask], REASONS.index(name), j + 1
        alive &= ~(hit_trail | hit_lock | hit_stop)
    px[alive] = paths[alive, -1]                                     # held to the 15:45 close
    return px, why, held


def summarize(px, why, held, cost=mf.COST_PER_SIDE):
    net = px * (1 - cost) / (1 + cost) - 1
    return dict(p_win=float(np.mean(net > 0)),
                p_stop=float(np.mean(why == REASONS.index('STOP_LOSS'))),
                p_eod=float(np.mean(why == REASONS.index('EOD'))),
                exp_ret=float(np.mean(net)),
                med_bars=float(np.median(held)))


# ── 5. features for one call, and for a table of calls ───────────────────────

def seed_for(sym, t, method):
    return zlib.crc32(f'{sym}|{pd.Timestamp(t).isoformat()}|{method}'.encode())   # same answer every run


def features_for_call(table: DayTable, sym, t, today_bars, n_paths=N_PATHS, lookback=LOOKBACK):
    """today_bars = this symbol's regular-hours bars of today up to and including the decision bar."""
    out = {n: np.nan for m in METHODS for n in feature_names(m)}
    s = int((t.hour * 60 + t.minute - mf.OPEN_MINUTE) // 15)
    entry = s + 1
    if s < 0 or entry >= EOD_SLOT:
        return out
    pool = table.pool_before(t.normalize(), lookback)
    if len(pool) < MIN_POOL:
        return out
    oc, cc = table.oc[pool], table.cc[pool]
    c = today_bars['close'].to_numpy(float)
    today_cc = np.log(c[1:s + 1] / c[:s]) if len(c) >= s + 1 and s >= 1 else np.array([])
    f = vol_factor(today_cc, cc[:, 1:s + 1])
    for m in METHODS:
        rng = np.random.default_rng(seed_for(sym, t, m))
        res = summarize(*exits_vectorized(simulate_paths(oc, cc, entry, m, n_paths, rng, f)))
        out.update({PREFIX[m] + k: v for k, v in res.items()})
    out['tm_vol_factor'] = f
    out['tm_pool_days'] = int(len(pool))
    return out


def add_features(cands: pd.DataFrame, raw_by_sym: dict, n_paths=N_PATHS, lookback=LOOKBACK, cache=None):
    """Adds both methods' features to every candidate BUY call (columns sym, t).
    cache = dict reused across folds: features depend only on (sym, t), not on the fold."""
    if cands.empty:
        return cands
    cache = {} if cache is None else cache
    tables = cache.setdefault('__tables__', {})          # one DayTable per symbol, built once
    rows = []
    for sym, t in zip(cands['sym'], cands['t']):
        key = (sym, pd.Timestamp(t))
        if key not in cache:
            if sym not in tables:
                tables[sym] = DayTable(raw_by_sym[sym])
            raw = raw_by_sym[sym]
            day = raw[(raw.index.normalize() == key[1].normalize()) & (raw.index <= key[1])]
            day = day[mf.minutes_of(day.index) >= mf.OPEN_MINUTE]
            cache[key] = features_for_call(tables[sym], sym, key[1], day, n_paths, lookback)
        rows.append(cache[key])
    return pd.concat([cands.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
