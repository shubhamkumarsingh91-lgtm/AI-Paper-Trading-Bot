#!/usr/bin/env python3
"""
backtest_regime.py — REGIME-SWITCHING PORTFOLIO BACKTEST
VERSION 2: uses the S&P 500's REAL historical membership (point-in-time)

DISCLAIMER: Independent personal project, not tied to any employer or agency.
Public market data only (Yahoo Finance prices; public S&P 500 membership history
from github.com/fja05680/sp500). No personal data (no SSNs/PHI).
Educational backtest — not financial advice.

WHAT CHANGED vs version 1 (only ONE thing, so we can see its effect)
  * On each day the bot may only pick stocks that were ACTUALLY in the
    S&P 500 on that day — not today's members. This removes most of the
    survivorship bias that inflated version 1 (+31%/yr).
  * Playbooks are unchanged (bear fix comes later, one change at a time).

REMAINING HONEST LIMITS (printed at the end too)
  * Yahoo has no prices for many delisted/bankrupt companies. Those stocks
    can't be picked at all, so some bias remains. Coverage % is reported.
  * If a held stock vanishes from the data, its missing days count as 0%
    return — that can hide a real crash. Treat results as an UPPER bound.

PLAYBOOKS
  BULL     -> top 10 by 6-month momentum (skip last month), above own 200-day avg,
              inverse-volatility sized, monthly rebalance
  SIDEWAYS -> 10 most oversold (worst 3-day drop) stocks above own 200-day avg,
              weekly rebalance
  BEAR     -> best of TLT / GLD / SHY by 3-month momentum, monthly rebalance
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

START = '2014-01-01'
TEST_START = '2015-01-01'
COST = 0.0005
N_TREND, N_REVERT = 10, 10
SAFE = ['TLT', 'GLD', 'SHY']
HIST_CSV = ('https://raw.githubusercontent.com/fja05680/sp500/master/'
            'S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv')
PERIODS = [('2015-01-01', '2019-12-31', '2015-2019 (calm bull + 2018 dip)'),
           ('2020-01-01', '2022-12-31', '2020-2022 (COVID crash + 2022 bear)'),
           ('2023-01-01', '2100-01-01', '2023-now (AI rally + 2026 shocks)')]


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return eq.iloc[-1] - 1, cagr, dd, sharpe


def load_membership():
    h = pd.read_csv(HIST_CSV)
    date_col, tick_col = h.columns[0], h.columns[1]
    h[date_col] = pd.to_datetime(h[date_col])
    h = h[h[date_col] >= pd.Timestamp(START) - pd.Timedelta(days=400)]
    snaps = {}
    for _, row in h.sort_values(date_col).iterrows():
        ticks = {t.strip().replace('.', '-') for t in str(row[tick_col]).split(',') if t.strip()}
        snaps[row[date_col]] = ticks
    snap_dates = pd.DatetimeIndex(sorted(snaps))
    print(f'Membership history: {len(snap_dates)} snapshots, '
          f'{snap_dates[0].date()} -> {snap_dates[-1].date()}')
    return snaps, snap_dates


def main():
    snaps, snap_dates = load_membership()
    ever = sorted(set().union(*snaps.values()))
    print(f'Stocks that were EVER in the S&P 500 in this period: {len(ever)}')
    print('Downloading prices (several minutes)...')
    px = yf.download(ever + ['SPY'] + SAFE, start=START, auto_adjust=True,
                     progress=False, threads=True)['Close']
    px = px.dropna(how='all')
    stock_cols = [s for s in ever if s in px.columns and px[s].notna().sum() > 50]
    print(f'Stocks with usable Yahoo data: {len(stock_cols)} of {len(ever)} '
          f'({len(stock_cols)/len(ever):.0%} coverage)')
    stocks = px[stock_cols]
    rets = px.pct_change().fillna(0)

    # Point-in-time membership mask: True if stock was in the index that day
    member = pd.DataFrame(False, index=px.index, columns=stock_cols)
    col_set = set(stock_cols)
    for d in px.index:
        k = snap_dates.searchsorted(d, side='right') - 1
        if k >= 0:
            m = [t for t in snaps[snap_dates[k]] if t in col_set]
            member.loc[d, m] = True

    # Regime detection (SPY)
    spy = px['SPY']
    s200, s50 = spy.rolling(200).mean(), spy.rolling(50).mean()
    bull = (spy > s200) & (s200 > s200.shift(20)) & (s50 > s200)
    bear = (spy < s200) & (s200 < s200.shift(20))
    regime = pd.Series('SIDEWAYS', index=px.index)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'

    above = (stocks > stocks.rolling(200).mean()) & member
    mom = stocks.shift(21) / stocks.shift(126) - 1
    vol = rets[stock_cols].rolling(63).std()
    r3 = stocks / stocks.shift(3) - 1
    safe_mom = px[SAFE] / px[SAFE].shift(63) - 1

    dates = px.index[px.index >= TEST_START]
    weights, w = {}, pd.Series({'SHY': 1.0})
    prev_reg, prev_month, days_since = None, None, 99
    for d in dates:
        reg = regime[d]
        new_month = d.month != prev_month
        prev_month = d.month
        days_since += 1
        rebal = (reg != prev_reg or (reg in ('BULL', 'BEAR') and new_month)
                 or (reg == 'SIDEWAYS' and days_since >= 5))
        if rebal:
            days_since = 0
            if reg == 'BULL':
                m = mom.loc[d][above.loc[d]].dropna().nlargest(N_TREND)
                iv = (1 / vol.loc[d, m.index].replace(0, np.nan)).dropna()
                w = iv / iv.sum() if len(iv) else pd.Series({'SHY': 1.0})
            elif reg == 'SIDEWAYS':
                c = r3.loc[d][above.loc[d]].dropna().nsmallest(N_REVERT)
                w = (pd.Series(1 / len(c), index=c.index) if len(c)
                     else pd.Series({'SHY': 1.0}))
            else:
                sm = safe_mom.loc[d].dropna()
                w = pd.Series({(sm.idxmax() if len(sm) else 'SHY'): 1.0})
        weights[d] = w
        prev_reg = reg

    W = pd.DataFrame(weights).T.fillna(0)
    held = W.shift(1).fillna(0)
    R = rets.loc[dates, W.columns]
    gross = (held * R).sum(axis=1)
    turnover = (W - W.shift(1).fillna(0)).abs().sum(axis=1).shift(1).fillna(0)
    port = gross - turnover * COST
    spy_r = rets.loc[dates, 'SPY']

    print('\n=== WHOLE PERIOD (since 2015) — POINT-IN-TIME MEMBERS ===')
    print(f'{"":14s}{"Total":>10s}{"Per year":>10s}{"Max drop":>10s}{"Sharpe":>8s}')
    for name, r in [('Regime bot', port), ('SPY hold', spy_r)]:
        t, c, dd, sh = stats(r)
        print(f'{name:14s}{t:>+10.0%}{c:>+10.1%}{dd:>+10.1%}{sh:>8.2f}')
    print('(Version 1 with today\'s members showed +31.0%/yr — compare.)')

    print('\n=== SEPARATE ERAS ===')
    for a, b, label in PERIODS:
        mask = (port.index >= a) & (port.index <= b)
        if mask.sum() < 20:
            continue
        _, c1, d1, _ = stats(port[mask])
        _, c2, d2, _ = stats(spy_r[mask])
        print(f'{label}\n   bot {c1:+.1%}/yr (max drop {d1:+.1%})   '
              f'SPY {c2:+.1%}/yr (max drop {d2:+.1%})')

    print('\n=== BY REGIME ===')
    reg_held = regime.shift(1).loc[dates]
    for reg in ['BULL', 'SIDEWAYS', 'BEAR']:
        m = reg_held == reg
        if m.sum() == 0:
            continue
        print(f'{reg:9s} {m.mean():5.0%} of days | bot {port[m].mean()*252:+.1%}/yr '
              f'vs SPY {spy_r[m].mean()*252:+.1%}/yr')

    print('\n=== YEAR BY YEAR ===')
    yr = pd.DataFrame({'bot': (1 + port).groupby(port.index.year).prod() - 1,
                       'SPY': (1 + spy_r).groupby(spy_r.index.year).prod() - 1})
    yr['beat_SPY'] = yr['bot'] > yr['SPY']
    print(yr.round(3).to_string())

    print(f'\nCurrent regime (latest close): {regime.iloc[-1]}')
    print('NOTE: delisted stocks missing from Yahoo -> results are an UPPER bound.')
    Path('state').mkdir(exist_ok=True)
    pd.DataFrame({'bot': port, 'spy': spy_r, 'regime': reg_held}).to_csv(
        'state/regime_backtest_daily.csv')
    print('Saved -> state/regime_backtest_daily.csv')


if __name__ == '__main__':
    main()
