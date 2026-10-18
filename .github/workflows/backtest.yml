#!/usr/bin/env python3
"""
backtest_regime.py — REGIME-SWITCHING PORTFOLIO BACKTEST (one playbook per market type)

DISCLAIMER: Independent personal project, not tied to any employer or agency.
Public market data only (Yahoo Finance daily prices). No personal data (no SSNs/PHI).
Educational backtest — not financial advice.

IDEA
  Step 1 — Detect the market regime every day from SPY (no look-ahead):
     BULL     = SPY > 200-day avg, 200-day avg rising, 50-day avg > 200-day avg
     BEAR     = SPY < 200-day avg and 200-day avg falling
     SIDEWAYS = everything else
  Step 2 — Use the right playbook for that regime:
     BULL     -> TREND: top 10 S&P 500 stocks by 6-month momentum (skipping the
                 last month), only stocks above their own 200-day avg,
                 sized by inverse volatility (calmer stocks get more money).
                 Rebalanced monthly.
     SIDEWAYS -> MEAN REVERSION: 10 most oversold stocks (worst 3-day drop)
                 that are still above their own 200-day avg. Rebalanced weekly.
     BEAR     -> DEFENSE: hold whichever of TLT (long bonds), GLD (gold),
                 SHY (short bonds ~ cash) has the best 3-month momentum.
                 Rebalanced monthly.
     Any regime change -> rebalance immediately.

HONESTY RULES
  * Whole portfolio is simulated, not single trades: real CAGR + max drawdown.
  * Weights decided at today's close, earn returns from tomorrow (no look-ahead).
  * 0.05% cost on every dollar traded.
  * Parameters fixed in advance — NOT tuned to the results.
  * Walk-forward style report: results split into separate periods so you can
    see if it works in different eras, not just on average.
  * Known limits: uses TODAY's S&P 500 members (some survivorship bias);
    daily returns assume target weights are held (small drift ignored).

HOW TO RUN
  pip install yfinance pandas numpy
  python backtest_regime.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

START = '2014-01-01'          # one warm-up year for the 200-day averages
TEST_START = '2015-01-01'
COST = 0.0005
N_TREND, N_REVERT = 10, 10
SAFE = ['TLT', 'GLD', 'SHY']
SP500_CSV = ('https://raw.githubusercontent.com/datasets/'
             's-and-p-500-companies/main/data/constituents.csv')
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


def main():
    syms = sorted(set(pd.read_csv(SP500_CSV)['Symbol'].astype(str)
                      .str.replace('.', '-', regex=False)))
    print(f'Universe: {len(syms)} S&P 500 stocks + SPY, {", ".join(SAFE)}')
    print('Downloading prices (a few minutes)...')
    px = yf.download(syms + ['SPY'] + SAFE, start=START, auto_adjust=True,
                     progress=False, threads=True)['Close']
    px = px.dropna(how='all')
    stock_cols = [s for s in syms if s in px.columns and px[s].notna().any()]
    stocks = px[stock_cols]
    rets = px.pct_change().fillna(0)

    # ── Regime detection (SPY) ──
    spy = px['SPY']
    s200, s50 = spy.rolling(200).mean(), spy.rolling(50).mean()
    bull = (spy > s200) & (s200 > s200.shift(20)) & (s50 > s200)
    bear = (spy < s200) & (s200 < s200.shift(20))
    regime = pd.Series('SIDEWAYS', index=px.index)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'

    # ── Signals ──
    above = stocks > stocks.rolling(200).mean()
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

    print('\n=== WHOLE PERIOD (since 2015) ===')
    print(f'{"":14s}{"Total":>10s}{"Per year":>10s}{"Max drop":>10s}{"Sharpe":>8s}')
    for name, r in [('Regime bot', port), ('SPY hold', spy_r)]:
        t, c, dd, sh = stats(r)
        print(f'{name:14s}{t:>+10.0%}{c:>+10.1%}{dd:>+10.1%}{sh:>8.2f}')

    print('\n=== WALK-FORWARD STYLE: SEPARATE ERAS ===')
    for a, b, label in PERIODS:
        mask = (port.index >= a) & (port.index <= b)
        if mask.sum() < 20:
            continue
        _, c1, d1, _ = stats(port[mask])
        _, c2, d2, _ = stats(spy_r[mask])
        print(f'{label}\n   bot {c1:+.1%}/yr (max drop {d1:+.1%})   '
              f'SPY {c2:+.1%}/yr (max drop {d2:+.1%})')

    print('\n=== BY REGIME (which playbook earns its keep?) ===')
    reg_held = regime.shift(1).loc[dates]
    for reg in ['BULL', 'SIDEWAYS', 'BEAR']:
        m = reg_held == reg
        if m.sum() == 0:
            continue
        print(f'{reg:9s} {m.mean():5.0%} of days | bot {port[m].mean()*252:+.1%}/yr '
              f'vs SPY {spy_r[m].mean()*252:+.1%}/yr while in this regime')

    print('\n=== YEAR BY YEAR ===')
    yr = pd.DataFrame({'bot': (1 + port).groupby(port.index.year).prod() - 1,
                       'SPY': (1 + spy_r).groupby(spy_r.index.year).prod() - 1})
    yr['beat_SPY'] = yr['bot'] > yr['SPY']
    print(yr.round(3).to_string())

    print(f'\nCurrent regime (latest close): {regime.iloc[-1]}')
    Path('state').mkdir(exist_ok=True)
    pd.DataFrame({'bot': port, 'spy': spy_r, 'regime': reg_held}).to_csv(
        'state/regime_backtest_daily.csv')
    print('Saved -> state/regime_backtest_daily.csv')


if __name__ == '__main__':
    main()
