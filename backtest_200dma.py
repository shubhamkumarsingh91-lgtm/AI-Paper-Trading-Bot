#!/usr/bin/env python3
"""
backtest_200dma.py — S&P 500 VERSION (fairer test + buy-and-hold comparison)

DISCLAIMER: Independent personal project, not tied to any employer or agency.
Public market data only (Yahoo Finance daily prices). No personal data (no SSNs/PHI).
Educational backtest — not financial advice.

WHAT CHANGED vs the first version
  * Universe = ALL current S&P 500 stocks (~500), not 15 hand-picked winners.
    Note: still uses TODAY's members, so some survivorship bias remains
    (companies that crashed out of the index are missing) — but far less.
  * Every trade is compared against SPY over the exact same dates
    -> "excess return" = did the strategy beat just holding the index?
  * Also reports SPY buy-and-hold for the whole period.

RULES (same as before, 200-day only by default)
  Entry: close > 200-day SMA, SMA rising vs 20 days ago,
         63-day ROC was > +15% in last 60 days and is now within +/-3%
  Exit:  close > 3% below SMA (trend break) or 63 trading days
  Fills at next day's open. 0.05% cost per side.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

START = '2015-01-01'
MA_LEN = 200
ROC_LEN, ROC_HIGH, ROC_BAND = 63, 0.15, 0.03
LOOKBACK_HIGH, HOLD_DAYS, BREAK_PCT = 60, 63, 0.03
COST = 0.0005
SP500_CSV = ('https://raw.githubusercontent.com/datasets/'
             's-and-p-500-companies/main/data/constituents.csv')


def get_universe():
    try:
        syms = pd.read_csv(SP500_CSV)['Symbol'].astype(str).str.replace('.', '-', regex=False)
        print(f'Universe: {len(syms)} S&P 500 stocks')
        return sorted(set(syms))
    except Exception as e:
        print(f'Could not load S&P 500 list ({e}) — aborting')
        raise SystemExit(1)


def clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def backtest(sym, df, spy):
    if len(df) < MA_LEN + 100:
        return []
    c, o = df['Close'], df['Open']
    sma = c.rolling(MA_LEN).mean()
    rising = sma > sma.shift(20)
    roc = c.pct_change(ROC_LEN)
    was_high = roc.rolling(LOOKBACK_HIGH).max() > ROC_HIGH
    reset = roc.abs() <= ROC_BAND
    signal = (c > sma) & rising & was_high & reset

    trades, i, n = [], MA_LEN, len(df)
    while i < n - 1:
        if signal.iloc[i]:
            entry_i = i + 1
            entry = o.iloc[entry_i]
            exit_i, reason = min(entry_i + HOLD_DAYS, n - 1), 'TIME'
            for j in range(entry_i, min(entry_i + HOLD_DAYS, n - 1)):
                if c.iloc[j] < sma.iloc[j] * (1 - BREAK_PCT):
                    exit_i, reason = j + 1, 'TREND_BREAK'
                    break
            exit_px = o.iloc[exit_i] if reason == 'TREND_BREAK' else c.iloc[exit_i]
            ret = exit_px / entry * (1 - COST) ** 2 - 1
            d_in, d_out = df.index[entry_i], df.index[exit_i]
            try:
                spy_ret = spy['Close'].asof(d_out) / spy['Open'].asof(d_in) - 1
            except Exception:
                spy_ret = float('nan')
            trades.append({'symbol': sym, 'entry_date': d_in.date(),
                           'exit_date': d_out.date(), 'return': ret,
                           'spy_return': spy_ret, 'excess': ret - spy_ret,
                           'reason': reason})
            i = exit_i + 1
        else:
            i += 1
    return trades


def main():
    syms = get_universe()
    spy = clean(yf.download('SPY', start=START, auto_adjust=True, progress=False))
    print('Downloading prices (this takes a minute or two)...')
    data = yf.download(syms, start=START, auto_adjust=True, progress=False,
                       group_by='ticker', threads=True)

    all_trades, used = [], 0
    for s in syms:
        try:
            df = data[s].dropna()
        except Exception:
            continue
        t = backtest(s, df, spy)
        if len(df):
            used += 1
        all_trades += t
    print(f'Stocks with data: {used}')

    if not all_trades:
        print('No trades found'); return
    df = pd.DataFrame(all_trades).dropna(subset=['spy_return'])
    wins = df[df['return'] > 0]
    losses = df[df['return'] <= 0]
    beat = (df['excess'] > 0).mean()

    print('\n=== 200-DAY STRATEGY ON THE S&P 500 ===')
    print(f'Trades:              {len(df)}')
    print(f'Win rate:            {len(wins)/len(df):.1%}')
    print(f'Avg return/trade:    {df["return"].mean():+.2%}')
    print(f'Median return/trade: {df["return"].median():+.2%}')
    print(f'Avg win:             {wins["return"].mean():+.2%}')
    print(f'Avg loss:            {losses["return"].mean():+.2%}')
    print(f'Worst trade:         {df["return"].min():+.2%}')
    print(f'Trend-break exits:   {(df["reason"]=="TREND_BREAK").mean():.0%}')

    print('\n=== VS JUST HOLDING SPY OVER THE SAME DATES ===')
    print(f'SPY avg over same windows: {df["spy_return"].mean():+.2%}')
    print(f'Avg EXCESS per trade:      {df["excess"].mean():+.2%}  <-- the real edge')
    print(f'Median excess per trade:   {df["excess"].median():+.2%}')
    print(f'Trades that beat SPY:      {beat:.1%}')

    spy_total = spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_cagr = (1 + spy_total) ** (1 / years) - 1
    print(f'\nSPY buy-and-hold since {START}: {spy_total:+.0%} total, {spy_cagr:+.1%} per year')

    print('\nBy year (avg return | avg excess vs SPY | trades):')
    df['year'] = pd.to_datetime(df['entry_date']).dt.year
    print(df.groupby('year').agg(trades=('return', 'count'),
                                 avg_return=('return', 'mean'),
                                 avg_excess=('excess', 'mean')).round(4).to_string())

    Path('state').mkdir(exist_ok=True)
    df.to_csv('state/backtest_results.csv', index=False)
    print('\nSaved -> state/backtest_results.csv')


if __name__ == '__main__':
    main()
