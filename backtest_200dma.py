#!/usr/bin/env python3
"""
backtest_200dma.py — 200-day trend + rate-of-change reset strategy backtest

DISCLAIMER: Independent personal project, not tied to any employer or agency.
Public market data only (Yahoo Finance daily prices). No personal data (no SSNs/PHI).
Educational backtest — not financial advice.

RULES
  Entry (checked at each daily close):
    1. Close > N-day simple moving average   (N tested at 150, 200, 250)
    2. That moving average is rising (higher than 20 trading days ago)
    3. 63-day rate of change (~3 months) was above +15% within the last 60 days
       and has now reset to between -3% and +3%  -> consolidation in an uptrend
  Exit (first that happens):
    a. Close falls more than 3% below the moving average (trend broken)
    b. 63 trading days (~3 months) have passed
  Trades fill at the NEXT day's open (no look-ahead). 0.05% cost per side.

  Testing 150/200/250 is a ROBUSTNESS check, not a search for the "best" number.
  If only one length works, the idea is fragile.

HOW TO RUN
  pip install yfinance pandas numpy
  python backtest_200dma.py
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'PANW', 'CRWD', 'AMZN', 'GOOGL',
           'META', 'JPM', 'COST', 'LLY', 'PLTR', 'AMD', 'NFLX']
START = '2015-01-01'
MA_LENGTHS = [150, 200, 250]
ROC_LEN, ROC_HIGH, ROC_BAND = 63, 0.15, 0.03
LOOKBACK_HIGH, HOLD_DAYS, BREAK_PCT = 60, 63, 0.03
COST = 0.0005

_cache = {}

def prices(sym):
    if sym not in _cache:
        df = yf.download(sym, start=START, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _cache[sym] = df
    return _cache[sym]

def backtest(sym, ma_len):
    df = prices(sym)
    if len(df) < ma_len + 100:
        return []
    c, o = df['Close'], df['Open']
    sma = c.rolling(ma_len).mean()
    rising = sma > sma.shift(20)
    roc = c.pct_change(ROC_LEN)
    was_high = roc.rolling(LOOKBACK_HIGH).max() > ROC_HIGH
    reset = roc.abs() <= ROC_BAND
    signal = (c > sma) & rising & was_high & reset

    trades, i, n = [], ma_len, len(df)
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
            trades.append({'symbol': sym, 'ma': ma_len,
                           'entry_date': df.index[entry_i].date(),
                           'exit_date': df.index[exit_i].date(),
                           'return': ret, 'reason': reason})
            i = exit_i + 1
        else:
            i += 1
    return trades

def summarize(df, label):
    wins = df[df['return'] > 0]
    losses = df[df['return'] <= 0]
    print(f'\n=== {label} ===')
    print(f'Trades:        {len(df)}')
    print(f'Win rate:      {len(wins)/len(df):.1%}')
    print(f'Avg return:    {df["return"].mean():+.2%} per trade (~3 months max)')
    print(f'Avg win:       {wins["return"].mean():+.2%}')
    print(f'Avg loss:      {losses["return"].mean():+.2%}' if len(losses) else 'Avg loss: n/a')
    print(f'Worst trade:   {df["return"].min():+.2%}')
    print(f'Trend breaks:  {(df["reason"]=="TREND_BREAK").mean():.0%} of exits')

def main():
    all_trades = []
    for ma in MA_LENGTHS:
        for s in SYMBOLS:
            all_trades += backtest(s, ma)
    if not all_trades:
        print('No trades found'); return
    df = pd.DataFrame(all_trades)
    for ma in MA_LENGTHS:
        sub = df[df['ma'] == ma]
        if len(sub):
            summarize(sub, f'{ma}-DAY AVERAGE')
    print('\n200-day results by year:')
    d200 = df[df['ma'] == 200].copy()
    d200['year'] = pd.to_datetime(d200['entry_date']).dt.year
    print(d200.groupby('year')['return'].agg(['count', 'mean']).to_string())
    Path('state').mkdir(exist_ok=True)
    df.to_csv('state/backtest_results.csv', index=False)
    print('\nSaved -> state/backtest_results.csv')

if __name__ == '__main__':
    main()
