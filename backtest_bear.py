"""
Bear-market bar sweep  —  BACKTEST ONLY. The live bots are not touched.

Independent personal project, not tied to any employer or agency.
Public market data and paper money only. No internal or personal data (no SSNs / PHI).
Backtest estimate with stated simplifications; not financial advice.

THE QUESTION
  In a bear market (yesterday's SPY close below its 50-day average) both bots skip every BUY
  unless the main AI is at least 68% sure (BEAR_MIN_CONF), and buy half size (6% vs 12%).
  Since Nov 9 2026 that rule has blocked almost every Day-Trade-100 buy.
  Would a lower bar (66%, 64%, 62%, or no extra bar) have done better over the last 12 months?

HOW (one data download, every bar tested on the same calls)
  1. Run backtest_meta.py (same walk-forward main AI, same labels and exits as the live bot)
     once with NO extra bear bar, so every BUY call the normal rule makes is saved.
  2. Every call is replayed as an independent trade, so "68% in bear" is exactly the subset of
     those calls with confidence >= 68% on bear days (and >= the normal cutoff on other days).
  3. Replay the $100k account for each bar with the live rules: max 5 open, one buy per bar
     (highest confidence), 12% per trade, 6% (or 12%) per trade on bear days, 0.03% cost per side.

  Simplifications (same as backtest_meta.py): no news / earnings / VIX vetoes, no real-trade
  relabeling, meta filter not applied (it is in shadow mode live).

Usage:  python backtest_bear.py --bot dt100            (12 months = 9 periods x 41 days)
        python backtest_bear.py --bot b1 --watchlist JPM,XOM,LLY,COST,CAT --tag diverse-a
                 (try Bot 1 on other stocks without editing watchlist.txt: they are traded AND
                  added to the training stocks, exactly like the live bot does with watchlist.txt)
        python backtest_bear.py --bot dt100 --fake     (offline self-test on random data)
"""
import argparse, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

import meta_filter as mf
import backtest_meta as bm

BARS = [('68% (live now)', 0.68), ('66%', 0.66), ('64%', 0.64), ('62%', 0.62), ('no extra bar', 0.0),
        ('no bear buys', 1.01)]   # 1.01 = no confidence can reach it -> zero buys on bear days
BEAR_CAPS = [0.06, 0.12]          # 6% = live half size, 12% = normal size


def no_extra_bear_bar():
    """Make backtest_meta save every call the normal cutoff makes, bear day or not."""
    orig = mf.candidates_for_symbol

    def patched(sym, df, thr, bear_by_day, bear_min_conf=0.68, max_hold=None):
        return orig(sym, df, thr, bear_by_day, bear_min_conf=0.0, max_hold=max_hold)
    mf.candidates_for_symbol = patched


def apply_bar(T, bar):
    """Calls the live rule would make with this bear bar. thr = the normal cutoff for that period."""
    thr = T['p'] - T['p_margin']                      # with no bear bar, need == thr exactly
    need = np.where(T['bear'] == 1, np.maximum(thr, bar), thr)
    return T[T['p'] >= need - 1e-12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', choices=['dt100', 'b1'], default='dt100')
    ap.add_argument('--folds', type=int, default=9)
    ap.add_argument('--fold-days', type=int, default=41)
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--watchlist', default=None, help='Bot 1 only: comma-separated tickers to trade instead of watchlist.txt')
    ap.add_argument('--tag', default=None, help='name used in the output file')
    a = ap.parse_args()
    if a.watchlist:
        if a.bot != 'b1':
            ap.error('--watchlist is for Bot 1 (b1) only')
        wl = [x.strip().upper() for x in a.watchlist.split(',') if x.strip()]
        bm.bot.FALLBACK_WATCHLIST = list(wl)                      # what backtest_meta trades for b1
        bm.bot.TRAINING_UNIVERSE = list(bm.bot.TRAINING_UNIVERSE) + [x for x in wl if x not in bm.bot.TRAINING_UNIVERSE]
    t0 = time.time()

    no_extra_bear_bar()
    tmp = Path(tempfile.mkdtemp()) / 'raw.json'
    sys.argv = ['backtest_meta.py', '--bot', a.bot, '--folds', str(a.folds), '--fold-days', str(a.fold_days),
                '--out', str(tmp)] + (['--fake'] if a.fake else [])
    bm.main()

    raw = json.loads(tmp.read_text())
    calls_f = tmp.with_name(tmp.stem + '_calls.csv')
    if not calls_f.exists():
        print('No BUY calls in the test period — nothing to compare.'); return
    T = pd.read_csv(calls_f)
    for col in ('t', 'entry_t', 'exit_t'):
        T[col] = pd.to_datetime(T[col], utc=True)
    T['day'] = T['t'].dt.tz_convert('America/New_York').dt.normalize()
    bear_days = sorted(T.loc[T['bear'] == 1, 'day'].dt.date.astype(str).unique())
    folds = sorted(T['fold'].unique())

    rows = []
    for name, bar in BARS:
        sub = apply_bar(T, bar)
        bear_calls = sub[sub['bear'] == 1]
        for cap in BEAR_CAPS:
            tr = bm.portfolio(sub, bear_cap=cap)
            bear_keys = set(zip(bear_calls['sym'], bear_calls['t']))
            tr_bear = [x for x in tr if (x['sym'], pd.Timestamp(x['t'])) in bear_keys]
            per_fold = [round(bm.stats(bm.portfolio(sub[sub['fold'] == f], bear_cap=cap)).get('total_pnl', 0.0), 2)
                        for f in folds]
            rows.append(dict(bar=name, bear_min_conf=bar, bear_size=cap,
                             account=bm.stats(tr), bear_day_trades=bm.stats(tr_bear),
                             bear_day_calls=bm.cand_stats(bear_calls), all_calls=bm.cand_stats(sub),
                             per_fold_total_pnl=per_fold))

    live = next(r for r in rows if r['bear_min_conf'] == 0.68 and r['bear_size'] == 0.06)
    for r in rows:
        r['periods_beating_live'] = (f"{sum(x > y for x, y in zip(r['per_fold_total_pnl'], live['per_fold_total_pnl']))}"
                                     f" of {len(folds)}")

    report = dict(
        bot=a.bot, fake_data=a.fake, watchlist=list(bm.bot.FALLBACK_WATCHLIST) if a.bot == 'b1' else 'top-100',
        traded_symbols=raw['settings']['traded_symbols'], training_symbols=raw['settings']['training_symbols'], generated=pd.Timestamp.now(tz='America/New_York').isoformat(timespec='seconds'),
        question='Lower the bear-market confidence bar (live 68%) and/or bear position size (live 6%)?',
        test_period=raw.get('test_period'), spy_buy_hold_return=raw.get('spy_buy_hold_return'),
        bear_days_in_test=len(bear_days), bear_days=bear_days,
        results=rows, runtime_sec=round(time.time() - t0),
        disclaimer='Independent personal project, not tied to any employer or agency. Public market data and '
                   'paper money only. Backtest estimate with stated simplifications; not financial advice.',
    )
    out = Path(f'state/backtest_bear_{a.bot}{"_" + a.tag if a.tag else ""}{"_fake" if a.fake else ""}.json')
    out.write_text(json.dumps(report, indent=2, default=str))

    spy = report['spy_buy_hold_return']
    print(f'\n=== BEAR-BAR SWEEP {a.bot} {report["test_period"]}'
          + (f' (SPY buy & hold {spy:+.2%})' if spy is not None else '')
          + f' | {len(bear_days)} bear days with calls'
          + (f' | stocks {",".join(report["watchlist"])}' if a.bot == 'b1' else '')
          + f' | traded {report["traded_symbols"]}, trained on {report["training_symbols"]}')
    print(f'{"bear bar":15s} {"size":>4s} | {"trades":>6s} {"hit":>6s} {"per trade":>9s} {"PF":>5s} '
          f'{"total":>7s} {"maxDD":>7s} | {"bear trades":>11s} {"bear hit":>8s} {"bear $/tr":>9s} | beats live')
    for r in rows:
        s, b = r['account'], r['bear_day_trades']
        if not s.get('trades'):
            print(f'{r["bar"]:15s} {r["bear_size"]:4.0%} | no trades'); continue
        pf = '-' if s['profit_factor'] is None else f'{s["profit_factor"]:.2f}'
        bt = (f'{b["trades"]:11d} {b["hit_rate"]:8.1%} {b["profit_per_trade"]:9.2f}' if b.get('trades')
              else f'{0:11d} {"-":>8s} {"-":>9s}')
        print(f'{r["bar"]:15s} {r["bear_size"]:4.0%} | {s["trades"]:6d} {s["hit_rate"]:6.1%} '
              f'{s["profit_per_trade"]:9.2f} {pf:>5s} {s["total_return"]:+7.2%} {s["max_drawdown"]:7.2%} | '
              f'{bt} | {r["periods_beating_live"]}')
    print(f'Saved {out} ({report["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
