"""
Middle-ground exit test for Bot 1  —  BACKTEST ONLY. The live bots are not touched.

Independent personal project, not tied to any employer or agency.
Public market data and paper money only. No internal or personal data (no SSNs / PHI).
Backtest estimate with stated simplifications; not financial advice.

THE PROBLEM (from the 12-month backtests)
  Bot 1 mostly buys after 2 PM, holds ~1 hour and is force-sold at 3:45 PM. Those trades are
  coin flips that pay costs. Holding everything overnight is no answer either: a stock can gap
  down at the open, so the -2% stop can't protect you.

THE MIDDLE GROUND BEING TESTED (combined in a grid against the live rule)
  When to buy   all day 9:45 AM - 3:30 PM (live)   vs   mornings only 9:45 - 11:30 AM
  How to sell   sell everything by 3:45 PM (live)
                vs "carry winners": at 3:45 PM a trade that is up >= +1% (or +0.5%) is kept
                   overnight with a breakeven stop; anything else is sold. Next morning, if the
                   stock opens below the buy price it is sold at that open (the real gap is
                   counted, it can be worse than -2%). Max 3 trading days, then sold at 3:45 PM.
  Losers never stay overnight; the -2% stop and the live profit locks are unchanged.

HOW
  One run of backtest_meta.py (same walk-forward AI, labels and bear rule as live) with every
  candidate replayed under each grid combination, then the $100k account per combination.
  Simplification: the AI's training label is the same for all combinations (the exit rule only
  changes how a BUY is replayed), no news / earnings / VIX vetoes, meta filter in shadow (off).

Usage:  python backtest_exits.py --bot b1 --label trade --watchlist AMD,GS,FCX,TSLA,DAL --tag trade
        python backtest_exits.py --bot b1 --fake --folds 2 --fold-days 30     (offline self-test)
"""
import argparse, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

import meta_filter as mf
import backtest_meta as bm

WINDOWS = {'all day': 15 * 60 + 15,          # last decision bar 15:15 -> buy by 15:30 (live)
           'mornings': 11 * 60 + 15}         # last decision bar 11:15 -> buy by 11:30
EXITS = {'sell by close': None,              # live
         'carry >= +1%': 0.01,
         'carry >= +0.5%': 0.005}
LIVE = ('all day', 'sell by close')
MAX_DAYS = 3


def make_exit(carry_min=None, max_days=MAX_DAYS, carry_floor=0.0):
    """Same as meta_filter.simulate_exit, plus: at 15:45 keep a trade that is up >= carry_min
    overnight with a stop at entry*(1+carry_floor). carry_min=None = identical to the live exit."""
    R = mf.RULES

    def sim(o, c, mins, days, i_entry, rules=R, max_hold=None):
        entry = o[i_entry]
        peak = entry
        cur, day_ix, carried = days[i_entry], 0, False
        n, j = len(c), i_entry
        while j < n:
            if days[j] != cur:
                if not carried:                          # day ended without a 15:45 bar
                    return c[j - 1], 'EOD', j - 1
                cur, day_ix = days[j], day_ix + 1
                if o[j] <= entry * (1 + carry_floor):    # opened below the overnight stop
                    return o[j], 'GAP_STOP', j
            if mins[j] >= mf.EOD_MINUTE:
                if (carry_min is not None and day_ix < max_days - 1
                        and o[j] / entry - 1 >= carry_min):
                    carried = True
                    k = j
                    while k < n and days[k] == cur:
                        k += 1
                    if k >= n:
                        return o[j], 'EOD', j
                    j = k
                    continue
                return o[j], ('EOD' if day_ix == 0 else 'CARRY_EOD'), j
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
            elif carried and pct <= carry_floor:
                return price, 'CARRY_STOP', j
            elif pct <= -rules['STOP_PCT']:
                return price, 'STOP_LOSS', j
            j += 1
        return c[n - 1], 'END_OF_DATA', n - 1
    return sim


def grid_candidates():
    """Replace candidates_for_symbol so every BUY is replayed under every grid combination."""
    orig_cands, orig_exit, orig_last = mf.candidates_for_symbol, mf.simulate_exit, mf.LAST_DECISION_MINUTE

    def patched(sym, df, thr, bear_by_day, bear_min_conf=0.68, max_hold=None):
        out = []
        for wname, last in WINDOWS.items():
            for ename, cmin in EXITS.items():
                mf.LAST_DECISION_MINUTE, mf.simulate_exit = last, make_exit(cmin)
                try:
                    rows = orig_cands(sym, df, thr, bear_by_day, bear_min_conf, max_hold)
                finally:
                    mf.LAST_DECISION_MINUTE, mf.simulate_exit = orig_last, orig_exit
                for r in rows:
                    r['combo'] = f'{wname} | {ename}'
                out += rows
        return out
    mf.candidates_for_symbol = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', choices=['dt100', 'b1'], default='b1')
    ap.add_argument('--label', choices=['classic', 'classic_day', 'trade'], default='classic')
    ap.add_argument('--folds', type=int, default=9)
    ap.add_argument('--fold-days', type=int, default=41)
    ap.add_argument('--watchlist', default=None, help='Bot 1 only: comma-separated tickers')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fake', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    if a.watchlist:
        wl = [x.strip().upper() for x in a.watchlist.split(',') if x.strip()]
        bm.bot.FALLBACK_WATCHLIST = list(wl)
        bm.bot.TRAINING_UNIVERSE = list(bm.bot.TRAINING_UNIVERSE) + [x for x in wl if x not in bm.bot.TRAINING_UNIVERSE]

    grid_candidates()
    tmp = Path(tempfile.mkdtemp()) / 'raw.json'
    sys.argv = ['backtest_meta.py', '--bot', a.bot, '--label', a.label, '--folds', str(a.folds),
                '--fold-days', str(a.fold_days), '--out', str(tmp)] + (['--fake'] if a.fake else [])
    bm.main()        # its own printed "RESULT" mixes all combinations: ignore it, the table below is per combination

    raw = json.loads(tmp.read_text())
    T = pd.read_csv(tmp.with_name(tmp.stem + '_calls.csv'))
    for col in ('t', 'entry_t', 'exit_t'):
        T[col] = pd.to_datetime(T[col], utc=True)
    folds = sorted(T['fold'].unique())

    rows = {}
    for combo, C in T.groupby('combo', sort=False):
        C = C.drop(columns=['combo'])
        tr = bm.portfolio(C)
        s = bm.stats(tr)
        rets = np.array([x['ret'] for x in tr]) if tr else np.array([])
        held_days = [pd.Timestamp(x['exit_t']).tz_convert('America/New_York').normalize()
                     != pd.Timestamp(x['entry_t']).tz_convert('America/New_York').normalize() for x in tr]
        rows[combo] = dict(
            account=s, worst_trade_pct=float(rets.min() * 100) if len(rets) else None,
            best_trade_pct=float(rets.max() * 100) if len(rets) else None,
            trades_losing_over_2pct=int((rets < -0.02).sum()), overnight_holds=int(sum(held_days)),
            per_fold_total_pnl=[round(bm.stats(bm.portfolio(C[C['fold'] == f])).get('total_pnl', 0.0), 2) for f in folds])
    live = rows[' | '.join(LIVE)]
    for r in rows.values():
        r['periods_beating_live'] = (f"{sum(x > y for x, y in zip(r['per_fold_total_pnl'], live['per_fold_total_pnl']))}"
                                     f" of {len(folds)}")

    report = dict(bot=a.bot, label=a.label, fake_data=a.fake,
                  watchlist=list(bm.bot.FALLBACK_WATCHLIST) if a.bot == 'b1' else 'top-100',
                  traded_symbols=raw['settings']['traded_symbols'], training_symbols=raw['settings']['training_symbols'],
                  test_period=raw.get('test_period'), spy_buy_hold_return=raw.get('spy_buy_hold_return'),
                  primary_cutoffs=[f['primary']['threshold'] for f in raw['folds']],
                  results=rows, runtime_sec=round(time.time() - t0),
                  disclaimer='Independent personal project, not tied to any employer or agency. Public market data '
                             'and paper money only. Backtest estimate with stated simplifications; not financial advice.')
    out = Path(f'state/backtest_exits_{a.bot}{"_" + a.tag if a.tag else ""}{"_fake" if a.fake else ""}.json')
    out.write_text(json.dumps(report, indent=2, default=str))

    spy = report['spy_buy_hold_return']
    print(f'\n=== EXIT / BUY-WINDOW GRID {a.bot} label={a.label} {report["test_period"]}'
          + (f' (SPY buy & hold {spy:+.2%})' if spy is not None else '')
          + f' | stocks {",".join(report["watchlist"]) if a.bot == "b1" else "top-100"}'
          + f' | traded {report["traded_symbols"]}, trained on {report["training_symbols"]}')
    print(f'{"buy window | exit":32s} | {"trades":>6s} {"hit":>6s} {"$/trade":>8s} {"PF":>5s} {"total":>7s} '
          f'{"maxDD":>7s} | {"worst":>6s} {"best":>6s} {">2% loss":>8s} {"overnight":>9s} | beats live')
    for combo, r in rows.items():
        s = r['account']
        if not s.get('trades'):
            print(f'{combo:32s} | no trades'); continue
        pf = '-' if s['profit_factor'] is None else f'{s["profit_factor"]:.2f}'
        print(f'{combo:32s} | {s["trades"]:6d} {s["hit_rate"]:6.1%} {s["profit_per_trade"]:8.2f} {pf:>5s} '
              f'{s["total_return"]:+7.2%} {s["max_drawdown"]:7.2%} | {r["worst_trade_pct"]:5.2f}% {r["best_trade_pct"]:5.2f}% '
              f'{r["trades_losing_over_2pct"]:8d} {r["overnight_holds"]:9d} | {r["periods_beating_live"]}')
        print(f'{"":32s}   exits {s["exits"]}')
    print(f'Saved {out} ({report["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
