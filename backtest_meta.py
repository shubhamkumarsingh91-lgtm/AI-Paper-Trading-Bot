#!/usr/bin/env python3
"""
WALK-FORWARD BACKTEST of the main AI, with and without the META FILTER.

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Public market data (Alpaca IEX bars) and paper money only. No internal, personal
or protected data (no SSNs/PHI). Results are a research estimate, not a promise.

WHY WALK-FORWARD: every model is only ever tested on dates AFTER everything it learned
from, the way it would have run live. Nothing from the future leaks backwards.

ONE FOLD (repeated for several back-to-back test periods):

  |---- main AI learns (TRAIN_DAYS) ----|-- filter learns (META_DAYS) --|-- TEST (FOLD_DAYS) --|
                                        ^ 1-day gap so labels can't peek ahead

  1. Main AI (same XGBoost + LightGBM 60/40 blend and labels as the live bot) trains on
     the first 80% of its window; its BUY cutoff is calibrated for precision on the last
     20%, exactly like train_model() does each night.
  2. It scores the filter window. Every BUY call there (precision-first rule: cutoff,
     RSI <= 78, above EMA50, not at the top of the Bollinger band, 68% in a bear market,
     no buys after 15:30 bar close) is replayed with the bot's real exit rules -> win/lose.
  3. Filter trains on the first 2/3 of those calls and picks how strict to be on the last
     1/3 (data it never trained on).
  4. TEST: main AI scores the test period; trades are simulated twice, without and with
     the filter, as a $100k account: 1 buy per 15-min bar (the highest-confidence call),
     max 5 positions, 12% of equity per trade (6% in a bear market), 0.03% cost per side.

KNOWN SIMPLIFICATIONS (stated so nobody over-trusts the numbers):
  - exits are checked on 15-min bar closes; the live bot checks every 10 minutes
  - no VIX gate, earnings blackout, dip-watch re-entry or TradingView STRONG_SELL veto
  - no time-decay sample weights or real-trade relabeling in the main AI
  - bear market = yesterday's SPY close below its 50-day average

Usage:
  python backtest_meta.py --bot dt100          # Day-Trade-100 universe (100 stocks)
  python backtest_meta.py --bot b1             # Bot 1 (5-stock watchlist, 14-stock training set)
  python backtest_meta.py --bot b1 --fake      # offline self-test on random data
"""
import argparse, ast, json, os, sys, time, tempfile, zlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('STATE_DIR', tempfile.mkdtemp(prefix='bt_'))   # never touch the bots' real logs
import ai_trading_bot as bot                                          # noqa: E402
import meta_filter as mf                                              # noqa: E402
from sklearn.metrics import precision_score                           # noqa: E402

SPY_COLS = ['spy_ret_1', 'spy_ret_5', 'spy_above_ema20']


def dt100_universe():
    tree = ast.parse(Path(__file__).with_name('daytrade100.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'UNIVERSE_100':
            return ast.literal_eval(node.value)
    raise RuntimeError('UNIVERSE_100 not found in daytrade100.py')


def fake_bars(symbol, days, seed_base=7):
    """Random-walk 15-min regular-hours bars, for the offline self-test only."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()) % 10_000 + seed_base)   # same data every run
    end = pd.Timestamp('2026-10-22', tz='America/New_York')
    dates = pd.bdate_range(end - pd.Timedelta(days=days), end, tz='America/New_York')
    idx = [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * k) for d in dates for k in range(26)]
    idx = pd.DatetimeIndex(idx)
    r = rng.normal(0.0001, 0.004, len(idx))
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, len(idx))))
    vol = rng.integers(1_000, 50_000, len(idx)).astype(float)
    return pd.DataFrame(dict(open=open_, high=high, low=low, close=close, volume=vol), index=idx)


def fetch_bars_until(symbol, days, end_ts=None):
    """Same request as bot.fetch_bars (15-min IEX bars, split-adjusted) but with a FIXED end date,
    so every run and every seed sees exactly the same data."""
    if end_ts is None:
        return bot.fetch_bars(symbol, days=days)
    from alpaca.data.requests import StockBarsRequest
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=bot.TF, start=end_ts - pd.Timedelta(days=days),
                           end=end_ts, adjustment='all', feed='iex')
    for attempt in range(10):                      # Alpaca allows ~200 requests/minute per key: back off and retry
        try:
            df = bot.data_client.get_stock_bars(req).df
            break
        except Exception as e:
            if 'too many' not in str(e).lower() or attempt == 9:
                raise
            wait = 15 * (attempt + 1)
            print(f'  {symbol}: rate limited, waiting {wait}s', flush=True)
            time.sleep(wait)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    return df.sort_index()


def spy_context(spy):
    c = spy['close']
    ctx = pd.DataFrame(index=spy.index)
    ctx['spy_ret_1'] = c.pct_change(1)
    ctx['spy_ret_5'] = c.pct_change(5)
    ctx['spy_above_ema20'] = (c > c.ewm(span=20, adjust=False).mean()).astype(float)
    return ctx.fillna(0.0)


def prep(sym, raw, ctx, sym_id, label='classic', max_hold=None):
    df = bot.add_features(raw)
    df = df.join(ctx, how='left', rsuffix='_spy')
    df[SPY_COLS] = df[SPY_COLS].ffill().fillna(0.0)
    df['trade_ret'] = mf.trade_labels(df, max_hold)             # what a trade from this bar earned (NaN = can't buy here)
    if label == 'trade':
        df['target'] = (df['trade_ret'] > 0).astype(float).where(df['trade_ret'].notna())
    else:
        df['target'] = bot.make_labels(df)
        if label == 'classic_day':
            # same +0.5%-in-2h label, but only for regular-hours bars whose 2-hour window ends by 15:45
            # (the live label lets bars after 13:45 look into the NEXT morning)
            m = df.index.hour * 60 + df.index.minute
            df.loc[(m < mf.OPEN_MINUTE) | (m > mf.EOD_MINUTE - 8 * 15), 'target'] = np.nan
    df['sym_id'] = sym_id
    df = df.dropna(subset=bot.FEATURES)                # keep every bar: exits need the bars after 15:15
    num = df.select_dtypes('float64').columns
    df[num] = df[num].astype('float32')
    return df


def train_primary(train, cutoff='precision', trade_ids=None, per_day=3, seed=42):
    """Same models and settings as train_model(): XGB + LGB, 60/40 blend.
    cutoff='precision' = the live rule (best precision); 'expectancy' = most money per call."""
    import xgboost as xgb
    import lightgbm as lgb
    train = train[train['target'].notna()].sort_index()
    X, y = train[bot.FEATURES + ['sym_id']], train['target']
    k = int(len(X) * 0.8)
    spw = (y == 0).sum() / ((y == 1).sum() + 1)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
                          eval_metric='aucpr', verbosity=0, n_jobs=-1, random_state=seed)
    m.fit(X.iloc[:k], y.iloc[:k])
    lg = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, max_depth=6,
                        min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
                        bagging_freq=5, scale_pos_weight=spw, verbose=-1, n_jobs=-1,
                        seed=seed, deterministic=True, force_row_wise=True),   # same seed -> same model every run
                   lgb.Dataset(X.iloc[:k], label=y.iloc[:k]), num_boost_round=200)
    blend = lambda Z: 0.6 * m.predict_proba(Z)[:, 1] + 0.4 * lg.predict(Z)
    pv = blend(X.iloc[k:])
    yv = y.iloc[k:]
    thr, prec, n_sig = bot.calibrate_threshold(pv, yv, max(10, int(0.01 * len(yv))))
    info = dict(threshold=thr, val_precision=prec, val_signals=n_sig,
                base_rate=float(yv.mean()), train_rows=int(len(X)), cutoff_rule=cutoff)
    if cutoff == 'expectancy':
        V = train.iloc[k:]
        ok = (V['rsi'] <= 78) & (V['above_ema50'] == 1) & (V['bb_pct'] <= 0.95)
        th, avg, n, hit = mf.calibrate_expectancy(pv, V['trade_ret'], ok, max(30, int(0.002 * len(V))))
        if n:
            info.update(threshold=th, val_avg_ret=avg, val_calls=n, val_hit=hit,
                        val_precision=float(precision_score(yv, (pv >= th).astype(int), zero_division=0)))
    if cutoff == 'rate':
        V = train.iloc[k:]
        ok = (V['rsi'] <= 78) & (V['above_ema50'] == 1) & (V['bb_pct'] <= 0.95)
        if trade_ids is not None:
            ok &= V['sym_id'].isin(trade_ids)
        mins = V.index.hour * 60 + V.index.minute
        th, per = mf.calibrate_rate(pv, mins, V.index.normalize(), ok, per_day)
        info.update(threshold=th, val_calls_per_day=per,
                    val_precision=float(precision_score(yv, (pv >= th).astype(int), zero_division=0)))
    return blend, info


def stats(trades, start_equity=100_000.0):
    """Money stats on a list of simulated portfolio trades (dicts with pnl, ret, reason, exit_t)."""
    if not trades:
        return dict(trades=0)
    pnl = np.array([t['pnl'] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = start_equity + np.cumsum(pnl[np.argsort([str(t['exit_t']) for t in trades], kind='stable')])   # stable: same-time exits keep their order
    peak = np.maximum.accumulate(np.r_[start_equity, eq])
    mdd = float(np.min(np.r_[start_equity, eq] / peak - 1))
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(losses.mean()) if len(losses) else 0.0
    return dict(
        trades=int(len(pnl)), wins=int(len(wins)), losses=int(len(losses)),
        hit_rate=float(len(wins) / len(pnl)),
        avg_win=aw, avg_loss=al,
        break_even_hit=float(abs(al) / (aw + abs(al))) if (aw + abs(al)) > 0 else None,
        profit_per_trade=float(pnl.mean()),
        profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        total_pnl=float(pnl.sum()), total_return=float(pnl.sum() / start_equity),
        max_drawdown=mdd,
        exits={r: int(sum(1 for t in trades if t['reason'] == r)) for r in sorted({t['reason'] for t in trades})},
    )


def portfolio(cands, start_equity=100_000.0, max_pos=5, cap=0.12, bear_cap=0.06, per_day=None,
              reentry_rules=False):
    """$100k account: one buy per bar (highest main-AI confidence), max 5 open, 12% per trade (6% bear).
    per_day = optional cap on new buys per day (first come, first served: no peeking at later calls).
    reentry_rules = guard against re-buying a stock that already topped out today:
      1. after a STOP_LOSS on a stock, no more buys of it that day ("one strike")
      2. after a profit exit, re-buy only if the new entry price is ABOVE the price it sold at
      3. at most 1 re-entry per stock per day        4. no re-entries after 14:30
      5. a re-entry is half size"""
    if cands.empty:
        return []
    equity, open_pos, done, bought = start_equity, [], [], {}
    today_log = {}            # (sym, day) -> dict(entries, struck, last_exit)
    for t, grp in cands.sort_values('t').groupby('t', sort=True):
        entry_t = grp['entry_t'].iloc[0]
        still = []
        for pos in sorted(open_pos, key=lambda q: q['exit_t']):
            if pos['exit_t'] < entry_t:
                equity += pos['pnl']; done.append(pos)
                L = today_log.setdefault((pos['sym'], pd.Timestamp(pos['t']).normalize()), dict(entries=0, struck=False))
                L['last_exit'] = pos['exit']
                L['struck'] |= pos['reason'] == 'STOP_LOSS'
            else:
                still.append(pos)
        open_pos = still
        if len(open_pos) >= max_pos:
            continue
        day = pd.Timestamp(t).normalize()
        if per_day and bought.get(day, 0) >= per_day:
            continue
        held = {q['sym'] for q in open_pos}
        g = grp[~grp['sym'].isin(held)]
        if reentry_rules and not g.empty:
            late = (pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute) > 14 * 60 + 30
            def allowed(r):
                L = today_log.get((r['sym'], day))
                if not L or L['entries'] == 0:
                    return True                                  # first buy of this stock today
                return (not L['struck'] and L['entries'] < 2 and not late
                        and 'last_exit' in L and r['entry'] > L['last_exit'])
            g = g[g.apply(allowed, axis=1)]
        if g.empty:
            continue
        best = g.loc[g['p'].idxmax()]
        size = equity * (bear_cap if best['bear'] == 1 else cap)
        if reentry_rules:
            L = today_log.setdefault((best['sym'], day), dict(entries=0, struck=False))
            if L['entries'] >= 1:
                size *= 0.5                                      # re-entry = half size
            L['entries'] += 1
        bought[day] = bought.get(day, 0) + 1
        open_pos.append(dict(sym=best['sym'], t=best['t'], entry_t=best['entry_t'], exit_t=best['exit_t'],
                             ret=float(best['ret']), pnl=float(size * best['ret']), size=float(size),
                             entry=float(best.get('entry', np.nan)), exit=float(best.get('exit', np.nan)),
                             reason=best['reason'], p=float(best['p']),
                             meta_p=float(best['meta_p']) if 'meta_p' in best else None))
    done += open_pos
    return done


def cand_stats(c):
    if c.empty:
        return dict(n=0)
    r = c['ret'].to_numpy()
    return dict(n=int(len(r)), hit_rate=float((r > 0).mean()), avg_ret=float(r.mean()),
                profit_factor=float(r[r > 0].sum() / -r[r <= 0].sum()) if (r <= 0).any() and r[r <= 0].sum() < 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', choices=['dt100', 'b1'], default='dt100')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--fold-days', type=int, default=42)
    ap.add_argument('--meta-days', type=int, default=120)
    ap.add_argument('--train-days', type=int, default=None)
    ap.add_argument('--label', choices=['classic', 'classic_day', 'trade'], default='classic',
                    help='classic = live label (+0.5%% in 2h); trade = did the real trade make money')
    ap.add_argument('--cutoff', choices=['precision', 'expectancy', 'rate'], default='precision',
                    help='rate = steady cutoff giving ~per-day calls; account takes at most --per-day buys a day')
    ap.add_argument('--per-day', type=int, default=3)
    ap.add_argument('--max-hold-bars', type=int, default=None, help='time exit, e.g. 8 = sell after 2 hours')
    ap.add_argument('--tag', default=None, help='name used in the output file')
    ap.add_argument('--reentry-rules', action='store_true', help='one strike / re-buy only above the sell price / 1 re-entry / none after 14:30 / half size')
    ap.add_argument('--fake', action='store_true', help='offline self-test on random data')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=42, help='model seed; run several to see how much is luck')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD: last day of data (fixed, so every run sees the same bars)')
    ap.add_argument('--cache', default=None, help='folder to save/read downloaded bars')
    ap.add_argument('--fetch-only', action='store_true', help='download (into --cache) and stop')
    ap.add_argument('--window', default=None, help='buy only between these ET times, e.g. 10:00-14:00')
    a = ap.parse_args()
    t_start = time.time()

    if a.bot == 'dt100':
        trade_syms = dt100_universe()
        train_syms = trade_syms + ['SPY']
        train_days = a.train_days or 365          # daytrade100.py: bot.TRAIN_DAYS = 365
    else:
        trade_syms = list(bot.FALLBACK_WATCHLIST)
        train_syms = list(bot.TRAINING_UNIVERSE)
        train_days = a.train_days or bot.TRAIN_DAYS
    span = train_days + a.meta_days + a.folds * a.fold_days + 45
    # 16:30 ET on the end day: after the close, and never in the future on that day's evening runs
    end_ts = min(pd.Timestamp(a.end + ' 16:30', tz='America/New_York').tz_convert('UTC'),
                 pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=20)) if a.end else None
    fetch = (lambda s: fake_bars(s, span)) if a.fake else (lambda s: fetch_bars_until(s, span, end_ts))

    def get(sym):
        """--cache DIR: read bars saved by a --fetch-only run, so all runs use identical data and hit Alpaca once."""
        if not a.cache:
            return fetch(sym)
        f = Path(a.cache) / f'{a.bot}_{sym}_{span}d_{a.end or "now"}.pkl.gz'
        if f.exists():
            return pd.read_pickle(f)
        df = fetch(sym)
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(f)
        return df
    win = None
    if a.window:
        w0, w1 = [int(x.split(':')[0]) * 60 + int(x.split(':')[1]) for x in a.window.split('-')]
        win = (w0, w1)
    print(f'Backtest {a.bot}: {len(train_syms)} training symbols, {len(trade_syms)} traded, '
          f'{span} calendar days of 15-min bars{" (FAKE DATA)" if a.fake else ""}', flush=True)

    spy_raw = get('SPY')
    ctx, bear = spy_context(spy_raw), mf.bear_days(spy_raw)
    data = {}
    for i, s in enumerate(train_syms):
        try:
            raw = spy_raw if s == 'SPY' else get(s)
            if len(raw) < 500:
                print(f'  skip {s}: only {len(raw)} bars'); continue
            data[s] = prep(s, raw, ctx, i, a.label, a.max_hold_bars)
        except Exception as e:
            print(f'  skip {s}: {e}')
    print(f'  loaded {len(data)} symbols in {time.time() - t_start:.0f}s', flush=True)
    if a.fetch_only:
        print('fetch-only: bars saved to', a.cache); return
    trade_syms = [s for s in trade_syms if s in data]

    end = max(df.index.max() for df in data.values()).normalize() + pd.Timedelta(days=1)
    folds, test_all = [], []
    for k in range(a.folds):
        T0 = end - pd.Timedelta(days=(a.folds - k) * a.fold_days)
        T1 = T0 + pd.Timedelta(days=a.fold_days)
        M0 = T0 - pd.Timedelta(days=a.meta_days)
        P0 = M0 - pd.Timedelta(days=train_days)
        P1 = M0 - pd.Timedelta(days=1)                         # 1-day gap: labels look 8 bars ahead
        train = pd.concat([df[(df.index >= P0) & (df.index < P1)] for df in data.values()])
        blend, pinfo = train_primary(train, a.cutoff, [data[s]['sym_id'].iloc[0] for s in trade_syms], a.per_day, seed=a.seed)
        del train

        meta_c, test_c = [], []
        for s in trade_syms:
            df = data[s]
            w = df[(df.index >= M0) & (df.index < T1)].copy()
            if w.empty:
                continue
            w['p'] = blend(w[bot.FEATURES + ['sym_id']])
            meta_c += mf.candidates_for_symbol(s, w[w.index < T0], pinfo['threshold'], bear, max_hold=a.max_hold_bars, buy_window=win)
            test_c += mf.candidates_for_symbol(s, w[w.index >= T0], pinfo['threshold'], bear, max_hold=a.max_hold_bars, buy_window=win)
        meta_c = mf.add_market_context(pd.DataFrame(meta_c), spy_raw, len(trade_syms))
        test_c = mf.add_market_context(pd.DataFrame(test_c), spy_raw, len(trade_syms))

        fold = dict(fold=k + 1, primary_train=[str(P0.date()), str(P1.date())],
                    filter_window=[str(M0.date()), str(T0.date())], test=[str(T0.date()), str(T1.date())],
                    primary=pinfo, filter_calls=int(len(meta_c)), test_calls=int(len(test_c)))
        if len(meta_c) >= 60 and not test_c.empty:
            meta_c = meta_c.sort_values('t')
            cut_at = meta_c['t'].iloc[int(len(meta_c) * 2 / 3)]
            fit, pick = meta_c[meta_c['t'] < cut_at], meta_c[meta_c['t'] >= cut_at]
            meta = mf.train_meta(fit, fit['win'], seed=a.seed)
            mp = meta.predict_proba(pick[mf.META_FEATURES].astype(float))[:, 1]
            cut, keep, table = mf.pick_cut(mp, pick['ret'].to_numpy())
            test_c['meta_p'] = meta.predict_proba(test_c[mf.META_FEATURES].astype(float))[:, 1]
            fold.update(filter_cut=cut, filter_keep=keep, filter_pick_table=table,
                        filter_importance=dict(sorted(zip(mf.META_FEATURES, map(float, meta.feature_importances_)),
                                                      key=lambda x: -x[1])[:8]))
        else:
            test_c['meta_p'] = 1.0
            fold.update(filter_cut=0.0, filter_keep=1.0, note='too few calls to train the filter; filter off')
        test_c['keep'] = test_c['meta_p'] >= fold['filter_cut']
        test_c['fold'] = k + 1
        fold['test_calls_kept'] = int(test_c['keep'].sum())
        fold['calls_without_filter'] = cand_stats(test_c)
        fold['calls_with_filter'] = cand_stats(test_c[test_c['keep']])
        test_all.append(test_c)
        folds.append(fold)
        print(f'  fold {k + 1}: test {fold["test"]} | cutoff {pinfo["threshold"]:.2f} '
              f'(val precision {pinfo["val_precision"]:.1%} vs base {pinfo["base_rate"]:.1%}) | '
              f'calls {len(test_c)} -> kept {fold["test_calls_kept"]} (keep {fold["filter_keep"]:.0%})', flush=True)

    T = pd.concat(test_all, ignore_index=True) if test_all else pd.DataFrame()
    base_trades = portfolio(T, per_day=a.per_day if a.cutoff == 'rate' else None, reentry_rules=a.reentry_rules)
    filt_trades = portfolio(T[T['keep']], per_day=a.per_day if a.cutoff == 'rate' else None, reentry_rules=a.reentry_rules) if not T.empty else []
    spy_rth = spy_raw[(spy_raw.index >= end - pd.Timedelta(days=a.folds * a.fold_days))]
    spy_ret = float(spy_rth['close'].iloc[-1] / spy_rth['open'].iloc[0] - 1) if len(spy_rth) else None

    # per-fold portfolio comparison (did the filter help in this period?)
    for f in folds:
        Tf = T[T['fold'] == f['fold']] if not T.empty else T
        f['account_without_filter'] = stats(portfolio(Tf, per_day=a.per_day if a.cutoff == 'rate' else None, reentry_rules=a.reentry_rules))
        f['account_with_filter'] = stats(portfolio(Tf[Tf['keep']], per_day=a.per_day if a.cutoff == 'rate' else None, reentry_rules=a.reentry_rules)) if not Tf.empty else dict(trades=0)
    helped = sum(1 for f in folds if f['account_with_filter'].get('trades', 0) and
                 f['account_with_filter'].get('profit_per_trade', -1e9) > f['account_without_filter'].get('profit_per_trade', -1e9))

    # transparency only (NOT used to choose anything): every keep share on the pooled test calls
    grid = []
    if not T.empty and 'meta_p' in T:
        order = np.argsort(-T['meta_p'].to_numpy())
        r = T['ret'].to_numpy()[order]
        for kk in (0.3, 0.5, 0.7, 1.0):
            n = max(1, int(round(kk * len(r))))
            grid.append(dict(keep=kk, n=n, hit_rate=float((r[:n] > 0).mean()), avg_ret=float(r[:n].mean())))

    report = dict(
        bot=a.bot, fake_data=a.fake, generated=pd.Timestamp.now(tz='America/New_York').isoformat(timespec='seconds'),
        settings=dict(seed=a.seed, end=a.end, window=a.window, label=a.label, cutoff=a.cutoff, per_day=a.per_day if a.cutoff == 'rate' else None, reentry_rules=a.reentry_rules, max_hold_bars=a.max_hold_bars, train_days=train_days, meta_days=a.meta_days, fold_days=a.fold_days, folds=a.folds,
                      cost_per_side=mf.COST_PER_SIDE, traded_symbols=len(trade_syms), training_symbols=len(data)),
        test_period=[folds[0]['test'][0], folds[-1]['test'][1]] if folds else None,
        spy_buy_hold_return=spy_ret,
        calls_without_filter=cand_stats(T), calls_with_filter=cand_stats(T[T['keep']]) if not T.empty else dict(n=0),
        account_without_filter=stats(base_trades), account_with_filter=stats(filt_trades),
        folds_where_filter_helped=f'{helped} of {len(folds)}',
        test_keep_grid_for_reference_only=grid,
        folds=folds,
        runtime_sec=round(time.time() - t_start),
        disclaimer='Independent personal project, not tied to any employer or agency. Public market data and paper money only. '
                   'Backtest estimate with stated simplifications; not financial advice.',
    )
    tag = f'_{a.tag}' if a.tag else ('' if (a.label, a.cutoff) == ('classic', 'precision') else f'_{a.label}_{a.cutoff}')
    out = Path(a.out or f'state/backtest_meta_{a.bot}{tag}{"_fake" if a.fake else ""}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    if not T.empty:   # every test BUY call, so the call-level numbers can be re-checked independently
        T.drop(columns=[c for c in ('day',) if c in T]).to_csv(out.with_name(out.stem + '_calls.csv'), index=False)
    trades_out = out.with_name(out.stem + '_trades.csv')
    pd.concat([pd.DataFrame(base_trades).assign(variant='no_filter'),
               pd.DataFrame(filt_trades).assign(variant='filter')]).to_csv(trades_out, index=False)

    def line(name, s):
        if not s.get('trades'):
            return f'{name:16s} no trades'
        return (f'{name:16s} trades {s["trades"]:4d} | hit {s["hit_rate"]:.1%} (break-even {s["break_even_hit"]:.1%}) | '
                f'avg win ${s["avg_win"]:,.0f} / loss ${s["avg_loss"]:,.0f} | per trade ${s["profit_per_trade"]:,.2f} | '
                f'PF {s["profit_factor"] if s["profit_factor"] is None else round(s["profit_factor"], 2)} | '
                f'total {s["total_return"]:+.2%} | max DD {s["max_drawdown"]:.2%}')
    print('\n=== RESULT', report['test_period'], f'(SPY buy & hold {spy_ret:+.2%})' if spy_ret is not None else '')
    print(line('Without filter', report['account_without_filter']))
    print(line('With filter', report['account_with_filter']))
    print('Filter helped profit per trade in', report['folds_where_filter_helped'], 'test periods')
    print(f'Saved {out} and {trades_out} ({report["runtime_sec"]}s)')


if __name__ == '__main__':
    try:
        main()
    except Exception:                       # print the error into the saved output too, then fail loudly
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
