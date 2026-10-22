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
import argparse, ast, json, os, sys, time, tempfile
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
    rng = np.random.default_rng(abs(hash(symbol)) % 10_000 + seed_base)
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


def spy_context(spy):
    c = spy['close']
    ctx = pd.DataFrame(index=spy.index)
    ctx['spy_ret_1'] = c.pct_change(1)
    ctx['spy_ret_5'] = c.pct_change(5)
    ctx['spy_above_ema20'] = (c > c.ewm(span=20, adjust=False).mean()).astype(float)
    return ctx.fillna(0.0)


def bear_days(spy):
    """bear[day] = 1 if the PREVIOUS day's SPY close was below the 50-day average of closes up to then."""
    daily = spy['close'].resample('1D').last().dropna()
    sma = daily.rolling(50).mean()
    flag = (daily < sma).astype(int).shift(1).fillna(0).astype(int)
    return {pd.Timestamp(d).normalize(): int(v) for d, v in flag.items()}


def prep(sym, raw, ctx, sym_id):
    df = bot.add_features(raw)
    df = df.join(ctx, how='left', rsuffix='_spy')
    df[SPY_COLS] = df[SPY_COLS].ffill().fillna(0.0)
    df['target'] = bot.make_labels(df)
    df['sym_id'] = sym_id
    df = df.dropna(subset=bot.FEATURES + ['target'])
    num = df.select_dtypes('float64').columns
    df[num] = df[num].astype('float32')
    return df


def train_primary(train):
    """Same models and settings as train_model(): XGB + LGB, 60/40 blend, cutoff calibrated for precision."""
    import xgboost as xgb
    import lightgbm as lgb
    train = train.sort_index()
    X, y = train[bot.FEATURES + ['sym_id']], train['target']
    k = int(len(X) * 0.8)
    spw = (y == 0).sum() / ((y == 1).sum() + 1)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
                          eval_metric='aucpr', verbosity=0, n_jobs=-1, random_state=42)
    m.fit(X.iloc[:k], y.iloc[:k])
    lg = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, max_depth=6,
                        min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
                        bagging_freq=5, scale_pos_weight=spw, verbose=-1, n_jobs=-1),
                   lgb.Dataset(X.iloc[:k], label=y.iloc[:k]), num_boost_round=200)
    blend = lambda Z: 0.6 * m.predict_proba(Z)[:, 1] + 0.4 * lg.predict(Z)
    pv = blend(X.iloc[k:])
    yv = y.iloc[k:]
    thr, prec, n_sig = bot.calibrate_threshold(pv, yv, max(10, int(0.01 * len(yv))))
    return blend, dict(threshold=thr, val_precision=prec, val_signals=n_sig,
                       base_rate=float(yv.mean()), train_rows=int(len(X)))


def stats(trades, start_equity=100_000.0):
    """Money stats on a list of simulated portfolio trades (dicts with pnl, ret, reason, exit_t)."""
    if not trades:
        return dict(trades=0)
    pnl = np.array([t['pnl'] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = start_equity + np.cumsum(pnl[np.argsort([t['exit_t'] for t in trades])])
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


def portfolio(cands, start_equity=100_000.0, max_pos=5, cap=0.12, bear_cap=0.06):
    """$100k account: one buy per bar (highest main-AI confidence), max 5 open, 12% per trade (6% bear)."""
    if cands.empty:
        return []
    equity, open_pos, done = start_equity, [], []
    for t, grp in cands.sort_values('t').groupby('t', sort=True):
        entry_t = grp['entry_t'].iloc[0]
        still = []
        for pos in sorted(open_pos, key=lambda q: q['exit_t']):
            if pos['exit_t'] < entry_t:
                equity += pos['pnl']; done.append(pos)
            else:
                still.append(pos)
        open_pos = still
        if len(open_pos) >= max_pos:
            continue
        held = {q['sym'] for q in open_pos}
        g = grp[~grp['sym'].isin(held)]
        if g.empty:
            continue
        best = g.loc[g['p'].idxmax()]
        size = equity * (bear_cap if best['bear'] == 1 else cap)
        open_pos.append(dict(sym=best['sym'], t=best['t'], entry_t=best['entry_t'], exit_t=best['exit_t'],
                             ret=float(best['ret']), pnl=float(size * best['ret']), size=float(size),
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
    ap.add_argument('--fake', action='store_true', help='offline self-test on random data')
    ap.add_argument('--out', default=None)
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
    get = (lambda s: fake_bars(s, span)) if a.fake else (lambda s: bot.fetch_bars(s, days=span))
    print(f'Backtest {a.bot}: {len(train_syms)} training symbols, {len(trade_syms)} traded, '
          f'{span} calendar days of 15-min bars{" (FAKE DATA)" if a.fake else ""}', flush=True)

    spy_raw = get('SPY')
    ctx, bear = spy_context(spy_raw), bear_days(spy_raw)
    data = {}
    for i, s in enumerate(train_syms):
        try:
            raw = spy_raw if s == 'SPY' else get(s)
            if len(raw) < 500:
                print(f'  skip {s}: only {len(raw)} bars'); continue
            data[s] = prep(s, raw, ctx, i)
        except Exception as e:
            print(f'  skip {s}: {e}')
    print(f'  loaded {len(data)} symbols in {time.time() - t_start:.0f}s', flush=True)
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
        blend, pinfo = train_primary(train)
        del train

        meta_c, test_c = [], []
        for s in trade_syms:
            df = data[s]
            w = df[(df.index >= M0) & (df.index < T1)].copy()
            if w.empty:
                continue
            w['p'] = blend(w[bot.FEATURES + ['sym_id']])
            meta_c += mf.candidates_for_symbol(s, w[w.index < T0], pinfo['threshold'], bear)
            test_c += mf.candidates_for_symbol(s, w[w.index >= T0], pinfo['threshold'], bear)
        meta_c = mf.add_market_context(pd.DataFrame(meta_c), spy_raw, len(trade_syms))
        test_c = mf.add_market_context(pd.DataFrame(test_c), spy_raw, len(trade_syms))

        fold = dict(fold=k + 1, primary_train=[str(P0.date()), str(P1.date())],
                    filter_window=[str(M0.date()), str(T0.date())], test=[str(T0.date()), str(T1.date())],
                    primary=pinfo, filter_calls=int(len(meta_c)), test_calls=int(len(test_c)))
        if len(meta_c) >= 60 and not test_c.empty:
            meta_c = meta_c.sort_values('t')
            cut_at = meta_c['t'].iloc[int(len(meta_c) * 2 / 3)]
            fit, pick = meta_c[meta_c['t'] < cut_at], meta_c[meta_c['t'] >= cut_at]
            meta = mf.train_meta(fit, fit['win'])
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
    base_trades = portfolio(T)
    filt_trades = portfolio(T[T['keep']]) if not T.empty else []
    spy_rth = spy_raw[(spy_raw.index >= end - pd.Timedelta(days=a.folds * a.fold_days))]
    spy_ret = float(spy_rth['close'].iloc[-1] / spy_rth['open'].iloc[0] - 1) if len(spy_rth) else None

    # per-fold portfolio comparison (did the filter help in this period?)
    for f in folds:
        Tf = T[T['fold'] == f['fold']] if not T.empty else T
        f['account_without_filter'] = stats(portfolio(Tf))
        f['account_with_filter'] = stats(portfolio(Tf[Tf['keep']])) if not Tf.empty else dict(trades=0)
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
        settings=dict(train_days=train_days, meta_days=a.meta_days, fold_days=a.fold_days, folds=a.folds,
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
    out = Path(a.out or f'state/backtest_meta_{a.bot}{"_fake" if a.fake else ""}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
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
    main()
