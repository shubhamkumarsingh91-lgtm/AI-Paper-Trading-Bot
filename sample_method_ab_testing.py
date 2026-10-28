#!/usr/bin/env python3
"""
SAMPLE METHOD A/B TESTING: does the Time Machine make the meta filter more precise,
and which way of sampling the rest of the day works better?

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Public market data (Alpaca IEX bars) and paper money only. No internal, personal or
protected data (no SSNs/PHI). Backtest estimate, not a promise and not financial advice.

SAFETY: this file only READS history and writes to results/sample_method_ab/.
It never places orders and the live bots do not import it.

THE ARMS (all four see the SAME main-AI buy calls, the SAME walk-forward splits and the
SAME filter settings; the only thing that changes is what the filter is allowed to see)
  none   no filter: every buy call the main AI makes
  base   today's meta filter (meta_filter.META_FEATURES; it already includes atr_pct,
         so this arm IS the "plain ATR" control from the plan)
  slot   base + 5 Time Machine features, 'slot' sampling (each future bar from a random past day)
  day    base + 5 Time Machine features, 'day' sampling (one whole past day per path)
Plus, for reference, two no-training rules: skip a call when TM p_win - p_stop < 0.05.

ONE FOLD = same windows as backtest_meta.py:
  |-- main AI learns --|-- filter learns (2/3 fit, 1/3 picks strictness) --|-- TEST --|

WHAT "BETTER" MEANS: precision (share of kept calls that made money after costs) on the
TEST periods, read together with how many calls were kept and the money per trade.
A filter can look precise just by keeping very few calls, so all three are reported.

Usage:
  python sample_method_ab_testing.py --bot dt100                    # 100-stock universe
  python sample_method_ab_testing.py --bot b1 --buy-window 10-14    # Bot 1, buys 10:00-14:00 only
  python sample_method_ab_testing.py --bot b1 --fake --folds 2      # offline self-test, random data
"""
import argparse, json, os, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('STATE_DIR', tempfile.mkdtemp(prefix='ab_'))   # never touch the bots' real logs
import ai_trading_bot as bot                                          # noqa: E402
import backtest_meta as bm                                            # noqa: E402
import meta_filter as mf                                              # noqa: E402
import time_machine as tm                                             # noqa: E402

ARMS = {
    'base': list(mf.META_FEATURES),
    'slot': list(mf.META_FEATURES) + tm.feature_names('slot'),
    'day': list(mf.META_FEATURES) + tm.feature_names('day'),
}
RULE_EDGE = 0.05                    # reference rule: keep a call only if p_win - p_stop >= 5 points
OUT_DIR = Path('results/sample_method_ab')


def train_filter(X: pd.DataFrame, y: pd.Series, features):
    """Same model and settings as meta_filter.train_meta, with a chosen feature list."""
    import xgboost as xgb
    pos = max(1, int(y.sum()))
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, scale_pos_weight=(len(y) - pos) / pos,
        eval_metric='aucpr', verbosity=0, n_jobs=-1, random_state=42,
    )
    m.fit(X[features].astype(float), y)
    return m


def in_window(cands: pd.DataFrame, window):
    """Keep calls whose decision bar starts inside [start, end) hours, e.g. (10, 14)."""
    if window is None or cands.empty:
        return cands
    mins = pd.DatetimeIndex(cands['t']).hour * 60 + pd.DatetimeIndex(cands['t']).minute
    return cands[(mins >= window[0] * 60) & (mins < window[1] * 60)].reset_index(drop=True)


def call_stats(c: pd.DataFrame):
    """Call-level numbers; precision = share of calls that made money after costs."""
    if c.empty:
        return dict(n=0)
    r = c['ret'].to_numpy(float)
    neg = r[r <= 0].sum()
    return dict(n=int(len(r)), precision=float((r > 0).mean()), avg_ret=float(r.mean()),
                profit_factor=float(r[r > 0].sum() / -neg) if neg < 0 else None)


def auc(y, score):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(score)
    if ok.sum() < 20 or len(set(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], score[ok]))


def diagnostics(T: pd.DataFrame):
    """Does each sampling method's p_win rank winners above losers (AUC, 0.5 = coin flip)?
    And is it just restating volatility (rank correlation with atr_pct)?"""
    if T.empty:
        return {}
    y = T['win'].to_numpy(int)
    out = {}
    for m in tm.METHODS:
        pw = T[tm.PREFIX[m] + 'p_win'].to_numpy(float)
        out[m] = dict(auc_p_win=auc(y, pw), auc_exp_ret=auc(y, T[tm.PREFIX[m] + 'exp_ret'].to_numpy(float)),
                      spearman_with_atr_pct=float(T[[tm.PREFIX[m] + 'p_stop', 'atr_pct']].corr('spearman').iloc[0, 1]),
                      mean_p_win=float(np.nanmean(pw)), actual_win_rate=float(y.mean()))
    out['main_ai_p_auc'] = auc(y, T['p'].to_numpy(float))
    out['slot_vs_day_p_win_spearman'] = float(T[['tms_p_win', 'tmd_p_win']].corr('spearman').iloc[0, 1])
    out['calls_without_tm_features'] = int(T['tms_p_win'].isna().sum())
    return out


def bootstrap_vs_base(T: pd.DataFrame, arm, n_boot=2000, seed=7):
    """Is arm's precision gain over the base filter bigger than luck?
    Resamples whole test DAYS with replacement (calls on one day move together) and recomputes
    precision(arm) - precision(base) each time. Returns the observed gap, its 90% range and the
    share of resamples where the arm was ahead. A range that includes 0 = not proven."""
    if T.empty:
        return None
    day = pd.DatetimeIndex(T['t']).normalize()
    days = day.unique()
    win = (T['ret'] > 0).to_numpy()
    ka, kb = T[f'keep_{arm}'].to_numpy(bool), T['keep_base'].to_numpy(bool)
    codes = days.get_indexer(day)
    # per-day counts: kept calls and kept winners, for both arms
    na, wa = np.bincount(codes, ka, len(days)), np.bincount(codes, ka & win, len(days))
    nb, wb = np.bincount(codes, kb, len(days)), np.bincount(codes, kb & win, len(days))
    if na.sum() == 0 or nb.sum() == 0:
        return None
    obs = wa.sum() / na.sum() - wb.sum() / nb.sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(n_boot, len(days)))
    with np.errstate(invalid='ignore', divide='ignore'):
        diff = wa[pick].sum(1) / na[pick].sum(1) - wb[pick].sum(1) / nb[pick].sum(1)
    diff = diff[~np.isnan(diff)]
    lo, hi = np.percentile(diff, [5, 95])
    return dict(precision_gain=float(obs), range_90=[float(lo), float(hi)],
                share_of_resamples_ahead=float((diff > 0).mean()), test_days=int(len(days)),
                verdict='proven' if lo > 0 else ('worse' if hi < 0 else 'not proven (could be luck)'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', choices=['dt100', 'b1'], default='dt100')
    ap.add_argument('--folds', type=int, default=9)
    ap.add_argument('--fold-days', type=int, default=41)
    ap.add_argument('--meta-days', type=int, default=120)
    ap.add_argument('--train-days', type=int, default=None)
    ap.add_argument('--label', choices=['classic', 'classic_day', 'trade'], default='trade',
                    help='trade = did the real trade make money (the fixed trade-matched label)')
    ap.add_argument('--buy-window', default=None, help='e.g. 10-14 = only buys decided 10:00-14:00 New York time')
    ap.add_argument('--paths', type=int, default=tm.N_PATHS)
    ap.add_argument('--lookback', type=int, default=tm.LOOKBACK)
    ap.add_argument('--fake', action='store_true', help='offline self-test on random data')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t_start = time.time()
    window = tuple(int(x) for x in a.buy_window.split('-')) if a.buy_window else None

    if a.bot == 'dt100':
        trade_syms = bm.dt100_universe()
        train_syms = trade_syms + ['SPY']
        train_days = a.train_days or 365
    else:
        trade_syms = list(bot.FALLBACK_WATCHLIST)
        train_syms = list(bot.TRAINING_UNIVERSE)
        train_days = a.train_days or bot.TRAIN_DAYS
    span = train_days + a.meta_days + a.folds * a.fold_days + 45
    get = (lambda s: bm.fake_bars(s, span)) if a.fake else (lambda s: bot.fetch_bars(s, days=span))
    print(f'Sample-method A/B {a.bot}: {len(train_syms)} training symbols, {len(trade_syms)} traded, '
          f'{span} days of 15-min bars, {a.paths} paths, {a.lookback}-day pool, window {a.buy_window or "all day"}'
          f'{" (FAKE DATA)" if a.fake else ""}', flush=True)

    spy_raw = get('SPY')
    ctx, bear = bm.spy_context(spy_raw), mf.bear_days(spy_raw)
    data, raw_by_sym = {}, {}
    for i, s in enumerate(train_syms):
        try:
            raw = spy_raw if s == 'SPY' else get(s)
            if len(raw) < 500:
                print(f'  skip {s}: only {len(raw)} bars'); continue
            data[s] = bm.prep(s, raw, ctx, i, a.label)
            if s in trade_syms:
                raw_by_sym[s] = raw
        except Exception as e:
            print(f'  skip {s}: {e}')
    trade_syms = [s for s in trade_syms if s in data]
    print(f'  loaded {len(data)} symbols in {time.time() - t_start:.0f}s', flush=True)

    end = max(df.index.max() for df in data.values()).normalize() + pd.Timedelta(days=1)
    cache, folds, test_all = {}, [], []
    for k in range(a.folds):
        T0 = end - pd.Timedelta(days=(a.folds - k) * a.fold_days)
        T1 = T0 + pd.Timedelta(days=a.fold_days)
        M0 = T0 - pd.Timedelta(days=a.meta_days)
        P0 = M0 - pd.Timedelta(days=train_days)
        P1 = M0 - pd.Timedelta(days=1)
        train = pd.concat([df[(df.index >= P0) & (df.index < P1)] for df in data.values()])
        blend, pinfo = bm.train_primary(train)
        del train

        meta_c, test_c = [], []
        for s in trade_syms:
            w = data[s][(data[s].index >= M0) & (data[s].index < T1)].copy()
            if w.empty:
                continue
            w['p'] = blend(w[bot.FEATURES + ['sym_id']])
            meta_c += mf.candidates_for_symbol(s, w[w.index < T0], pinfo['threshold'], bear)
            test_c += mf.candidates_for_symbol(s, w[w.index >= T0], pinfo['threshold'], bear)
        meta_c = in_window(mf.add_market_context(pd.DataFrame(meta_c), spy_raw, len(trade_syms)), window)
        test_c = in_window(mf.add_market_context(pd.DataFrame(test_c), spy_raw, len(trade_syms)), window)
        t_tm = time.time()
        meta_c = tm.add_features(meta_c, raw_by_sym, a.paths, a.lookback, cache)
        test_c = tm.add_features(test_c, raw_by_sym, a.paths, a.lookback, cache)

        fold = dict(fold=k + 1, test=[str(T0.date()), str(T1.date())], main_ai=pinfo,
                    filter_calls=int(len(meta_c)), test_calls=int(len(test_c)),
                    tm_seconds=round(time.time() - t_tm, 1))
        if test_c.empty:
            folds.append(fold); continue
        test_c['fold'] = k + 1
        enough = len(meta_c) >= 60
        if enough:
            meta_c = meta_c.sort_values('t')
            cut_at = meta_c['t'].iloc[int(len(meta_c) * 2 / 3)]
            fit, pick = meta_c[meta_c['t'] < cut_at], meta_c[meta_c['t'] >= cut_at]
            enough = fit['win'].nunique() == 2 and len(pick) >= 10
        for arm, feats in ARMS.items():
            if enough:
                model = train_filter(fit, fit['win'], feats)
                cut, keep, table = mf.pick_cut(model.predict_proba(pick[feats].astype(float))[:, 1],
                                               pick['ret'].to_numpy())
                test_c[f'{arm}_p'] = model.predict_proba(test_c[feats].astype(float))[:, 1]
                fold[f'{arm}_filter'] = dict(cut=cut, keep=keep, pick_table=table,
                                             top_features=dict(sorted(zip(feats, map(float, model.feature_importances_)),
                                                                      key=lambda x: -x[1])[:6]))
            else:
                cut = 0.0
                test_c[f'{arm}_p'] = 1.0
                fold[f'{arm}_filter'] = dict(cut=0.0, keep=1.0, note='too few calls to train; filter off')
            test_c[f'keep_{arm}'] = test_c[f'{arm}_p'] >= cut
        test_c['keep_none'] = True
        for m in tm.METHODS:   # reference rule, no training: NaN features -> keep the call
            edge = test_c[tm.PREFIX[m] + 'p_win'] - test_c[tm.PREFIX[m] + 'p_stop']
            test_c[f'keep_rule_{m}'] = edge.isna() | (edge >= RULE_EDGE)
        for arm in ['none', *ARMS, 'rule_slot', 'rule_day']:
            fold[f'calls_{arm}'] = call_stats(test_c[test_c[f'keep_{arm}']])
        test_all.append(test_c)
        folds.append(fold)
        print(f'  fold {k + 1}: test {fold["test"]} | calls {len(test_c)} | precision '
              + ' | '.join(f'{arm} {fold[f"calls_{arm}"].get("precision", float("nan")):.1%} '
                           f'(n={fold[f"calls_{arm}"]["n"]})' for arm in ['none', *ARMS])
              + f' | TM {fold["tm_seconds"]}s', flush=True)

    T = pd.concat(test_all, ignore_index=True) if test_all else pd.DataFrame()
    arms_all = ['none', *ARMS, 'rule_slot', 'rule_day']
    summary, accounts = {}, {}
    for arm in arms_all:
        kept = T[T[f'keep_{arm}']] if not T.empty else T
        trades = bm.portfolio(kept)
        accounts[arm] = trades
        summary[arm] = dict(calls=call_stats(kept), account=bm.stats(trades))
    beats = {}
    for arm in ('slot', 'day'):
        wins = [f for f in folds if f.get(f'calls_{arm}', {}).get('n') and f.get('calls_base', {}).get('n')
                and f[f'calls_{arm}']['precision'] > f['calls_base']['precision']]
        beats[arm] = f'{len(wins)} of {sum(1 for f in folds if "calls_base" in f)}'
    spy_rth = spy_raw[spy_raw.index >= end - pd.Timedelta(days=a.folds * a.fold_days)]
    spy_ret = float(spy_rth['close'].iloc[-1] / spy_rth['open'].iloc[0] - 1) if len(spy_rth) else None

    report = dict(
        bot=a.bot, fake_data=a.fake,
        generated=pd.Timestamp.now(tz='America/New_York').isoformat(timespec='seconds'),
        settings=dict(label=a.label, buy_window=a.buy_window, paths=a.paths, lookback_days=a.lookback,
                      min_pool_days=tm.MIN_POOL, vol_clip=list(tm.VOL_CLIP), rule_edge=RULE_EDGE,
                      folds=a.folds, fold_days=a.fold_days, meta_days=a.meta_days, train_days=train_days,
                      cost_per_side=mf.COST_PER_SIDE, traded_symbols=len(trade_syms)),
        test_period=[folds[0]['test'][0], folds[-1]['test'][1]] if folds else None,
        spy_buy_hold_return=spy_ret,
        arms=summary,
        folds_where_precision_beat_base=beats,
        luck_check_vs_base={arm: bootstrap_vs_base(T, arm) for arm in ('slot', 'day')},
        diagnostics=diagnostics(T),
        folds=folds,
        runtime_sec=round(time.time() - t_start),
        disclaimer='Independent personal project, not tied to any employer or agency. Public market data and paper '
                   'money only; no internal or personal data. Backtest estimate with stated simplifications; '
                   'not financial advice.',
    )
    tag = f'{a.bot}_{a.label}' + (f'_w{a.buy_window}' if a.buy_window else '_allday') + ('_fake' if a.fake else '')
    out = Path(a.out or OUT_DIR / f'ab_{tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    if not T.empty:   # every test call with every arm's decision, so all numbers can be re-checked
        T.drop(columns=[c for c in ('day',) if c in T]).to_csv(out.with_name(out.stem + '_calls.csv'), index=False)
        pd.concat([pd.DataFrame(v).assign(arm=k) for k, v in accounts.items() if v]).to_csv(
            out.with_name(out.stem + '_trades.csv'), index=False)

    print(f'\n=== RESULT {report["test_period"]}' + (f' (SPY buy & hold {spy_ret:+.2%})' if spy_ret is not None else ''))
    print(f'{"arm":10s} {"calls":>6s} {"precision":>9s} {"avg/call":>9s} {"trades":>6s} {"$/trade":>9s} {"total":>8s} {"maxDD":>7s}')
    for arm in arms_all:
        c, s = summary[arm]['calls'], summary[arm]['account']
        if not c.get('n'):
            print(f'{arm:10s} no calls'); continue
        print(f'{arm:10s} {c["n"]:6d} {c["precision"]:9.1%} {c["avg_ret"]:+9.3%} {s.get("trades", 0):6d} '
              f'{s.get("profit_per_trade", 0):9,.2f} {s.get("total_return", 0):+8.2%} {s.get("max_drawdown", 0):7.2%}')
    print('Precision beat the base filter in:', beats)
    for arm, b in report['luck_check_vs_base'].items():
        if b:
            print(f'Luck check {arm} vs base: gain {b["precision_gain"]:+.1%}, 90% range '
                  f'[{b["range_90"][0]:+.1%}, {b["range_90"][1]:+.1%}] over {b["test_days"]} days -> {b["verdict"]}')
    print('Diagnostics:', json.dumps(report['diagnostics'], default=str))
    print(f'Saved {out} ({report["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
