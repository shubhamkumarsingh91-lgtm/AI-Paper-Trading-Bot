#!/usr/bin/env python3
"""
NEWS BACKTEST — does overnight news predict the next trading day?

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Public market data and public news headlines (Alpaca news API) only; paper money.
No internal, personal or protected data (no SSNs/PHI). Research estimate, not advice.

THE IDEA BEING TESTED
  News that comes out after the close (4:00 PM) or before the open (9:25 AM)
  -> does that stock do well the next day, after the open?

HOW NEWS IS SCORED (no AI on purpose)
  A fixed word list (POSITIVE / NEGATIVE below) scores each headline + summary:
  +1 per positive phrase, -1 per negative phrase, summed per stock per night.
  No Gemini/LLM: a model trained on 2025-2026 text may already "know" what happened
  next (look-ahead bias). If plain words show an edge, a smarter reader can only add.

TWO WAYS TO BUY (both use the bot's real exits: -2% stop, profit locks,
trailing stop, everything sold at 15:45, 0.03% cost per side)
  open    : buy at 9:30 open
  confirm : buy at 10:00 only if the price is above the 9:30 open (it held its gains)

ALSO MEASURED
  gap = 9:30 open vs previous close: how much of the move already happened overnight,
  before any bot could buy.

Usage:
  python backtest_news.py            # last 365 days, Day-Trade-100's 100 stocks
  python backtest_news.py --fake     # offline self-test on random bars + random headlines
"""
import argparse, json, os, re, tempfile, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('STATE_DIR', tempfile.mkdtemp(prefix='bt_'))   # never touch the bots' real logs
import ai_trading_bot as bot                                          # noqa: E402
import meta_filter as mf                                              # noqa: E402
import backtest_meta as bm                                            # noqa: E402

ET = 'America/New_York'

POSITIVE = [
    r'\bbeats?\b', r'\btops? (estimates|expectations|forecasts?)\b', r'\bbetter[- ]than[- ]expected\b',
    r'\b(raises?|raised|lifts?|boosts?|hikes?) (its |full[- ]year |annual )?(guidance|outlook|forecast)\b',
    r'\brecord (revenue|sales|quarter|profit|earnings)\b', r'\bupgrade[sd]?\b',
    r'\b(raises?|raised|lifts?|boosts?) (its |the )?price target\b', r'\bprice target (raised|increased|lifted)\b',
    r'\b(fda )?approv(al|ed|es)\b', r'\b(wins?|won|awarded|secures?|lands?) (a |an |\$)?.{0,30}(contract|deal|order)\b',
    r'\b(buyback|share repurchase)\b', r'\b(raises?|increases?|hikes?) (its |quarterly )?dividend\b',
    r'\bto be acquired\b', r'\bstrong demand\b', r'\b(surges?|soars?|jumps?|rall(y|ies))\b',
]
NEGATIVE = [
    r'\bmiss(es|ed)?\b', r'\bfalls? short\b', r'\bworse[- ]than[- ]expected\b',
    r'\b(cuts?|lowers?|lowered|slashes?|reduces?) (its |full[- ]year |annual )?(guidance|outlook|forecast)\b',
    r'\bdowngrade[sd]?\b', r'\b(cuts?|lowers?|lowered|slashes?) (its |the )?price target\b',
    r'\bprice target (cut|lowered|reduced)\b', r'\b(lawsuit|sued|probe|investigation|subpoena)\b',
    r'\brecalls?\b', r'\bwarns?\b', r'\bweak (demand|guidance|outlook)\b', r'\b(delay(s|ed)?|halt(s|ed)?)\b',
    r'\b(plunges?|slumps?|tumbles?|sinks?|drops?)\b', r'\blayoffs?\b',
]
EARNINGS = re.compile(r'\b(earnings|quarter(ly)?|q[1-4]\b|results|revenue|eps|guidance)\b', re.I)
_POS = [re.compile(p, re.I) for p in POSITIVE]
_NEG = [re.compile(p, re.I) for p in NEGATIVE]


def score_text(text: str) -> int:
    """+1 for each positive phrase found, -1 for each negative phrase found."""
    return sum(1 for p in _POS if p.search(text)) - sum(1 for p in _NEG if p.search(text))


def fetch_news(symbols, start, end):
    """All Alpaca news for these symbols, month by month (Alpaca pages through 50 at a time)."""
    from alpaca.data.requests import NewsRequest
    out, t = [], start
    while t < end:
        t2 = min(t + timedelta(days=30), end)
        for attempt in range(4):
            try:
                res = bot.news_client.get_news(NewsRequest(symbols=','.join(symbols), start=t, end=t2,
                                                           limit=None, include_content=False))
                items = getattr(res, 'news', None) or (res.data.get('news', []) if hasattr(res, 'data') else [])
                for n in items:
                    out.append(dict(time=pd.Timestamp(n.created_at).tz_convert(ET), headline=n.headline or '',
                                    summary=n.summary or '', symbols=list(n.symbols or [])))
                break
            except Exception as e:
                print(f'  news {t.date()}: retry {attempt + 1} ({e})'); time.sleep(5 * (attempt + 1))
        print(f'  news up to {t2.date()}: {len(out)} articles', flush=True)
        t = t2
    return out


def fake_news(symbols, days):
    rng = np.random.default_rng(3)
    words = ['beats estimates', 'misses estimates', 'upgraded to buy', 'downgraded', 'raises guidance',
             'cuts guidance', 'announces conference', 'shares jump', 'probe launched', 'record revenue']
    out = []
    for d in days:
        for s in rng.choice(symbols, size=min(len(symbols), 3), replace=False):
            out.append(dict(time=pd.Timestamp(d) + pd.Timedelta(hours=7), headline=f'{s} {rng.choice(words)}',
                            summary='', symbols=[s]))
    return out


def day_rows(sym, bars, news_by_day):
    """One row per trading day: overnight news score + what each way of buying earned."""
    rth = bars[(mf.minutes_of(bars.index) >= mf.OPEN_MINUTE) & (mf.minutes_of(bars.index) < 16 * 60)]
    if rth.empty:
        return []
    o, c = rth['open'].to_numpy(float), rth['close'].to_numpy(float)
    mins, days = mf.minutes_of(rth.index), rth.index.normalize().to_numpy()
    rows, prev_close = [], None
    for day, idx in pd.Series(np.arange(len(rth)), index=rth.index.normalize()).groupby(level=0):
        ii = idx.to_numpy()
        i_open = ii[mins[ii] == mf.OPEN_MINUTE]
        i_10 = ii[mins[ii] == 10 * 60]
        if len(i_open) == 0 or prev_close is None:
            prev_close = c[ii[-1]]; continue
        i0 = i_open[0]
        art = news_by_day.get((sym, pd.Timestamp(day)), [])
        sc = sum(a[0] for a in art)
        r = dict(sym=sym, day=pd.Timestamp(day).date(), n_news=len(art), score=sc,
                 earnings=int(any(a[1] for a in art)), gap=o[i0] / prev_close - 1)
        ex, why, j = mf.simulate_exit(o, c, mins, days, i0)
        r.update(open_ret=mf.net_return(o[i0], ex), open_exit=why)
        if len(i_10) and o[i_10[0]] > o[i0]:
            ex, why, j = mf.simulate_exit(o, c, mins, days, i_10[0])
            r.update(confirm=1, confirm_ret=mf.net_return(o[i_10[0]], ex), confirm_exit=why)
        else:
            r.update(confirm=0, confirm_ret=np.nan, confirm_exit='')
        rows.append(r)
        prev_close = c[ii[-1]]
    return rows


def bucket(s):
    return 'strong positive (+2 or more)' if s >= 2 else 'positive (+1)' if s == 1 else \
           'neutral news (0)' if s == 0 else 'negative (-1 or less)'


def summarize(r):
    r = r.dropna()
    if not len(r):
        return dict(n=0)
    w, l = r[r > 0], r[r <= 0]
    return dict(n=int(len(r)), hit_rate=float((r > 0).mean()), avg_ret=float(r.mean()),
                profit_factor=float(w.sum() / -l.sum()) if l.sum() < 0 else None)


def assign_news(news, trade_days, known_syms):
    """Overnight window for trading day D = previous trading day 16:00 -> D 09:25 (ET).
    Returns {(symbol, D): [(score, is_earnings), ...]}. News during market hours is ignored."""
    window_of = {}
    for prev, d in zip(trade_days, trade_days[1:]):
        window_of[d] = (prev + pd.Timedelta(hours=16), d + pd.Timedelta(hours=9, minutes=25))
    starts = pd.DatetimeIndex([w[0] for w in window_of.values()])
    keys = list(window_of)
    news_by_day = {}
    for n in news:
        k = starts.searchsorted(n['time'], side='right') - 1
        if k < 0:
            continue
        d = keys[k]
        if not (window_of[d][0] <= n['time'] <= window_of[d][1]):
            continue                       # published during market hours: not overnight news
        text = f"{n['headline']}. {n['summary']}"
        sc, earn = score_text(text), bool(EARNINGS.search(text))
        for sym in n['symbols']:
            if sym in known_syms:
                news_by_day.setdefault((sym, d), []).append((sc, earn))
    return news_by_day


RULE_TEXT = 'strong good news (score +2 or more) AND opened more than 1% above the previous close -> buy at 9:30; top 3 a day'


def rule_rows(D):
    """The fixed rule, decided in advance (found on Oct 2025-Oct 2026, never tuned on earlier data)."""
    return D[(D['score'] >= 2) & (D['gap'] > 0.01)]


def daily_drawdown(trades, start=100_000.0):
    """Worst drop from a peak on END-OF-DAY balances. Several trades close at the same 15:45 bell, so an
    intraday order would be arbitrary (and change the answer); daily balances don't depend on it."""
    if not trades:
        return None
    d = pd.DataFrame(trades).groupby('day', sort=True)['pnl'].sum().to_numpy()
    c = np.r_[start, start + np.cumsum(d)]
    return float((c / np.maximum.accumulate(c) - 1).min())


def account_open(R, top=3):
    """$100k: each day buy up to `top` rule stocks at 9:30 (highest score, then most articles), 12% of equity each."""
    P = R.sort_values(['day', 'score', 'n_news'], ascending=[True, False, False]).groupby('day').head(top)
    eq, trades = 100_000.0, []
    for day, g in P.groupby('day'):
        size = eq * 0.12
        for _, r in g.iterrows():
            trades.append(dict(sym=r['sym'], day=str(day), ret=float(r['open_ret']), size=size,
                               pnl=float(size * r['open_ret']), reason=r['open_exit'], exit_t=str(day)))
        eq += float((size * g['open_ret']).sum())
    return trades


def rule_report(R, spy, top=3, seed=0):
    """Per-trade stats with a bootstrap 90% range, months positive, and the $100k account vs SPY."""
    out = dict(calls=summarize(R['open_ret']))
    x = R['open_ret'].dropna().to_numpy()
    if len(x) > 2:
        rng = np.random.default_rng(seed)
        boots = [rng.choice(x, len(x)).mean() for _ in range(5000)]
        out.update(avg_90pct_range=[float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
                   t_stat=float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))))
    m = R.groupby('month')['open_ret'].mean()
    out['months_positive'] = f'{int((m > 0).sum())} of {len(m)}'
    out['by_month'] = {k: [int(c), float(v)] for k, (c, v) in R.groupby('month')['open_ret'].agg(['count', 'mean']).iterrows()}
    t = account_open(R, top)
    out['account'] = bm.stats(t)
    if t:
        out['account']['max_drawdown'] = daily_drawdown(t)
    if len(R):
        d0, d1 = pd.Timestamp(R['day'].min()).tz_localize(ET), pd.Timestamp(R['day'].max()).tz_localize(ET) + pd.Timedelta(days=1)
        sp = spy[(spy.index >= d0) & (spy.index < d1)]
        sp = sp[(mf.minutes_of(sp.index) >= mf.OPEN_MINUTE) & (mf.minutes_of(sp.index) < 16 * 60)]
        out['spy_buy_hold'] = float(sp['close'].iloc[-1] / sp['open'].iloc[0] - 1) if len(sp) else None
    return out, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=365)
    ap.add_argument('--top', type=int, default=3, help='account: buys per day (best news first)')
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--split', default=None,
                    help='YYYY-MM-DD: report the FIXED RULE separately before (unseen) and from (seen) this date')
    a = ap.parse_args()
    t0 = time.time()
    syms = bm.dt100_universe()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=a.days + 7)
    get = (lambda s: bm.fake_bars(s, a.days + 7)) if a.fake else (lambda s: bot.fetch_bars(s, days=a.days + 7))

    bars = {}
    for s in syms + ['SPY']:
        try:
            bars[s] = get(s)
        except Exception as e:
            print(f'  skip {s}: {e}')
    print(f'Loaded bars for {len(bars)} symbols in {time.time() - t0:.0f}s', flush=True)
    trade_days = sorted({d for df in bars.values() for d in df.index.normalize().unique()})
    news = fake_news(syms, trade_days) if a.fake else fetch_news(syms, start, end)
    print(f'{len(news)} articles', flush=True)

    news_by_day = assign_news(news, trade_days, set(bars))

    rows = []
    for s in syms:
        if s in bars:
            rows += day_rows(s, bars[s], news_by_day)
    D = pd.DataFrame(rows)
    D['bucket'] = np.where(D['n_news'] == 0, 'no overnight news', D['score'].map(bucket))
    D['month'] = pd.to_datetime(D['day']).dt.to_period('M').astype(str)

    order = ['strong positive (+2 or more)', 'positive (+1)', 'neutral news (0)', 'negative (-1 or less)', 'no overnight news']
    table = []
    for b in order:
        g = D[D['bucket'] == b]
        months = g.groupby('month')['open_ret'].mean()
        table.append(dict(bucket=b, stock_days=int(len(g)), avg_gap=float(g['gap'].mean()) if len(g) else None,
                          buy_at_open=summarize(g['open_ret']), buy_at_10_if_holding=summarize(g['confirm_ret']),
                          months_positive_open=f'{int((months > 0).sum())} of {len(months)}'))
    earn = D[(D['earnings'] == 1) & (D['score'] >= 1)]
    earn_tbl = dict(stock_days=int(len(earn)), avg_gap=float(earn['gap'].mean()) if len(earn) else None,
                    buy_at_open=summarize(earn['open_ret']), buy_at_10_if_holding=summarize(earn['confirm_ret']))

    # $100k account: each day buy the top-N strongest positive news stocks that held their gains (10:00 entry)
    P = D[(D['score'] >= 2) & (D['confirm'] == 1)].sort_values(['day', 'score', 'n_news'], ascending=[True, False, False])
    P = P.groupby('day').head(a.top)
    eq, trades = 100_000.0, []
    for day, g in P.groupby('day'):
        size = eq * 0.12
        pnl = float((size * g['confirm_ret']).sum())
        for _, r in g.iterrows():
            trades.append(dict(sym=r['sym'], day=str(day), ret=float(r['confirm_ret']), size=size,
                               pnl=float(size * r['confirm_ret']), reason=r['confirm_exit'], exit_t=str(day)))
        eq += pnl
    acct = bm.stats(trades)
    if trades:
        acct['max_drawdown'] = daily_drawdown(trades)
    spy = bars['SPY']
    spy_rth = spy[(mf.minutes_of(spy.index) >= mf.OPEN_MINUTE) & (mf.minutes_of(spy.index) < 16 * 60)]
    first_day = pd.Timestamp(D['day'].min()).tz_localize(ET)
    spy_rth = spy_rth[spy_rth.index >= first_day]
    spy_ret = float(spy_rth['close'].iloc[-1] / spy_rth['open'].iloc[0] - 1)

    fixed = {}
    if a.split:
        cut = pd.Timestamp(a.split).date()
        R = rule_rows(D)
        for name, part in [('unseen (before ' + a.split + ')', R[R['day'] < cut]), ('seen (from ' + a.split + ')', R[R['day'] >= cut]),
                           ('all', R)]:
            rr, tt = rule_report(part, bars['SPY'], a.top)
            base = D[(D['n_news'] == 0) & ((D['day'] < cut) if name.startswith('unseen') else (D['day'] >= cut) if name.startswith('seen') else True)]
            rr['no_news_baseline'] = summarize(base['open_ret'])
            fixed[name] = rr
            pd.DataFrame(tt).to_csv(Path(a.out or 'state/backtest_news.json').with_name(
                'backtest_news_rule_' + name.split()[0] + '_trades.csv'), index=False)

    report = dict(
        fixed_rule=dict(rule=RULE_TEXT, results=fixed) if fixed else None,
        fake_data=a.fake, generated=pd.Timestamp.now(tz=ET).isoformat(timespec='seconds'),
        period=[str(D['day'].min()), str(D['day'].max())], symbols=len(bars) - 1, articles=len(news),
        overnight_news_stock_days=int((D['n_news'] > 0).sum()), stock_days=int(len(D)),
        buckets=table, positive_earnings_news=earn_tbl,
        account=dict(rule=f'each day buy up to {a.top} stocks with news score >= +2 that are above their 9:30 open at 10:00; '
                          '12% of equity each; live exit rules', **acct),
        spy_buy_hold_return=spy_ret, runtime_sec=round(time.time() - t0),
        disclaimer='Independent personal project, not tied to any employer or agency. Public market data and news only, '
                   'paper money. Backtest estimate; not financial advice.',
    )
    out = Path(a.out or f'state/backtest_news{"_fake" if a.fake else ""}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    D.to_csv(out.with_name(out.stem + '_days.csv'), index=False)
    pd.DataFrame(trades).to_csv(out.with_name(out.stem + '_trades.csv'), index=False)

    print(f"\n=== NEWS BACKTEST {report['period']} | {report['articles']} articles | SPY {spy_ret:+.2%}")
    for t in table:
        f = lambda s: f"{s['n']:5d} hit {s['hit_rate']:.1%} avg {s['avg_ret'] * 100:+.3f}%" if s.get('n') else '    0'
        print(f"{t['bucket']:30s} days {t['stock_days']:6d} gap {0 if t['avg_gap'] is None else t['avg_gap'] * 100:+.2f}% | "
              f"open: {f(t['buy_at_open'])} | 10:00-if-holding: {f(t['buy_at_10_if_holding'])} | months+ {t['months_positive_open']}")
    if acct.get('trades'):
        print(f"Account: {acct['trades']} trades, hit {acct['hit_rate']:.1%}, ${acct['profit_per_trade']:.2f}/trade, "
              f"total {acct['total_return']:+.2%}, max drop {acct['max_drawdown']:.2%}")
    for name, rr in fixed.items():
        c, ac = rr['calls'], rr['account']
        print(f"FIXED RULE {name:28s}: {c.get('n', 0)} stock-days, hit {c.get('hit_rate', 0):.1%}, avg {c.get('avg_ret', 0) * 100:+.3f}% "
              f"(90% range {rr.get('avg_90pct_range', [0, 0])[0] * 100:+.3f}..{rr.get('avg_90pct_range', [0, 0])[1] * 100:+.3f}%), "
              f"months+ {rr['months_positive']} | account {ac.get('trades', 0)} trades {ac.get('total_return', 0):+.2%} "
              f"DD {ac.get('max_drawdown', 0):.2%} | SPY {rr.get('spy_buy_hold') or 0:+.2%} | no-news avg {rr['no_news_baseline'].get('avg_ret', 0) * 100:+.3f}%")
    print(f'Saved {out} ({report["runtime_sec"]}s)')


if __name__ == '__main__':
    main()
