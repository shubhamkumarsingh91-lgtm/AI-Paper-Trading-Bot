#!/usr/bin/env python3
"""One-time backfill (Oct 22, 2026): add precision-at-threshold, base rate and lift to past
retrains in learning_history.json, read from the bot's own log lines. Rows get
'backfilled': true because base_rate here is the whole training set's rate, not the
validation slice's. Safe to re-run (skips rows that already have the field).

Independent personal project, not tied to any employer or agency. Public market data only.
"""
import json, re, sys
from datetime import datetime

def run(d):
    hist = json.load(open(f'{d}/learning_history.json'))
    cal, bal = [], []
    for line in open(f'{d}/ai_bot.log', errors='ignore'):
        m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*Calibrated decision threshold: ([\d.]+) \(precision ([\d.]+)% on (\d+) signals', line)
        if m: cal.append((datetime.fromisoformat(m[1]), float(m[3]) / 100, int(m[4])))
        m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*Class balance: ([\d,]+) BUY / ([\d,]+) HOLD', line)
        if m:
            b, h = int(m[2].replace(',', '')), int(m[3].replace(',', ''))
            bal.append((datetime.fromisoformat(m[1]), b / (b + h)))
    n = 0
    for r in hist:
        if 'precision_at_threshold' in r: continue
        t = datetime.fromisoformat(r['timestamp'][:19])
        c = [x for x in cal if abs((x[0] - t).total_seconds()) < 300]
        b = [x for x in bal if 0 <= (t - x[0]).total_seconds() < 3600]
        if not c or not b: continue
        p, sig = min(c, key=lambda x: abs((x[0] - t).total_seconds()))[1:]
        base = max(b, key=lambda x: x[0])[1]
        r.update(precision_at_threshold=round(p, 4), signals_at_threshold=sig,
                 base_rate=round(base, 4), lift=round(p / base, 3), backfilled=True)
        n += 1
    json.dump(hist, open(f'{d}/learning_history.json', 'w'), indent=2)
    print(f'{d}: backfilled {n} of {len(hist)} retrains')

for d in sys.argv[1:] or ['state', 'state_dt100']:
    run(d)
