#!/usr/bin/env python3
# REGIME-SIDEWAYS PAPER TRADER (live version of backtest_regime.py v3)
# BULL or BEAR -> hold 100% SPY.  SIDEWAYS -> 10 oversold S&P 500 stocks
# still above their own 200-day average, inverse-volatility weighted,
# rebalanced weekly (Mondays) or immediately when the regime changes.
# Also keeps a SHADOW LOG of what the bull and bear playbooks would pick,
# so they can be re-evaluated later without risking money.
#
# DISCLAIMER: Independent personal project. Not tied to any employer or agency.
# Public market data only. No internal or personal data (no SSNs / PHI).
# Paper trading only. Educational, not investment advice.

import os, sys, json, io, datetime as dt, subprocess
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
    import yfinance as yf
import pandas as pd
import requests

KEY = os.environ["REGIME_ALPACA_API_KEY"]
SECRET = os.environ["REGIME_ALPACA_SECRET_KEY"]
BASE = "https://paper-api.alpaca.markets"          # PAPER ONLY - hardcoded on purpose
H = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}
STATE_DIR = "state"
STATE_FILE = f"{STATE_DIR}/regime_paper_state.json"
SCORE_FILE = f"{STATE_DIR}/regime_scorecard.csv"
SHADOW_FILE = f"{STATE_DIR}/regime_shadow_log.csv"
TOP_N, MIN_TRADE = 10, 25.0                      # skip trades under $25
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")

def api(method, path, **kw):
    r = requests.request(method, BASE + path, headers=H, timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.text else {}

# ---------- 0. Only run when the market is open ----------
clock = api("GET", "/v2/clock")
if not clock["is_open"] and not DRY_RUN:
    print("Market closed - nothing to do."); sys.exit(0)
os.makedirs(STATE_DIR, exist_ok=True)
state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
today = dt.date.today()

# ---------- 1. Current S&P 500 members (latest snapshot) ----------
m = pd.read_csv(io.StringIO(requests.get(MEMBERS_URL, timeout=60).text))
members = sorted({t.strip().replace(".", "-") for t in m.iloc[-1]["tickers"].split(",")})

# ---------- 2. Prices (today's partial bar included ~15 min before close) ----------
px = yf.download(members + ["SPY", "TLT", "GLD", "SHY"], period="400d",
                 auto_adjust=True, progress=False, threads=True)["Close"].ffill()
spy = px["SPY"]
ma200, ma50 = spy.rolling(200).mean(), spy.rolling(50).mean()
rising = ma200.iloc[-1] > ma200.iloc[-21]
if spy.iloc[-1] > ma200.iloc[-1] and rising and ma50.iloc[-1] > ma200.iloc[-1]:
    regime = "BULL"
elif spy.iloc[-1] < ma200.iloc[-1] and not rising:
    regime = "BEAR"
else:
    regime = "SIDEWAYS"
print(f"{today}  regime: {regime}  (last run: {state.get('regime')})")

stocks = px[[c for c in members if c in px.columns]].dropna(axis=1, thresh=250)
above = stocks.iloc[-1] > stocks.rolling(200).mean().iloc[-1]
vol = stocks.pct_change().rolling(20).std().iloc[-1]

def sideways_picks():
    drop3 = (stocks.iloc[-1] / stocks.iloc[-4] - 1)[above & (vol > 0)]
    picks = drop3.nsmallest(TOP_N).index
    iv = 1 / vol[picks]
    return (iv / iv.sum()).round(4).to_dict()

# ---------- 3. Shadow log: what every playbook would pick today ----------
mom = (stocks.iloc[-22] / stocks.iloc[-127] - 1)[above]       # 6-month momentum, skip last month
bull_pick = list(mom.nlargest(TOP_N).index)
safe = px[["TLT", "GLD", "SHY"]]
bear_pick = (safe.iloc[-1] / safe.iloc[-64] - 1).idxmax()
side = sideways_picks()
row = pd.DataFrame([{"date": today, "regime": regime, "bull_picks": " ".join(bull_pick),
                     "bear_pick": bear_pick, "sideways_picks": " ".join(side)}])
row.to_csv(SHADOW_FILE, mode="a", header=not os.path.exists(SHADOW_FILE), index=False)

# ---------- 4. Decide whether to rebalance ----------
changed = regime != state.get("regime")
monday = today.weekday() == 0
if regime == "SIDEWAYS":
    target = side
    rebalance = changed or monday
else:
    target = {"SPY": 1.0}
    rebalance = changed or not state.get("initialised")

acct = api("GET", "/v2/account")
equity = float(acct["equity"])
positions = {p["symbol"]: float(p["market_value"]) for p in api("GET", "/v2/positions")}

if rebalance:
    print("Rebalancing to:", target)
    want = {s: w * equity * 0.98 for s, w in target.items()}     # keep 2% cash buffer
    orders = []
    for s in set(positions) | set(want):
        diff = want.get(s, 0) - positions.get(s, 0)
        if s not in want and s in positions:
            orders.append(("close", s, None))
        elif abs(diff) >= MIN_TRADE:
            orders.append(("sell" if diff < 0 else "buy", s, abs(round(diff, 2))))
    orders.sort(key=lambda o: o[0] == "buy")                     # sells first, free up cash
    for side_, s, amt in orders:
        print(f"  {side_:5} {s:6} {'' if amt is None else f'${amt:,.2f}'}")
        if DRY_RUN:
            continue
        try:
            if side_ == "close":
                api("DELETE", f"/v2/positions/{s}")
            else:
                api("POST", "/v2/orders", json={"symbol": s, "notional": str(amt), "side": side_,
                                                 "type": "market", "time_in_force": "day"})
        except requests.HTTPError as e:
            print(f"  !! order failed for {s}: {e.response.text}")
else:
    print("No rebalance today - holding.")

# ---------- 5. Scorecard vs SPY ----------
start_eq = state.get("start_equity", equity)
start_spy = state.get("start_spy", float(spy.iloc[-1]))
score = pd.DataFrame([{"date": today, "regime": regime, "equity": round(equity, 2),
                       "bot_return": round(equity / start_eq - 1, 4),
                       "spy_return": round(float(spy.iloc[-1]) / start_spy - 1, 4)}])
score.to_csv(SCORE_FILE, mode="a", header=not os.path.exists(SCORE_FILE), index=False)
print(f"Since start: bot {equity/start_eq-1:+.2%}  vs  SPY {spy.iloc[-1]/start_spy-1:+.2%}")

if not DRY_RUN:
    state.update({"regime": regime, "initialised": True, "start_equity": start_eq,
                  "start_spy": start_spy, "last_run": str(today), "target": target})
    json.dump(state, open(STATE_FILE, "w"), indent=2)
