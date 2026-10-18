#!/usr/bin/env python3
# VERSION 3: SPY + SIDEWAYS ONLY. Hold SPY in BULL and BEAR regimes;
# trade the 10-stock oversold playbook only when the market is SIDEWAYS.
# Uses point-in-time S&P 500 membership (same as version 2).
#
# DISCLAIMER: Independent personal project. Not tied to any employer or agency.
# Uses public market data only. No internal or personal data (no SSNs / PHI).
# Educational backtest, not investment advice.

import io, sys, subprocess
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
    import yfinance as yf
import numpy as np
import pandas as pd
import requests

START = "2014-01-01"      # extra year so 200-day averages exist by 2015
TEST_START = "2015-01-01"
COST = 0.0005             # 0.05% per dollar traded
TOP_N = 10
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")

# ---------- 1. Point-in-time membership ----------
raw = pd.read_csv(io.StringIO(requests.get(MEMBERS_URL, timeout=60).text))
raw["date"] = pd.to_datetime(raw["date"])
raw = raw[raw["date"] >= "2012-01-01"].sort_values("date")
def clean(t):
    return t.strip().replace(".", "-").split("-1")[0].split("-2")[0] if t else t
raw["set"] = raw["tickers"].apply(lambda s: {clean(t) for t in str(s).split(",") if t.strip()})
all_tickers = sorted(set().union(*raw["set"]))
print(f"Membership history: {len(raw)} snapshots, {raw['date'].min().date()} -> {raw['date'].max().date()}")
print(f"Stocks that were EVER in the S&P 500 in this period: {len(all_tickers)}")

# ---------- 2. Prices ----------
print("Downloading prices (several minutes)...")
px = yf.download(all_tickers + ["SPY"], start=START, auto_adjust=True,
                 progress=False, threads=True)["Close"]
px = px.dropna(axis=1, how="all")
print(f"Stocks with usable Yahoo data: {px.shape[1]-1} of {len(all_tickers)} "
      f"({(px.shape[1]-1)/len(all_tickers):.0%} coverage)")
spy = px["SPY"]
stocks = px.drop(columns=["SPY"])
rets = stocks.pct_change()
spy_ret = spy.pct_change().fillna(0)

# membership mask: on each day, which tickers were members
snap = raw.set_index("date")["set"]
member_on = snap.reindex(px.index, method="ffill")

# ---------- 3. Regime ----------
ma200, ma50 = spy.rolling(200).mean(), spy.rolling(50).mean()
rising = ma200 > ma200.shift(20)
regime = pd.Series("SIDEWAYS", index=spy.index)
regime[(spy > ma200) & rising & (ma50 > ma200)] = "BULL"
regime[(spy < ma200) & ~rising] = "BEAR"

# ---------- 4. Sideways picks (oversold but still in uptrend) ----------
stock_ma200 = stocks.rolling(200).mean()
drop3 = stocks / stocks.shift(3) - 1
vol = rets.rolling(20).std()

dates = px.index[px.index >= TEST_START]
weights = {}
current = pd.Series({"SPY": 1.0})
prev_reg = None
for i, d in enumerate(dates):
    reg = regime[d]
    rebalance = (reg != prev_reg) or (reg == "SIDEWAYS" and d.weekday() == 0)
    if rebalance:
        if reg == "SIDEWAYS":
            mem = member_on.get(d) or set()
            ok = [t for t in stocks.columns if t in mem
                  and stocks.at[d, t] > stock_ma200.at[d, t]
                  and pd.notna(drop3.at[d, t]) and pd.notna(vol.at[d, t]) and vol.at[d, t] > 0]
            picks = drop3.loc[d, ok].nsmallest(TOP_N).index
            if len(picks):
                iv = 1 / vol.loc[d, picks]
                current = iv / iv.sum()
            else:
                current = pd.Series({"SPY": 1.0})
        else:
            current = pd.Series({"SPY": 1.0})
    weights[d] = current
    prev_reg = reg

# ---------- 5. Daily returns (weights at close, earned next day) ----------
port = []
prev_w = pd.Series(dtype=float)
for i in range(len(dates) - 1):
    d, nxt = dates[i], dates[i + 1]
    w = weights[d]
    r = 0.0
    for t, wt in w.items():
        rr = spy_ret[nxt] if t == "SPY" else rets.at[nxt, t]
        r += wt * (0 if pd.isna(rr) else rr)
    turnover = w.subtract(prev_w, fill_value=0).abs().sum()
    port.append((nxt, r - turnover * COST))
    prev_w = w
bot = pd.Series(dict(port))
spyr = spy_ret.reindex(bot.index)

def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return eq.iloc[-1] - 1, cagr, dd, sh

print("\n=== WHOLE PERIOD (since 2015) — SPY + SIDEWAYS ONLY ===")
print(f"{'':16}{'Total':>8}{'Per year':>10}{'Max drop':>10}{'Sharpe':>8}")
for name, r in [("SPY+sideways", bot), ("SPY hold", spyr)]:
    t, c, dd, sh = stats(r)
    print(f"{name:16}{t:+8.0%}{c:+10.1%}{dd:+10.1%}{sh:8.2f}")
print("(Version 2 full regime bot showed +8.9%/yr, Sharpe 0.45 — compare.)")

print("\n=== SEPARATE ERAS ===")
for lbl, a, b in [("2015-2019", "2015", "2019"), ("2020-2022", "2020", "2022"), ("2023-now", "2023", "2030")]:
    sb, ss = stats(bot[a:b]), stats(spyr[a:b])
    print(f"{lbl}: bot {sb[1]:+.1%}/yr (max drop {sb[2]:+.1%})   SPY {ss[1]:+.1%}/yr (max drop {ss[2]:+.1%})")

side = regime.reindex(bot.index) == "SIDEWAYS"
ex = (bot - spyr)[side]
print("\n=== THE SIDEWAYS EDGE ITSELF ===")
print(f"Sideways days: {side.sum()} ({side.mean():.0%} of days)")
print(f"Avg daily excess vs SPY on those days: {ex.mean():+.3%}  | days beating SPY: {(ex > 0).mean():.0%}")
print(f"Annualised excess while sideways: {ex.mean()*252:+.1%}")
# split-half check: is the edge stable or luck?
h = len(ex) // 2
print(f"First half of sideways days excess/yr: {ex.iloc[:h].mean()*252:+.1%} | second half: {ex.iloc[h:].mean()*252:+.1%}")

print("\n=== YEAR BY YEAR ===")
yb = pd.DataFrame({"bot": (1 + bot).groupby(bot.index.year).prod() - 1,
                   "SPY": (1 + spyr).groupby(spyr.index.year).prod() - 1})
yb["beat_SPY"] = yb["bot"] > yb["SPY"]
print(yb.round(3).to_string())
print(f"\nCurrent regime (latest close): {regime.iloc[-1]}")
print("NOTE: delisted stocks missing from Yahoo -> results are an UPPER bound.")
pd.DataFrame({"bot": bot, "spy": spyr, "regime": regime.reindex(bot.index)}).to_csv("state/regime_backtest_daily.csv")
print("Saved -> state/regime_backtest_daily.csv")
