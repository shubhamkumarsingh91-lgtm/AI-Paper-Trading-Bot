#!/usr/bin/env python3
"""
events.py — turns a bot's own records into a short, plain-English event list for Agent HQ
(the "town" page that shows what each AI agent is doing).

DISCLAIMER: Independent personal project. Not tied to any employer or agency. Paper trading
(fake money) and public market data only. No internal or personal data (no SSNs/PHI).

Nothing here is invented: every event comes from a line the bot logged, a trade it saved, or a
balance it recorded. Each bot writes its own file, so bots never edit the same file at once:
  python events.py day    --bot b1 --dir state        --out state/events_b1.json
  python events.py day    --bot b3 --dir state_dt100  --out state_dt100/events.json
  python events.py regime --bot b2 --dir state        --out state/events_regime.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

KEEP = 160                    # events kept per bot
ET = ZoneInfo("America/New_York")
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[\w+\] ?(.*)$")
REASONS = {"EOD": "end-of-day close", "STOP_LOSS": "stop-loss", "PROFIT_LOCK": "locking in profit",
           "TRAILING_STOP": "trailing stop", "TAKE_PROFIT": "take-profit", "TIME_STOP": "time limit"}


def _iso(stamp: str) -> str:
    # GitHub runners log in UTC
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _tail(path: Path, max_bytes: int = 1_500_000) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read().decode("utf-8", "replace")
    lines = data.splitlines()
    return lines[1:] if size > max_bytes else lines


def runs_from_log(lines: list[str]) -> list[dict]:
    """Group log lines into runs: 'Run mode: X' ... 'Run complete'."""
    runs, cur, stamp = [], None, None
    for raw in lines:
        m = TS.match(raw)
        if m:
            stamp, msg = m.group(1), m.group(2).strip()
        elif stamp and raw.strip():
            msg = raw.strip()          # continuation of a multi-line log message (e.g. the "── Scan #" header)
        else:
            continue
        mm = re.match(r"Run mode: (\w+)", msg)
        if mm:
            cur = {"mode": mm.group(1).upper(), "t": _iso(stamp), "lines": []}
            runs.append(cur)
            continue
        if cur is not None:
            cur["lines"].append((_iso(stamp), msg))
    return runs


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"


def day_events(bot: str, run: dict, threshold_default: float = 0.60) -> list[dict]:
    """Events for one run of a day-trading bot (Bot 1 or Day-Trade-100)."""
    ev = []
    L = run["lines"]
    text = "\n".join(m for _, m in L)
    add = lambda t, place, kind, s, **kw: ev.append({"t": t, "bot": bot, "place": place, "kind": kind, "text": s, **kw})

    if run["mode"] == "RETRAIN":
        wf = re.search(r"Walk-forward:\s+([\d.]+)%", text)
        n = re.search(r"Total: ([\d,]+) samples", text)
        trend = re.search(r"Trend vs last retrain: precision ([+-][\d.]+)pp, walk-forward ([+-][\d.]+)pp", text)
        s = "Retrained its AI overnight"
        if n:
            s += f" on {n.group(1)} price bars"
        if wf:
            s += f". When it says “buy”, it was right {float(wf.group(1)):.0f}% of the time in testing"
            if trend:
                d = float(trend.group(2))
                s += f" ({'up' if d >= 0 else 'down'} {abs(d):.1f} points from last time)"
        add(run["t"], "lab", "retrain", s + ".")
        return ev

    if run["mode"] == "PREMARKET":
        bits = []
        if "earnings calendar" in text.lower():
            bits.append("checked which companies report earnings soon")
        if "News re-ranking off" in text or "news" in text.lower():
            bits.append("read the morning market news")
        vix = re.search(r"VIX fear gauge: ([\d.]+)", text)
        if vix:
            bits.append(f"noted the fear index (VIX) at {float(vix.group(1)):.1f}")
        add(run["t"], "newsroom", "premarket", "Morning prep before the open: " + (", ".join(bits) or "loaded its model and watchlist") + ".")
        return ev

    if run["mode"] == "EOD":
        sells = [(t, m) for t, m in L if m.startswith("🔴 SELL")]
        if sells:
            add(sells[0][0], "broker", "eod", f"Closing time: sold its {len(sells)} open position{'s' if len(sells) != 1 else ''} so it holds no stock overnight.")
        else:
            add(run["t"], "vault", "eod", "Market closed: no open positions to sell. Recorded the day’s balance.")
        for t, m in sells:
            ev.extend(_sell(bot, t, m))
        return ev

    if run["mode"] != "SCAN":
        return ev

    head = next(((t, m) for t, m in L if m.startswith("── Scan #")), None)
    if not head:
        return ev
    t0 = head[0]
    hm = re.search(r"\$([\d,]+\.?\d*) equity \| (\d+)/(\d+) positions", head[1])
    eq = float(hm.group(1).replace(",", "")) if hm else None
    held, maxpos = (int(hm.group(2)), int(hm.group(3))) if hm else (None, None)
    thr = threshold_default
    tm = re.search(r"threshold=([\d.]+)", text)
    if tm:
        thr = float(tm.group(1))
    checked = len({m.split(":")[0].strip() for _, m in L if re.match(r"^[A-Z.]{1,6}\s*: (HOLD|BUY|SELL)\b", m)})
    liked = []
    for _, m in L:
        mm = re.match(r"^\[([A-Z.]+)\] ML BUY .*ens=([\d.]+)", m)
        if mm:
            liked.append((mm.group(1), float(mm.group(2))))
    liked.sort(key=lambda x: -x[1])
    s = f"Scanned {checked or 'its'} stock{'s' if checked != 1 else ''}"
    if held is not None:
        s += f" (holding {held} of {maxpos} positions" + (f", balance {_money(eq)})" if eq else ")")
    strong = [(sym, p) for sym, p in liked if p >= thr]
    passed = []                                   # final decisions after every check (vetoes, TradingView...)
    for _, m in L:
        mm = re.match(r"^([A-Z.]{1,6})\s*: BUY\s+score=\S+ ml=(\d+)%", m)
        if mm:
            passed.append((mm.group(1), int(mm.group(2)) / 100))
    passed.sort(key=lambda x: -x[1])
    if passed:
        top = ", ".join(f"{sym} {p:.0%}" for sym, p in passed[:3])
        s += f". {len(passed)} passed all its checks: {top}" + ("…" if len(passed) > 3 else "")
    elif strong:
        s += f". The AI liked {len(strong)} above its {thr:.0%} bar, but other checks said no"
    elif liked:
        s += f". None cleared its {thr:.0%} confidence bar (closest: {liked[0][0]} {liked[0][1]:.0%})"
    add(t0, "floor", "scan", s + ".", count=checked)

    did = False
    for t, m in L:
        rc = re.match(r"📉 REGIME CHANGE: (\w+) → (\w+) \(SPY \$([\d.,]+) vs SMA50 \$([\d.,]+)\)", m) or \
             re.match(r"📈 REGIME CHANGE: (\w+) → (\w+) \(SPY \$([\d.,]+) vs SMA50 \$([\d.,]+)\)", m)
        if rc:
            add(t, "tower", "regime", f"Noticed the market turned {rc.group(2)}: SPY (${rc.group(3)}) is {'below' if rc.group(2) == 'BEAR' else 'above'} its 50-day average (${rc.group(4)}).")
        bs = re.match(r"(\S+): 🐻 BEAR MARKET skip \(ML (\d+)% < (\d+)% required\)", m)
        if bs:
            near = bs.group(2) == bs.group(3)          # the log rounds, e.g. 67.6% shows as 68%
            add(t, "floor", "skip", f"Picked {bs.group(1)} ({'just under ' if near else ''}{bs.group(2)}% confident) but skipped it: in a falling market it needs {bs.group(3)}% before buying.", symbol=bs.group(1))
            did = True
        eb = re.match(r"(\S+)\s*: 📅 EARNINGS BLACKOUT \((\d+)d away\)", m)
        if eb:
            add(t, "floor", "skip", f"Skipped {eb.group(1)}: it reports earnings in {eb.group(2)} day{'s' if eb.group(2) != '1' else ''}, too risky for a day trade.", symbol=eb.group(1))
        if m.startswith("🟢 BUY"):
            b = re.search(r"BUY\s+(\d+)×(\S+)\s+@\s+\$\s*([\d.,]+).*?ML:(\d+)%", m)
            if b:
                q, sym, px, ml = int(b.group(1)), b.group(2), float(b.group(3).replace(",", "")), b.group(4)
                add(t, "broker", "buy", f"Bought {q} {sym} at ${px:,.2f} (about {_money(q * px)}). The AI was {ml}% confident.", symbol=sym)
                did = True
        elif m.startswith("🔴 SELL"):
            ev.extend(_sell(bot, t, m))
            did = True
        elif m.startswith("👁"):
            w = re.search(r"👁\s+(\S+) → dip watch", m)
            if w:
                add(t, "floor", "watch", f"Put {w.group(1)} on a dip watch: it will buy back only if the price turns up again.", symbol=w.group(1))
    if did:
        return ev
    if "Max positions" in text:
        add(t0, "floor", "wait", f"All {maxpos} position slots are full, so it is only watching what it already holds.")
        return ev
    if "No actionable signal" in text:
        # explain why it didn't buy: the AI liked something but another check said no
        why = []
        for _, m in L:
            mm = re.match(r"^([A-Z.]{1,6})\s*: HOLD .*?ml=(\d+)% tv=(\S+)", m)
            if mm and int(mm.group(2)) >= thr * 100:
                sym, ml, tv = mm.group(1), int(mm.group(2)), mm.group(3)
                why.append(f"{sym} ({ml}%) but TradingView said {tv}" if tv.startswith(("SELL", "STRONG_SELL")) else f"{sym} ({ml}%) but a safety check said no")
        if why:
            add(t0, "floor", "hold", "No buy this time. The AI liked " + "; ".join(why[:2]) + ".")
        else:
            add(t0, "floor", "hold", f"No buy this time: nothing reached its {thr:.0%} confidence bar.")
    return ev


def _sell(bot: str, t: str, m: str) -> list[dict]:
    s = re.search(r"SELL\s+(\d+)×(\S+)\s+@\s+\$\s*([\d.,]+)\s*\|\s*P&L: \$\s*([+-]?\s*[\d.,]+)\s*\(([+-]?[\d.]+)%\)\s*\[(\w+)\]", m)
    if not s:
        return []
    sym, pnl, pct, why = s.group(2), float(s.group(4).replace(" ", "").replace(",", "")), float(s.group(5)), s.group(6)
    word = "made" if pnl >= 0 else "lost"
    return [{"t": t, "bot": bot, "place": "broker", "kind": "sell", "symbol": sym, "pnl": pnl,
             "text": f"Sold {sym} ({REASONS.get(why, why.lower().replace('_', ' '))}): {word} {_money(abs(pnl))}, {pct:+.1f}%."}]


def day_bot(bot: str, d: Path) -> list[dict]:
    ev = []
    for run in runs_from_log(_tail(d / "ai_bot.log")):
        ev.extend(day_events(bot, run))
    # one retrain per day is the story; back-to-back test retrains are noise
    last_retrain = {}
    for i, e in enumerate(ev):
        if e["kind"] == "retrain":
            last_retrain[e["t"][:10]] = i
    keep = set(last_retrain.values())
    ev = [e for i, e in enumerate(ev) if e["kind"] != "retrain" or i in keep]
    card = d / "scorecard.csv"
    if card.exists():
        rows = list(csv.DictReader(card.open()))
        for r in rows:
            t = datetime.fromisoformat(r["date"] + "T15:55").replace(tzinfo=ET).astimezone(timezone.utc)
            ev.append({"t": t.isoformat().replace("+00:00", "Z"), "bot": bot, "place": "vault", "kind": "balance",
                       "text": f"Day’s closing balance: {_money(float(r['equity']))} ({float(r['bot_return']):+.2%} since its first recorded day; the market (SPY) {float(r['spy_return']):+.2%})."})
    return ev


def regime_bot(bot: str, d: Path) -> list[dict]:
    ev = []
    card = d / "regime_scorecard.csv"
    state = json.loads((d / "regime_paper_state.json").read_text()) if (d / "regime_paper_state.json").exists() else {}
    start = float(state.get("start_equity") or 25000)
    rows = list(csv.DictReader(card.open())) if card.exists() else []
    prev = None
    for r in rows:
        t = datetime.fromisoformat(r["date"] + "T15:45").replace(tzinfo=ET).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        mood, eq = r["regime"], float(r["equity"])
        plan = {"BULL": "the market is rising steadily, so it keeps holding SPY",
                "SIDEWAYS": "the market is choppy, so it buys strong stocks that just dipped",
                "BEAR": "the market is falling, so it holds SPY and tests a defensive plan on the side"}.get(mood, "")
        ev.append({"t": t, "bot": bot, "place": "tower", "kind": "mood", "mood": mood,
                   "text": f"Read the market mood: {mood}" + (f" (it was {prev} yesterday)" if prev and prev != mood else "") + f". {plan[:1].upper() + plan[1:]}." if plan else f"Read the market mood: {mood}."})
        ev.append({"t": t, "bot": bot, "place": "vault", "kind": "balance",
                   "text": f"Balance {_money(eq)}: {'up' if eq >= start else 'down'} {_money(abs(eq - start))} ({eq / start - 1:+.2%}) since it started; the market (SPY) {float(r['spy_return']):+.2%}."})
        prev = mood
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["day", "regime"])
    ap.add_argument("--bot", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    ev = day_bot(a.bot, d) if a.kind == "day" else regime_bot(a.bot, d)
    ev.sort(key=lambda e: e["t"])
    out = {"bot": a.bot, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "events": ev[-KEEP:]}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"events: {len(out['events'])} for {a.bot} -> {a.out}")


if __name__ == "__main__":
    main()
