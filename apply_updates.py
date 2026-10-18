#!/usr/bin/env python3
"""
apply_updates.py — one-shot patcher for ai_trading_bot.py

DISCLAIMER: Independent personal project, not tied to any employer or agency.
Paper trading only. Public market data only. No personal data (no SSNs/PHI).

What it does (safe to run many times — it skips if already patched):
  1. Replaces main() with a one-job-per-run version (--mode premarket|scan|eod|retrain)
  2. Adds runtime_state.json so watchlist, peak prices, dip-watch and earnings
     dates survive between GitHub Actions runs (fixes the overnight "reset")
  3. Adds a VIX fear-gauge filter: VIX >= 25 half-size, VIX >= 30 no new buys
"""
from pathlib import Path

BOT = Path("ai_trading_bot.py")
src = BOT.read_text(encoding="utf-8")

if "RUNTIME_FILE" in src:
    print("apply_updates: already patched — nothing to do")
    raise SystemExit(0)

VIX_BLOCK = '''# ══════════════════════════════════════════════════════════
# VIX FEAR GAUGE — blocks new buys during market panic
# Free Yahoo Finance data, no API key needed
# ══════════════════════════════════════════════════════════
VIX_CAUTION = 25    # above this -> half-size positions
VIX_PANIC   = 30    # above this -> no new buys at all
_vix_cache  = {'value': None, 'checked': None}


def get_vix():
    """Fetch current VIX level. Cached for 30 minutes."""
    now = now_et()
    if _vix_cache['checked'] and (now - _vix_cache['checked']).total_seconds() < 1800:
        return _vix_cache['value']
    try:
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=5d'
        r = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        vix = float(r.json()['chart']['result'][0]['meta']['regularMarketPrice'])
        _vix_cache.update(value=vix, checked=now)
        log.info(f'VIX fear gauge: {vix:.1f}')
        return vix
    except Exception as e:
        log.warning(f'VIX fetch failed: {e} — trading normally')
        return None


'''

VIX_GATE = '''    # ── VIX panic gate ────────────────────────────────────
    vix = get_vix()
    if vix is not None and vix >= VIX_PANIC:
        log.info(f'  {symbol}: VIX {vix:.1f} >= {VIX_PANIC} — market panic, no new buys')
        return False

'''

VIX_CAUTION_BLOCK = '''    if vix is not None and vix >= VIX_CAUTION:
        cap = min(cap, POSITION_CAP / 2)
        log.info(f'  {symbol}: VIX {vix:.1f} elevated — half-size position')
'''

NEW_MAIN = '''# ══════════════════════════════════════════════════════════
# RUNTIME STATE — survives between GitHub Actions runs
# ══════════════════════════════════════════════════════════
import argparse
RUNTIME_FILE = STATE_DIR / 'runtime_state.json'


def save_runtime_state() -> None:
    RUNTIME_FILE.write_text(json.dumps({
        'watchlist':      WATCHLIST,
        'position_highs': position_highs,
        'dip_watch': {
            s: {**w, 'exit_time': w['exit_time'].isoformat()}
            for s, w in dip_watch.items()
        },
        'earnings_cache': {
            s: (d.isoformat() if d else None) for s, d in earnings_cache.items()
        },
        'earnings_cache_date': _earnings_cache_date,
    }, indent=2))


def load_runtime_state() -> None:
    global WATCHLIST, position_highs, dip_watch, earnings_cache, _earnings_cache_date
    if not RUNTIME_FILE.exists():
        return
    try:
        s = json.loads(RUNTIME_FILE.read_text())
        WATCHLIST      = s.get('watchlist', FALLBACK_WATCHLIST)
        position_highs = s.get('position_highs', {})
        dip_watch = {
            sym: {**w, 'exit_time': datetime.fromisoformat(w['exit_time'])}
            for sym, w in s.get('dip_watch', {}).items()
        }
        earnings_cache = {
            sym: (datetime.fromisoformat(d) if d else None)
            for sym, d in s.get('earnings_cache', {}).items()
        }
        _earnings_cache_date = s.get('earnings_cache_date', '')
        log.info(f'Runtime state restored — watchlist {WATCHLIST}')
    except Exception as e:
        log.warning(f'Runtime state load failed: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['premarket', 'scan', 'eod', 'retrain'])
    mode = parser.parse_args().mode
    log.info(f'Run mode: {mode.upper()}')

    load_log()
    load_runtime_state()

    if mode == 'retrain':
        nightly_retrain()
    else:
        load_or_train()
        if mode == 'premarket':
            refresh_earnings_cache()
            run_news_intelligence()
            generate_morning_brief()
        elif mode == 'scan':
            refresh_earnings_cache()   # no-op if already done today
            et = now_et()
            if et.hour in (10, 12, 14) and et.minute < 10:
                run_intraday_news_update()
            scan()
        elif mode == 'eod':
            if is_market_open() and get_positions():
                close_all_positions('EOD')
            send_daily_report()

    save_runtime_state()
    log.info('Run complete — state saved')


if __name__ == '__main__':
    main()
'''

def must_find(marker):
    i = src.find(marker)
    if i == -1:
        raise SystemExit(f"apply_updates: could not find {marker!r} — file changed, aborting")
    return i

# 1. VIX helper above place_buy
i = must_find("def place_buy(")
src = src[:i] + VIX_BLOCK + src[i:]

# 2. Panic gate above the market regime gate (inside place_buy)
i = must_find("    # ── Market regime gate")
src = src[:i] + VIX_GATE + src[i:]

# 3. Caution sizing right before qty_usd is computed
i = must_find("    qty_usd = min(")
src = src[:i] + VIX_CAUTION_BLOCK + src[i:]

# 4. Replace main() and everything after it
i = must_find("def main():")
src = src[:i] + NEW_MAIN

BOT.write_text(src, encoding="utf-8")
print("apply_updates: ai_trading_bot.py patched (runtime state + one-job main + VIX filter)")
