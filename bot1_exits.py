"""
BOT 1 EXIT ADD-ON  —  middle-ground exits for Bot 1 ONLY.

Independent personal project, not tied to any employer or agency.
Paper trading only. Public market data only. No internal or personal data (no SSNs / PHI).

WHY (12-month backtest, backtest_exits.py, PR #23)
  Bot 1 sold everything by 3:50 PM, so most trades were 1-hour coin flips. Holding winners
  overnight beat that in all 6 head-to-head tests. Losers still never stay overnight.

WHAT THIS ADDS
  1. Carry winners: at the 3:50 PM close-out, a position that is up >= +1% is KEPT overnight with
     a breakeven floor (sell if it trades at or below the buy price). Everything else is sold as
     before. At most 2 nights (3 trading days), then it is sold at the close (CARRY_EOD).
  2. Real stop orders at Alpaca: every open position gets a GTC stop order at the bot's current
     floor: -2% at first, then breakeven / +2% / trailing as the existing profit locks arm (the
     floor only moves up). Between the bot's 10-minute checks Alpaca sells the moment the price
     hits it; after an overnight gap down it sells at the opening price (a gap can still lose
     more than 2% - no stop order can prevent that).
  A sell done by an Alpaca stop order is written to trade_log.json like any other sell, so the
  nightly retrain still learns from it.

HOW (ai_trading_bot.py is NOT changed - Day-Trade-100 imports that file and must stay as it is)
  This file imports ai_trading_bot, swaps in 4 functions (place_buy, place_sell,
  manage_open_positions, close_all_positions), then runs its normal main().
  Its own memory lives in STATE_DIR/bot1_exits.json.
  Run: python bot1_exits.py --mode scan       (same modes as ai_trading_bot.py)
"""
import json, time
from datetime import datetime

import ai_trading_bot as bot
from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

CARRY_MIN_PCT = 0.01        # up >= +1% at the close -> keep overnight
CARRY_MAX_NIGHTS = 2        # 3 trading days at most
STATE_FILE = bot.STATE_DIR / 'bot1_exits.json'
log = bot.log

state = {'carried': {}, 'stops': {}}
_orig = {}                  # the original ai_trading_bot functions (filled by install())


# ── memory between GitHub Actions runs ───────────────────────────────────
def load_state():
    global state
    try:
        if STATE_FILE.exists():
            s = json.loads(STATE_FILE.read_text())
            state = {'carried': s.get('carried', {}), 'stops': s.get('stops', {})}
    except Exception as e:
        log.warning(f'bot1_exits state not loaded: {e}')


def save_state():
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.warning(f'bot1_exits state not saved: {e}')


def today():
    return bot.now_et().strftime('%Y-%m-%d')


def status_of(order):
    s = getattr(order, 'status', '')
    return str(getattr(s, 'value', s)).lower().split('.')[-1]


# ── the floor = the price the bot would sell at (same tiers as manage_open_positions) ──
def floor_price(sym, entry, peak):
    peak_pct = peak / entry - 1
    if peak_pct >= bot.TRAIL_ACTIVATE_PCT:
        return max(peak * (1 - bot.TRAIL_PCT), entry * (1 + bot.TRAIL_FLOOR_PCT)), 'TRAILING_STOP'
    if peak_pct >= bot.LOCK_PROFIT_AT:
        return entry * (1 + bot.LOCK_PROFIT_FLOOR), 'PROFIT_LOCK'
    if peak_pct >= bot.LOCK_BREAKEVEN_AT:
        return entry, 'PROFIT_LOCK'
    if sym in state['carried']:
        return entry, 'CARRY_STOP'
    return entry * (1 - bot.STOP_PCT), 'STOP_LOSS'


# ── Alpaca stop orders ────────────────────────────────────────────────────
def open_sell_orders(sym):
    orders = bot.trade_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym]))
    return [o for o in orders if getattr(o, 'side', None) == OrderSide.SELL]


def cancel_sell_orders(sym, wait_sec=8.0):
    """Cancel this symbol's open sell orders (a stop order holds the shares, so a market sell
    would be refused with 'held_for_orders'). Returns True when none are left open."""
    for o in open_sell_orders(sym):
        try:
            bot.trade_client.cancel_order_by_id(o.id)
        except Exception as e:
            log.warning(f'  {sym}: cancel of order {o.id} failed: {e}')
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if not open_sell_orders(sym):
            return True
        time.sleep(0.5)
    return not open_sell_orders(sym)


def record_stop_fill(sym, info, order):
    """An Alpaca stop order sold the position between checks: log it like place_sell would."""
    fill = float(order.filled_avg_price)
    qty = float(order.filled_qty or info['qty'])
    entry = float(info['entry'])
    pnl = (fill - entry) * qty
    pct = fill / entry - 1
    reason = info.get('kind', 'STOP_LOSS')
    win = pnl > 0
    log.info(f'🔴 SELL {qty:5.0f}×{sym} @ ${fill:8.2f} | P&L: ${pnl:+8.2f} ({pct:+.1%}) [{reason}, Alpaca stop order]')
    bot.send_telegram(
        f'{"✅" if win else "❌"} <b>PAPER BOT — SELL [{reason}]</b> (Alpaca stop order)\n'
        f'Stock: <b>{sym}</b>  ·  {qty:.0f} shares @ ${fill:.2f}\n'
        f'Entry: ${entry:.2f}  →  Exit: ${fill:.2f}\n'
        f'P&L: <b>${pnl:+.2f} ({pct:+.1%})</b>')
    bot.save_log({'action': 'SELL', 'symbol': sym, 'qty': qty, 'entry_price': entry, 'exit_price': fill,
                  'pnl': pnl, 'pct': pct, 'win': win, 'reason': reason, 'via': 'alpaca_stop_order',
                  'carried_nights': state['carried'].get(sym, {}).get('nights', 0),
                  'timestamp': datetime.now().isoformat()})
    if reason in ('STOP_LOSS', 'TRAILING_STOP', 'PROFIT_LOCK'):
        try:
            bot.set_dip_watch(sym, reason, fill)
        except Exception as e:
            log.warning(f'  {sym}: dip watch not set: {e}')
    bot.position_highs.pop(sym, None)
    state['stops'].pop(sym, None)
    state['carried'].pop(sym, None)


def stop_filled(sym):
    """If this symbol's stop order has filled, log the sell and return True."""
    info = state['stops'].get(sym)
    if not info:
        return False
    try:
        o = bot.trade_client.get_order_by_id(info['id'])
    except Exception:
        return False
    if status_of(o) == 'filled' and o.filled_avg_price:
        record_stop_fill(sym, info, o)
        return True
    return False


def reconcile(positions):
    """Positions that vanished since the last run: was it our stop order? Then log the sell."""
    for sym in list(state['stops']):
        if sym in positions:
            continue
        if not stop_filled(sym):
            state['stops'].pop(sym, None)          # cancelled / expired / sold some other way
            state['carried'].pop(sym, None)
    for sym in list(state['carried']):
        if sym not in positions:
            state['carried'].pop(sym, None)


def sync_stop(sym, pos):
    """Keep one GTC stop order at the bot's floor. The floor only moves up."""
    qty = int(float(pos.qty))
    entry = float(pos.avg_entry_price)
    current = float(pos.current_price)
    peak = max(bot.position_highs.get(sym, entry), current)
    floor, kind = floor_price(sym, entry, peak)
    stop = round(floor, 2)
    if qty < 1 or stop >= current * 0.998:
        return              # at/above the price: the bot's own check sells it; Alpaca would refuse it
    cur = state['stops'].get(sym)
    if cur and cur['qty'] == qty and stop <= cur['stop'] + 0.005:
        return              # already there (or higher): the floor never moves down
    if not cancel_sell_orders(sym):
        log.warning(f'  {sym}: old stop order still open — new stop not placed this run')
        return
    if stop_filled(sym):
        return
    o = bot.trade_client.submit_order(StopOrderRequest(
        symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC, stop_price=stop))
    state['stops'][sym] = dict(id=str(o.id), stop=stop, qty=qty, entry=entry, kind=kind,
                               placed=datetime.now().isoformat())
    log.info(f'  {sym:5s}: 🛡️ Alpaca stop order at ${stop:.2f} ({stop / entry - 1:+.1%}, {kind})')


# ── the 4 swapped-in functions ────────────────────────────────────────────
def place_buy(symbol, price, equity, signal):
    ok = _orig['place_buy'](symbol, price, equity, signal)
    if ok:
        time.sleep(2)                                   # market order fills in ~1s on paper
        try:
            pos = bot.get_positions().get(symbol)
            if pos is not None:
                sync_stop(symbol, pos)                  # protect it right away, not at the next scan
        except Exception as e:
            log.warning(f'  {symbol}: stop order not placed after buy (next scan retries): {e}')
        save_state()
    return ok


def place_sell(symbol, pos, reason):
    try:
        cancel_sell_orders(symbol)
    except Exception as e:
        log.warning(f'  {symbol}: could not cancel stop order before selling: {e}')
    if stop_filled(symbol):                             # Alpaca already sold it
        save_state()
        return
    if reason == 'EOD' and symbol in state['carried']:
        reason = 'CARRY_EOD'
    state['stops'].pop(symbol, None)
    state['carried'].pop(symbol, None)
    _orig['place_sell'](symbol, pos, reason)
    save_state()


def manage_open_positions():
    positions = bot.get_positions()
    reconcile(positions)
    # carried overnight: breakeven floor (also catches a gap below the buy price at the open)
    for sym, pos in positions.items():
        if sym not in state['carried'] or sym not in bot.WATCHLIST:
            continue
        entry, cur = float(pos.avg_entry_price), float(pos.current_price)
        peak = max(bot.position_highs.get(sym, entry), cur)
        if peak / entry - 1 < bot.LOCK_BREAKEVEN_AT and cur <= entry:
            log.info(f'  {sym:5s}: 🌙 CARRY STOP | held overnight, now {cur / entry - 1:+.1%} — selling')
            bot.place_sell(sym, pos, 'CARRY_STOP')
    _orig['manage_open_positions']()                    # the 4 live tiers, unchanged
    for sym, pos in bot.get_positions().items():
        if sym in bot.WATCHLIST:
            try:
                sync_stop(sym, pos)
            except Exception as e:
                log.warning(f'  {sym}: stop order sync failed: {e}')
    save_state()


def close_all_positions(reason='EOD'):
    if reason != 'EOD':
        return _orig['close_all_positions'](reason)
    d = today()
    for sym, pos in bot.get_positions().items():
        if sym not in bot.WATCHLIST:
            continue
        c = state['carried'].get(sym)
        if c and c.get('last_carry') == d:
            continue                                     # already decided today (EOD can run twice)
        entry, cur = float(pos.avg_entry_price), float(pos.current_price)
        pct = cur / entry - 1
        nights = c['nights'] if c else 0
        if pct >= CARRY_MIN_PCT and nights < CARRY_MAX_NIGHTS:
            state['carried'][sym] = dict(since=c['since'] if c else d, nights=nights + 1,
                                         last_carry=d, entry=entry)
            log.info(f'  {sym:5s}: 🌙 CARRY OVERNIGHT | {pct:+.1%} >= +{CARRY_MIN_PCT:.0%} '
                     f'(night {nights + 1} of {CARRY_MAX_NIGHTS}) — floor moves to breakeven')
            bot.send_telegram(f'🌙 <b>PAPER BOT — HOLD OVERNIGHT</b>\nStock: <b>{sym}</b> up {pct:+.1%}\n'
                              f'Night {nights + 1} of {CARRY_MAX_NIGHTS} · stop at breakeven ${entry:.2f}')
            try:
                sync_stop(sym, pos)
            except Exception as e:
                log.warning(f'  {sym}: overnight stop order not placed: {e}')
        else:
            bot.place_sell(sym, pos, 'EOD')
    save_state()


def install():
    for name in ('place_buy', 'place_sell', 'manage_open_positions', 'close_all_positions'):
        _orig.setdefault(name, getattr(bot, name))
    bot.place_buy = place_buy
    bot.place_sell = place_sell
    bot.manage_open_positions = manage_open_positions
    bot.close_all_positions = close_all_positions


if __name__ == '__main__':
    load_state()
    install()
    try:
        bot.main()
    finally:
        save_state()
