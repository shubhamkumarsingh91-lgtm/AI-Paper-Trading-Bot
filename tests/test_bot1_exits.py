"""
Hand-checked tests for bot1_exits.py with a fake Alpaca account (no network, no real orders).
Run: python -m pytest -q tests/test_bot1_exits.py

Independent personal project, not tied to any employer or agency. Synthetic data only.
"""
import os, sys, tempfile, itertools
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STATE_DIR', tempfile.mkdtemp())
import ai_trading_bot as bot      # noqa: E402
import bot1_exits as bx           # noqa: E402
from alpaca.trading.enums import OrderSide   # noqa: E402

ORIG_MANAGE = bot.manage_open_positions      # the real live tiers
ORIG_COMBINED = bot.combined_signal          # the real live buy decision


class FakeAlpaca:
    def __init__(self):
        self.orders, self.ids = {}, itertools.count(1)

    def submit_order(self, req):
        o = NS(id=f'o{next(self.ids)}', symbol=req.symbol, qty=req.qty, side=req.side,
               stop_price=getattr(req, 'stop_price', None), tif=req.time_in_force,
               status='new', filled_avg_price=None, filled_qty=None)
        self.orders[o.id] = o
        return o

    def get_orders(self, req):
        return [o for o in self.orders.values() if o.status == 'new' and o.symbol in req.symbols]

    def cancel_order_by_id(self, oid):
        self.orders[oid].status = 'canceled'

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def fill(self, oid, price):
        o = self.orders[oid]
        o.status, o.filled_avg_price, o.filled_qty = 'filled', price, o.qty


@pytest.fixture
def acct(monkeypatch, tmp_path):
    a = NS(alpaca=FakeAlpaca(), positions={}, sells=[], logs=[], tg=[], day='2026-11-17')
    monkeypatch.setattr(bot, 'trade_client', a.alpaca)
    monkeypatch.setattr(bot, 'get_positions', lambda: dict(a.positions))
    monkeypatch.setattr(bot, 'send_telegram', lambda m: a.tg.append(m))
    monkeypatch.setattr(bot, 'save_log', lambda e: a.logs.append(e))
    monkeypatch.setattr(bot, 'set_dip_watch', lambda *x: None)
    monkeypatch.setattr(bot, 'WATCHLIST', ['AMD', 'GS', 'FCX'])
    monkeypatch.setattr(bot, 'position_highs', {})
    monkeypatch.setattr(bx, 'STATE_FILE', tmp_path / 'bot1_exits.json')
    monkeypatch.setattr(bx, 'today', lambda: a.day)
    monkeypatch.setattr(bx.time, 'sleep', lambda s: None)
    monkeypatch.setattr(bx, 'state', {'carried': {}, 'stops': {}})

    def fake_sell(sym, pos, reason):          # stands in for the live market sell
        a.sells.append((sym, reason))
        a.positions.pop(sym, None)
    bx._orig.clear()
    bx._orig.update(place_sell=fake_sell, place_buy=lambda *x: True,
                    manage_open_positions=ORIG_MANAGE, close_all_positions=lambda r='EOD': None)
    for n in ('place_buy', 'place_sell', 'manage_open_positions', 'close_all_positions'):
        monkeypatch.setattr(bot, n, getattr(bx, n))
    return a


def pos(entry, cur, qty=10):
    return NS(qty=str(qty), avg_entry_price=str(entry), current_price=str(cur), unrealized_pl=str((cur - entry) * qty))


def open_stops(a, sym):
    return [o for o in a.alpaca.orders.values() if o.symbol == sym and o.status == 'new']


def test_new_position_gets_minus_2pct_gtc_stop(acct):
    acct.positions['AMD'] = pos(100.0, 100.5)
    bot.manage_open_positions()
    [o] = open_stops(acct, 'AMD')
    assert (o.stop_price, o.qty, o.side, o.tif.value) == (98.0, 10, OrderSide.SELL, 'gtc')   # 100 * (1 - 2%)
    bot.manage_open_positions()                    # nothing changed -> same single order, no churn
    assert len(acct.alpaca.orders) == 1


def test_stop_moves_up_to_breakeven_and_never_down(acct):
    acct.positions['AMD'] = pos(100.0, 100.5)
    bot.manage_open_positions()                    # stop 98.00
    acct.positions['AMD'] = pos(100.0, 101.6)      # peak +1.6% >= +1.5% -> breakeven lock
    bot.manage_open_positions()
    [o] = open_stops(acct, 'AMD')
    assert o.stop_price == 100.0
    acct.positions['AMD'] = pos(100.0, 100.8)      # falls back but stays above the floor
    bot.manage_open_positions()
    [o2] = open_stops(acct, 'AMD')
    assert o2.id == o.id and acct.sells == []      # not lowered, not sold


def test_eod_carries_winner_sells_the_rest(acct):
    acct.positions.update(AMD=pos(100.0, 101.2), GS=pos(50.0, 50.25), FCX=pos(40.0, 39.5))
    # AMD +1.2% >= +1% -> carried;  GS +0.5% -> sold;  FCX -1.25% -> sold
    bot.close_all_positions('EOD')
    assert sorted(acct.sells) == [('FCX', 'EOD'), ('GS', 'EOD')]
    assert bx.state['carried']['AMD'] == dict(since='2026-11-17', nights=1, last_carry='2026-11-17', entry=100.0)
    [o] = open_stops(acct, 'AMD')
    assert o.stop_price == 100.0                   # breakeven floor overnight (GTC)
    bot.close_all_positions('EOD')                 # EOD running twice the same day changes nothing
    assert len(acct.sells) == 2 and bx.state['carried']['AMD']['nights'] == 1


def test_max_two_nights_then_sold_as_carry_eod(acct):
    acct.positions['AMD'] = pos(100.0, 102.0)
    bot.close_all_positions('EOD')                         # night 1
    acct.day = '2026-11-18'; bot.close_all_positions('EOD')  # night 2
    assert bx.state['carried']['AMD']['nights'] == 2
    acct.day = '2026-11-19'; bot.close_all_positions('EOD')  # day 3: sold even though still up
    assert acct.sells == [('AMD', 'CARRY_EOD')] and open_stops(acct, 'AMD') == []
    assert 'AMD' not in bx.state['carried']


def test_gap_below_buy_price_next_morning_sold(acct):
    acct.positions['AMD'] = pos(100.0, 101.2)
    bot.close_all_positions('EOD')
    acct.day = '2026-11-18'
    acct.positions['AMD'] = pos(100.0, 99.1)       # opened below the buy price
    bot.manage_open_positions()
    assert acct.sells == [('AMD', 'CARRY_STOP')]
    assert open_stops(acct, 'AMD') == []           # the stop order was cancelled before the sell


def test_alpaca_stop_fill_between_runs_is_logged_once(acct):
    acct.positions['AMD'] = pos(100.0, 100.5, qty=10)
    bot.manage_open_positions()
    [o] = open_stops(acct, 'AMD')
    acct.alpaca.fill(o.id, 97.9)                   # Alpaca sold it at 97.90
    del acct.positions['AMD']
    bot.manage_open_positions()
    [e] = acct.logs
    assert (e['symbol'], e['reason'], e['via'], e['exit_price']) == ('AMD', 'STOP_LOSS', 'alpaca_stop_order', 97.9)
    assert e['pnl'] == pytest.approx(-21.0) and e['pct'] == pytest.approx(-0.021) and e['win'] is False   # (97.9-100)*10
    bot.manage_open_positions()
    assert len(acct.logs) == 1 and acct.sells == [] and bx.state['stops'] == {}


def test_sell_after_stop_already_filled_does_not_sell_twice(acct):
    acct.positions['AMD'] = pos(100.0, 100.5)
    bot.manage_open_positions()
    [o] = open_stops(acct, 'AMD')
    acct.alpaca.fill(o.id, 98.0)
    bot.place_sell('AMD', acct.positions['AMD'], 'EOD')     # position object is stale
    assert acct.sells == [] and [e['reason'] for e in acct.logs] == ['STOP_LOSS']


def test_minus_2pct_still_sold_by_live_tier_and_stop_cancelled_first(acct):
    acct.positions['AMD'] = pos(100.0, 100.5)
    bot.manage_open_positions()
    acct.positions['AMD'] = pos(100.0, 97.9)       # the live tier-1 rule fires
    bot.manage_open_positions()
    assert acct.sells == [('AMD', 'STOP_LOSS')] and open_stops(acct, 'AMD') == []


def test_bot2_code_untouched():
    # Day-Trade-100 imports ai_trading_bot; importing bot1_exits must not change it by itself
    import importlib, subprocess
    out = subprocess.run([sys.executable, '-c',
                          'import ai_trading_bot as b, bot1_exits; '
                          'print(b.place_sell.__module__, b.close_all_positions.__module__)'],
                         capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         env={**os.environ, 'STATE_DIR': tempfile.mkdtemp()})
    assert out.stdout.split()[-2:] == ['ai_trading_bot', 'ai_trading_bot'], out.stderr[-500:]


# ── Nov 19: TradingView exchange lookup + blocked-buy log ─────────────────
from alpaca.trading.enums import AssetExchange   # noqa: E402


def test_exchange_looked_up_once_and_remembered(acct, monkeypatch):
    calls = []

    def get_asset(sym):
        calls.append(sym)
        return NS(exchange={'GS': AssetExchange.NYSE, 'SPYG': AssetExchange.ARCA}[sym])
    monkeypatch.setattr(acct.alpaca, 'get_asset', get_asset, raising=False)
    m = bx.ExchangeMap({'AMD': 'NASDAQ'})
    assert m.get('AMD', 'NASDAQ') == 'NASDAQ' and calls == []          # known: no lookup
    assert m.get('GS', 'NASDAQ') == 'NYSE' and m.get('GS', 'NASDAQ') == 'NYSE'
    assert calls == ['GS'] and bx.state['exchanges'] == {'GS': 'NYSE'}  # asked once, saved
    assert m.get('SPYG', 'NASDAQ') == 'AMEX'                           # TradingView calls NYSE Arca "AMEX"
    m2 = bx.ExchangeMap({})                                            # next run: comes from the saved state
    assert m2.get('GS', 'NASDAQ') == 'NYSE' and calls == ['GS', 'SPYG']


def test_exchange_lookup_failure_falls_back_and_is_not_saved(acct, monkeypatch):
    def boom(sym):
        raise RuntimeError('network down')
    monkeypatch.setattr(acct.alpaca, 'get_asset', boom, raising=False)
    assert bx.ExchangeMap({}).get('DAL', 'NASDAQ') == 'NASDAQ'
    assert 'DAL' not in bx.state.get('exchanges', {})


def test_install_swaps_exchange_map_and_get_tv_analysis_uses_it(acct, monkeypatch):
    seen = {}

    class FakeTA:
        def __init__(self, symbol, screener, exchange, interval):
            seen[symbol] = exchange

        def get_analysis(self):
            raise RuntimeError('stop here')
    monkeypatch.setattr(bot, 'TA_Handler', FakeTA)
    monkeypatch.setattr(bot, 'TV_EXCHANGE', dict(bot.TV_EXCHANGE))
    monkeypatch.setattr(acct.alpaca, 'get_asset', lambda s: NS(exchange=AssetExchange.NYSE), raising=False)
    bx._orig.pop('combined_signal', None)
    for n in ('place_buy', 'place_sell', 'manage_open_positions', 'close_all_positions', 'combined_signal'):
        monkeypatch.setattr(bot, n, getattr(bot, n))      # restore after the test
    bx.install()
    bot.get_tv_analysis('FCX')
    bot.get_tv_analysis('AMD')
    assert seen == {'FCX': 'NYSE', 'AMD': 'NASDAQ'}


def _ml(conf, above=1.0, rsi=50, bb=0.5):
    return {'confidence': conf, 'above_ema50': above, 'rsi': rsi, 'bb_pct': bb, 'vol_ratio': 1.0}


def test_blocked_buy_is_logged_with_the_rule_in_words(acct, monkeypatch):
    lines = []
    monkeypatch.setattr(bx, 'log', NS(info=lines.append, warning=lines.append))
    monkeypatch.setattr(bot, 'ML_DECISION_THRESHOLD', 0.66)
    monkeypatch.setattr(bot, 'BUY_RULE', 'precision_first')
    bx._orig['combined_signal'] = ORIG_COMBINED
    sig = bx.combined_signal('FCX', _ml(0.70, above=0.0), None)        # cleared 66%, below its trend average
    assert sig['signal'] == 'HOLD'
    assert lines == ['  FCX  : ⛔ BUY BLOCKED — AI 70% cleared its 66% cutoff, but: price is below its 50-bar trend average']
    sig = bx.combined_signal('AMD', _ml(0.67, rsi=80, bb=0.97), {'rec': 'STRONG_SELL', 'adx': 20})
    assert 'overbought (RSI above 78); at the very top of its Bollinger band; TradingView says STRONG SELL' in lines[-1]
    assert bx.state['blocked'] == {'2026-11-17': {'BELOW_EMA50': 1, 'RSI>78': 1, 'TOP_OF_BAND': 1, 'TV_STRONG_SELL': 1}}


def test_not_blocked_cases_log_nothing(acct, monkeypatch):
    lines = []
    monkeypatch.setattr(bx, 'log', NS(info=lines.append, warning=lines.append))
    monkeypatch.setattr(bot, 'ML_DECISION_THRESHOLD', 0.66)
    monkeypatch.setattr(bot, 'BUY_RULE', 'precision_first')
    bx._orig['combined_signal'] = ORIG_COMBINED
    assert bx.combined_signal('GS', _ml(0.60, above=0.0), None)['signal'] == 'HOLD'   # under the cutoff anyway
    sig = bx.combined_signal('GS', _ml(0.70), None)                                  # clean buy, unchanged
    assert (sig['signal'], sig['score']) == ('BUY', 0.70)
    assert lines == [] and bx.state.get('blocked', {}) == {}
