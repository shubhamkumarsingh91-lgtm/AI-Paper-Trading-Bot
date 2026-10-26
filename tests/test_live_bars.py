"""Live decisions must use finished 15-minute bars only, and the news request must validate.
Independent personal project, not tied to any employer or agency. Synthetic data only. Not financial advice."""
import os, sys
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ALPACA_API_KEY', 'test'); os.environ.setdefault('ALPACA_SECRET_KEY', 'test')
import ai_trading_bot as bot


def bars(*starts):
    idx = pd.to_datetime(list(starts)).tz_localize('America/New_York')
    return pd.DataFrame({'close': range(len(idx)), 'volume': [1000] * len(idx)}, index=idx)


def test_unfinished_bar_is_dropped():
    df = bars('2026-10-26 12:30', '2026-10-26 12:45', '2026-10-26 13:00')
    now = datetime(2026, 10, 26, 17, 1, tzinfo=timezone.utc)          # 1:01 PM ET: the 1:00 bar is 1 minute old
    out = bot.drop_unfinished_bar(df, now)
    assert [t.strftime('%H:%M') for t in out.index] == ['12:30', '12:45']


def test_bar_kept_once_finished():
    df = bars('2026-10-26 12:45', '2026-10-26 13:00')
    now = datetime(2026, 10, 26, 17, 15, tzinfo=timezone.utc)         # 1:15 PM ET: the 1:00 bar just closed
    assert len(bot.drop_unfinished_bar(df, now)) == 2


def test_empty_is_safe():
    assert bot.drop_unfinished_bar(bars(), datetime.now(timezone.utc)).empty


def test_news_request_accepts_the_symbols_we_send():
    from alpaca.data.requests import NewsRequest
    end = datetime.now(timezone.utc)
    NewsRequest(symbols=','.join(bot.SCAN_UNIVERSE[:30]), start=end - pd.Timedelta(hours=3), end=end, limit=100, sort='desc')
