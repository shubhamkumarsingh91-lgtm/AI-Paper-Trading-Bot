"""Hand-checked tests for backtest_news.py. Independent personal project; synthetic data only."""
import os, sys, tempfile
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STATE_DIR', tempfile.mkdtemp())
import backtest_news as bn   # noqa: E402

ET = 'America/New_York'


@pytest.mark.parametrize('text,score', [
    ('Nvidia beats estimates and raises guidance', 2),
    ('Broadcom upgraded to buy; price target raised', 2),          # upgrade + target raised
    ('UnitedHealth misses estimates, cuts guidance', -2),
    ('Analyst downgrades Intel; lowers price target', -2),
    ('Apple announces developer conference dates', 0),
    ('Pfizer wins $2 billion government contract', 1),
    ('Company under SEC investigation, shares plunge', -2),
])
def test_score_text(text, score):
    assert bn.score_text(text) == score


def _bars(day, opens, closes, prev_day_close=100.0):
    """prev day: one 15:45 bar closing at prev_day_close; then 'day' bars from 9:30."""
    p = pd.Timestamp(f'{day} 09:30', tz=ET) - pd.Timedelta(days=1) + pd.Timedelta(hours=6, minutes=15)
    idx = [p] + list(pd.date_range(f'{day} 09:30', periods=len(opens), freq='15min', tz=ET))
    o = [prev_day_close] + opens
    c = [prev_day_close] + closes
    return pd.DataFrame(dict(open=o, close=c, high=c, low=c, volume=1.0), index=pd.DatetimeIndex(idx))


def test_day_rows_by_hand():
    # prev close 100 ; 9:30 open 102 (gap +2%) ; 10:00 open 103 > 102 -> confirm entry ; flat after ; 15:45 open 103.5
    n = 26
    opens = [102, 102.5, 103] + [103.2] * (n - 4) + [103.5]
    closes = [102.5, 103, 103.2] + [103.2] * (n - 4) + [103.6]
    b = _bars('2026-10-21', opens, closes)
    d = pd.Timestamp('2026-10-21', tz=ET)
    rows = bn.day_rows('X', b, {('X', d): [(2, True)]})
    assert len(rows) == 1
    r = rows[0]
    assert r['score'] == 2 and r['n_news'] == 1 and r['earnings'] == 1
    assert r['gap'] == pytest.approx(102 / 100 - 1)
    # open entry 102 -> peak 103.2 (+1.18% < 1.5%) never stops -> EOD at 15:45 open 103.5
    assert r['open_exit'] == 'EOD' and r['open_ret'] == pytest.approx(103.5 * 0.9997 / (102 * 1.0003) - 1)
    assert r['confirm'] == 1 and r['confirm_ret'] == pytest.approx(103.5 * 0.9997 / (103 * 1.0003) - 1)


def test_day_rows_no_confirm_when_below_open():
    n = 26
    opens = [102, 101.5, 101] + [101] * (n - 4) + [101]
    closes = [101.5, 101, 101] + [101] * (n - 4) + [101]
    rows = bn.day_rows('X', _bars('2026-10-21', opens, closes), {})
    assert rows[0]['confirm'] == 0 and np.isnan(rows[0]['confirm_ret']) and rows[0]['n_news'] == 0


def test_overnight_window():
    days = [pd.Timestamp(d, tz=ET) for d in ('2026-10-16', '2026-10-19', '2026-10-20')]   # Fri, Mon, Tue
    t = lambda x: pd.Timestamp(x, tz=ET)
    news = [
        dict(time=t('2026-10-16 17:00'), headline='NVDA beats estimates', summary='', symbols=['NVDA']),   # Fri evening -> Mon
        dict(time=t('2026-10-18 12:00'), headline='NVDA upgraded', summary='', symbols=['NVDA']),         # Sunday -> Mon
        dict(time=t('2026-10-19 09:20'), headline='AVGO misses', summary='', symbols=['AVGO', 'ZZZ']),    # Mon pre-open -> Mon
        dict(time=t('2026-10-19 11:00'), headline='NVDA raises guidance', summary='', symbols=['NVDA']),  # Mon market hours -> ignored
        dict(time=t('2026-10-19 16:30'), headline='AVGO record revenue', summary='', symbols=['AVGO']),   # Mon evening -> Tue
    ]
    got = bn.assign_news(news, days, {'NVDA', 'AVGO'})
    assert got == {('NVDA', days[1]): [(1, False), (1, False)],
                   ('AVGO', days[1]): [(-1, False)],
                   ('AVGO', days[2]): [(1, True)]}       # 'revenue' marks it as earnings news; ZZZ ignored
