#!/usr/bin/env python3
"""
DAY-TRADE-100 — the self-learning AI bot, scanning 100 liquid S&P 500 stocks.

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Uses public market data and an Alpaca PAPER (fake money) account only.
No internal, personal or protected data (no SSNs/PHI).

HOW IT WORKS (no changes to ai_trading_bot.py):
  This file imports the original bot and swaps a few settings before it runs:
    1. Keys      -> DAYTRADE100_ALPACA_* (its own paper account)
    2. Memory    -> state_dt100/ (its own model, trade log, learning history)
    3. Stocks    -> UNIVERSE_100 below, for both training and trading
    4. TradingView calls OFF (100 calls every 10 min would get rate-limited;
       the bot already redistributes that weight to the ML model)
    5. Morning news re-ranking OFF, so it always scans all 100
    6. Training window 365 days (900 days x 100 stocks is too heavy for a
       free GitHub runner)
    7. Adds a daily scorecard vs SPY: state_dt100/scorecard.csv
    8. Smaller time-decay rate so sample weights don't overflow with 100 stocks
    9. Simpler buy rule (model threshold + safety vetoes), since TradingView is off
"""
import os, sys, csv
from datetime import datetime

# ── 1 + 2: point the original bot at the new account and new folder ──
os.environ['ALPACA_API_KEY']    = os.environ['DAYTRADE100_ALPACA_API_KEY']
os.environ['ALPACA_SECRET_KEY'] = os.environ['DAYTRADE100_ALPACA_SECRET_KEY']
os.environ['STATE_DIR']         = 'state_dt100'
os.environ.pop('GEMINI_API_KEY', None)       # no Gemini for this test
os.environ.pop('TELEGRAM_BOT_TOKEN', None)   # keep alerts from mixing with bot #1

import ai_trading_bot as bot   # reads the env vars above at import time
bot.USER_WATCHLIST = []        # watchlist.txt is Bot 1's; this bot always trades its 100

# ── 3: the 100 most heavily traded S&P 500 names (fixed list) ──
UNIVERSE_100 = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','GOOG','META','TSLA','AVGO','BRK.B',
    'JPM','LLY','V','UNH','XOM','MA','COST','HD','PG','JNJ',
    'NFLX','ABBV','BAC','CRM','ORCL','AMD','CVX','KO','WMT','MRK',
    'PEP','ADBE','TMO','CSCO','ACN','LIN','MCD','ABT','WFC','INTC',
    'QCOM','DIS','TXN','INTU','IBM','AMAT','CAT','GE','VZ','PFE',
    'NOW','UBER','CMCSA','T','AMGN','ISRG','GS','MS','SPGI','BKNG',
    'HON','LOW','UNP','RTX','NEE','BA','PLTR','MU','LRCX','PANW',
    'ADI','KLAC','SBUX','C','SCHW','BLK','DE','AXP','MDT','GILD',
    'CRWD','ANET','MRVL','DELL','SMCI','COIN','PYPL','F','GM','NKE',
    'CVS','MO','PM','COP','SLB','OXY','FCX','DAL','CCL','WBD',
]
bot.TRAINING_UNIVERSE  = UNIVERSE_100 + ['SPY']
bot.WATCHLIST          = list(UNIVERSE_100)
bot.FALLBACK_WATCHLIST = list(UNIVERSE_100)
bot.SCAN_UNIVERSE      = list(UNIVERSE_100)
bot.TRAIN_DAYS         = 365                      # 6

# -- 8: FIX (Oct 18) time-decay weights overflowed with 100 stocks --
# The original formula is exp(TIME_DECAY_RATE * row_number). With ~650,000
# rows that became infinity, and inf/inf = NaN, so XGBoost refused to train
# ("Weights must be positive values"). This rate gives the oldest row about
# 5% weight and the newest 100%, and stays safe up to millions of rows.
bot.TIME_DECAY_RATE    = 3.0 / 700_000

# ── 4: TradingView off ──
bot.get_tv_analysis = lambda symbol: None

# ── 4b: FIX (Oct 19, 2026) buy rule for a bot without TradingView ──
# With TradingView off, the original 3-source score needed the model to be
# ~72%+ sure AND RSI in a narrow band to reach 0.48. The model's highest score
# ever is ~0.74, so bot #3 could almost never buy (13 scans, 0 trades, even
# with SCHW/PANW/DAL at 64%). New rule: BUY when the model clears its own
# tested threshold and no safety veto fires. The candidate list is ranked by
# model confidence. Bot #1 keeps the original rule.
_orig_combined = bot.combined_signal
def _dt100_signal(symbol, ml, tv):
    sig  = _orig_combined(symbol, ml, tv)         # keeps the original reasons for the log
    conf = float(ml.get('confidence', 0.0))
    rsi  = float(ml.get('rsi', 50))
    bb   = float(ml.get('bb_pct', 0.5))
    vetoes = []
    if rsi > 78:                       vetoes.append(f'RSI>{78}')
    if ml.get('above_ema50', 0) == 0:  vetoes.append('BELOW_EMA50')
    if bb > 0.95:                      vetoes.append('TOP_OF_BAND')
    if conf >= bot.ML_DECISION_THRESHOLD and not vetoes:
        sig['signal'] = 'BUY'
        sig['score']  = round(conf, 3)            # ranks candidates by model confidence
    else:
        sig['signal'] = 'HOLD'
        sig['score']  = round(conf - bot.ML_DECISION_THRESHOLD, 3)
    if vetoes:
        sig['reasons'] = sig.get('reasons', []) + ['VETO:' + '+'.join(vetoes)]
    return sig
bot.combined_signal = _dt100_signal

# ── 5: never shrink the list to a news-picked 10 ──
bot.run_news_intelligence    = lambda: bot.log.info('News re-ranking off — scanning all 100')
bot.run_intraday_news_update = lambda: None
_orig_load = bot.load_runtime_state
def _load_keep_100():
    _orig_load()
    bot.WATCHLIST = list(UNIVERSE_100)
bot.load_runtime_state = _load_keep_100

# ── 7: scorecard vs SPY, written at end of day ──
SCORE_FILE = bot.STATE_DIR / 'scorecard.csv'
def write_scorecard():
    if os.environ.get('DRY_RUN') == '1':
        bot.log.info('DRY_RUN — scorecard not saved'); return
    today = bot.now_et().strftime('%Y-%m-%d')
    equity = bot.get_account_equity()
    spy = float(bot.fetch_bars('SPY', days=5)['close'].iloc[-1])
    rows = list(csv.reader(open(SCORE_FILE))) if SCORE_FILE.exists() else []
    if rows and rows[-1][0] == today:
        return                                   # already logged today
    if not rows:
        rows = [['date', 'equity', 'spy_close', 'bot_return', 'spy_return']]
        start_eq, start_spy = equity, spy
    else:
        start_eq, start_spy = float(rows[1][1]), float(rows[1][2])
    rows.append([today, round(equity, 2), round(spy, 2),
                 round(equity / start_eq - 1, 4), round(spy / start_spy - 1, 4)])
    csv.writer(open(SCORE_FILE, 'w', newline='')).writerows(rows)
    bot.log.info(f'Scorecard: bot {rows[-1][3]:+.2%} vs SPY {rows[-1][4]:+.2%} since start')

if __name__ == '__main__':
    bot.main()                                   # same --mode premarket|scan|eod|retrain
    if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode') + 1] == 'eod':
        write_scorecard()
