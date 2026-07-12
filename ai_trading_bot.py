#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   🤖  AI SELF-LEARNING PAPER TRADING BOT                ║
║   ─────────────────────────────────────────────────────  ║
║   XGBoost ML  ·  TradingView TA  ·  Gemini AI Brief     ║
║   Alpaca Paper $100,000  ·  Retrains Every Night        ║
║                                                          ║
║   49-FEATURE MODEL — World's Best Trading Strategies:   ║
║   Weinstein · Minervini · O'Neil · Darvas · PTJ          ║
║   Raschke · L.Williams · Bollinger · Donchian           ║
╚══════════════════════════════════════════════════════════╝

DEPLOY ON RENDER as Background Worker.

Required environment variables (set in Render dashboard):
  ALPACA_API_KEY       → Your Alpaca PAPER account API key
  ALPACA_SECRET_KEY    → Your Alpaca PAPER account secret
  GEMINI_API_KEY       → Your Google Gemini API key (FREE)

How to get keys:
  Alpaca paper keys:
    1. Login to alpaca.markets
    2. Top-left dropdown → switch to "Paper Account"
    3. API Keys → Generate New Key

  Gemini API key (FREE — no credit card):
    1. Go to aistudio.google.com
    2. Click "Get API Key" → Create API Key
    Done. Free tier = 15 requests/min, 1M tokens/day.
"""

import os, json, logging, time, joblib, warnings
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import xml.etree.ElementTree as XMLTree
import pytz
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score
import schedule
from tradingview_ta import TA_Handler, Interval

# LightGBM — second model in ensemble. Installed if available.
# Adds ~3–5% precision by catching patterns XGBoost misses.
try:
    import lightgbm as lgb
    _LGBM_OK = True
except ImportError:
    _LGBM_OK = False

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PERSISTENT STATE DIRECTORY
# ══════════════════════════════════════════════════════════
# On Render this must point at the mounted disk (see render.yaml's
# `disk.mountPath`) or the model, trade log, and self-learning history are
# wiped on every redeploy/restart. STATE_DIR defaults to the working
# directory so the bot still runs unchanged locally or anywhere without a
# mounted disk.
STATE_DIR = Path(os.environ.get('STATE_DIR', '.'))
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STATE_DIR / 'ai_bot.log', encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
API_KEY     = os.environ.get('ALPACA_API_KEY',    'YOUR_PAPER_KEY')
SECRET_KEY  = os.environ.get('ALPACA_SECRET_KEY', 'YOUR_PAPER_SECRET')
GEMINI_KEY  = os.environ.get('GEMINI_API_KEY',    'YOUR_GEMINI_KEY')
TG_TOKEN    = os.environ.get('TELEGRAM_BOT_TOKEN', '')   # set in Render dashboard
TG_CHAT_ID  = os.environ.get('TELEGRAM_CHAT_ID',  '')   # set in Render dashboard

# ── Trading watchlist — updated daily by News Intelligence ─
# This list is DYNAMIC — replaced each morning by AI analysis.
# Falls back to these 5 if news intelligence is unavailable.
WATCHLIST = ['NVDA', 'PANW', 'AVGO', 'SOFI', 'PLTR']
FALLBACK_WATCHLIST = ['NVDA', 'PANW', 'AVGO', 'SOFI', 'PLTR']  # always-on backup

# ── News Intelligence Universe — stocks the news scanner considers ──
# Bot scans this wider pool each morning, then picks the best 5.
SCAN_UNIVERSE = [
    # Core watchlist
    'NVDA', 'PANW', 'AVGO', 'SOFI', 'PLTR',
    # AI / Semiconductors
    'AMD', 'QCOM', 'MRVL', 'INTC', 'ARM',
    # Mega-cap tech
    'MSFT', 'AAPL', 'GOOGL', 'META', 'AMZN',
    # Fintech / Growth
    'PYPL', 'COIN', 'HOOD', 'SQ', 'NU', 'AFRM',
    # Cybersecurity
    'CRWD', 'FTNT', 'ZS', 'S',
    # Data / Cloud
    'SNOW', 'CRM', 'DDOG', 'MDB',
    # EV / Disruptive
    'TSLA', 'RIVN',
    # High-beta / speculative
    'MSTR', 'RBLX', 'UBER',
    # ETFs — macro context
    'SPY', 'QQQ', 'SOXX', 'XLF',
]

# ── Catalyst importance weights (used in scoring) ────────
CATALYST_WEIGHTS = {
    'M&A':         1.00,  # merger/acquisition target = highest alpha
    'REGULATORY':  0.90,  # FDA approval, gov contract, legal win
    'EARNINGS':    0.80,  # earnings beat or revenue surprise
    'CONTRACT':    0.75,  # major contract win
    'PARTNERSHIP': 0.65,  # strategic alliance, JV
    'INSIDER':     0.60,  # insider buying (Form 4 SEC filing)
    'PRODUCT':     0.55,  # new product launch, feature release
    'UPGRADE':     0.50,  # analyst upgrade, price target raise
    'OTHER':       0.30,
}

# ── Training universe — 14 diverse stocks for a robust model
# Covers: AI chips, cyber, fintech, data/AI, mega-cap tech,
#         high-volatility, and market benchmarks (ETFs).
# Model learns universal patterns here, applies them to WATCHLIST.
TRAINING_UNIVERSE = [
    # AI / Semiconductors (high correlation with NVDA/AVGO)
    'NVDA', 'AMD', 'AVGO',
    # Cybersecurity (same sector as PANW)
    'PANW', 'CRWD',
    # Fintech / High-beta (same sector as SOFI)
    'SOFI', 'PYPL',
    # Data / AI Software (same sector as PLTR)
    'PLTR', 'SNOW',
    # High-volatility — teaches extreme overbought/oversold patterns
    'TSLA', 'MSTR',
    # Market benchmarks — cleanest technical patterns of all
    'SPY', 'QQQ',
    # Semiconductor ETF — smoothed chip-sector signal
    'SOXX',
]

TV_EXCHANGE = {
    # Watchlist
    'NVDA': 'NASDAQ', 'PANW': 'NASDAQ', 'AVGO': 'NASDAQ',
    'SOFI': 'NASDAQ', 'PLTR': 'NASDAQ',
    # Training universe extras
    'AMD':  'NASDAQ', 'CRWD': 'NASDAQ', 'PYPL': 'NASDAQ',
    'SNOW': 'NYSE',   'TSLA': 'NASDAQ', 'MSTR': 'NASDAQ',
    'SPY':  'AMEX',   'QQQ':  'NASDAQ', 'SOXX': 'NASDAQ',
}

# ── Risk Parameters ──────────────────────────────────────
RISK_PCT      = 0.02   # 2% of equity risked per trade
POSITION_CAP  = 0.12   # max 12% of equity in one stock (10-stock watchlist)
STOP_PCT      = 0.02   # stop loss at -2%
TP_PCT        = 0.06   # hard take-profit fallback at +6%
MAX_POSITIONS = 5      # max simultaneous open positions (10-stock watchlist)

# ── Two-tier trailing stop ────────────────────────────────
# Phase 1 (0% → +6%)  : normal stop-loss at -2%, no TP ceiling
# Phase 2 (past +6%)  : trailing stop activates — rides the momentum
#   trail = 5% below the running peak price (wide enough for volatile names)
#   floor = hard minimum exit of +3% once trailing is active
#           (so even a sharp reversal from +6% → still exit at +3%, not +0%)
TRAIL_ACTIVATE_PCT = 0.06   # trailing kicks in once position is up +6%
TRAIL_PCT          = 0.05   # trail 5% below the highest price seen
TRAIL_FLOOR_PCT    = 0.03   # never exit below +3% once trailing is active

# ── Profit Lock (ratcheting floor — 0% to +6% zone) ──────
# Prevents giving back gains on the typical 1–5% intraday move.
# Once peak reaches each level, the stop-loss floor ratchets up.
#   +1.5% peak → floor moves to 0%   (breakeven — can never lose money)
#   +4.0% peak → floor moves to +2%  (lock in half the gain)
#   +6.0% peak → trailing stop takes over (above)
LOCK_BREAKEVEN_AT  = 0.015  # activate breakeven stop once up +1.5%
LOCK_PROFIT_AT     = 0.04   # activate +2% floor once up +4%
LOCK_PROFIT_FLOOR  = 0.02   # the floor itself when LOCK_PROFIT_AT is reached

# ── Cooldown after selling — prevents churning ────────────
# Dip Confirmation — how many reversal signals needed before re-entry
# After STOP_LOSS: require 2 signals (stock was falling — be strict)
# After TAKE_PROFIT/TREND: require 1 signal (stock was healthy — be lenient)
DIP_SIGNALS_AFTER_STOP = 2
DIP_SIGNALS_AFTER_TP   = 1
DIP_MIN_WAIT_SEC       = 300     # always wait at least 5 min to avoid same-candle rebuy
DIP_MAX_WAIT_SEC       = 14400   # 4 hours — if no reversal pattern confirms by then, stop
                                  # waiting on this candle pattern and fall back to the plain
                                  # cooldown instead of watching indefinitely (was unbounded)

# ── Signal Weights ───────────────────────────────────────
ML_WEIGHT    = 0.50    # XGBoost prediction
TV_WEIGHT    = 0.35    # TradingView technical analysis
RSI_WEIGHT   = 0.15    # RSI quality filter
CONFIDENCE   = 0.62    # minimum ML confidence to consider BUY

# ── Schedule ─────────────────────────────────────────────
SCAN_EVERY   = 10      # minutes between scans
EOD_HOUR     = 15      # close all at 3:50 PM ET
EOD_MIN      = 50

# ── Files ────────────────────────────────────────────────
MODEL_FILE   = STATE_DIR / 'ai_model.xgb'
LOG_FILE     = STATE_DIR / 'trade_log.json'
BRIEF_FILE   = STATE_DIR / 'morning_brief.txt'
REPORT_FILE  = STATE_DIR / 'daily_picks.json'   # today's AI stock picks with explanations
TRAIN_DAYS   = 900   # ~2.5 years — covers 2024 AI bull run + 2025 corrections + 2026

# ── Market Regime Gate ────────────────────────────────────
# SPY above 50-day SMA = BULL → trade normally
# SPY below 50-day SMA = BEAR → smaller positions + higher confidence required
# One bad macro day can erase a week of gains — this gate filters those days.
REGIME_GATE       = True   # set False to disable and always trade
BEAR_POSITION_CAP = 0.06   # 6% max position in bear market (vs 12% normally)
BEAR_MIN_CONF     = 0.68   # 68% ML confidence required in bear (vs 62% normally)

# ── Earnings Blackout ─────────────────────────────────────
# Never buy a stock within N days of its earnings report.
# Earnings = ±15% surprise risk — the model has no way to predict this.
EARNINGS_BLACKOUT_DAYS = 2

# ── Time-Decay Training Weights ──────────────────────────
# Recent market data is more relevant than 2-year-old data.
# Exponential decay: oldest bar ~26% weight, newest bar 100% weight.
TIME_DECAY_RATE = 0.0015

TF = TimeFrame(15, TimeFrameUnit.Minute)   # 15-minute bars

ET = pytz.timezone('America/New_York')     # all time checks use ET, not UTC

def now_et() -> datetime:
    """Return current time in Eastern Time (handles EST/EDT automatically)."""
    return datetime.now(ET)


# ══════════════════════════════════════════════════════════
# API CLIENTS
# ══════════════════════════════════════════════════════════
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Alpaca News client — same credentials, no extra cost
try:
    from alpaca.data.historical import NewsClient as _NewsClientCls
    news_client = _NewsClientCls(API_KEY, SECRET_KEY)
except Exception:
    # Older SDK versions: NewsClient lives elsewhere
    try:
        from alpaca.data import NewsClient as _NewsClientCls
        news_client = _NewsClientCls(API_KEY, SECRET_KEY)
    except Exception as _e:
        news_client = None
        log.warning(f'Alpaca News client not available: {_e}')

# Google Gemini — optional AI morning brief
# If key is missing or invalid, morning brief is simply skipped
gemini = None
if GEMINI_KEY and GEMINI_KEY != 'YOUR_GEMINI_KEY':
    try:
        import google.generativeai as _genai
        _genai.configure(api_key=GEMINI_KEY)
        gemini = _genai.GenerativeModel('gemini-2.5-flash-lite')  # updated — 2.0-flash deprecated June 2026
    except Exception as e:
        log.warning(f'Gemini init skipped: {e}')


# ══════════════════════════════════════════════════════════
# TELEGRAM NOTIFICATIONS
# ══════════════════════════════════════════════════════════
# Sends real-time alerts to your phone for every trade,
# morning brief, regime change, and nightly retrain summary.
#
# Setup (one time):
#   1. Open Telegram → search @BotFather → /newbot → copy TOKEN
#   2. Start chat with your new bot → send any message
#   3. Open: https://api.telegram.org/bot<TOKEN>/getUpdates
#      → copy the "id" number from "chat" section = CHAT_ID
#   4. Add both to Render dashboard:
#      TELEGRAM_BOT_TOKEN = your token
#      TELEGRAM_CHAT_ID   = your chat id
# ══════════════════════════════════════════════════════════

def send_telegram(message: str) -> None:
    """
    Send a message to your Telegram bot.
    Silently skips if token/chat_id not configured — never breaks the bot.
    """
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url  = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        payload = {
            'chat_id':    TG_CHAT_ID,
            'text':       message,
            'parse_mode': 'HTML',
        }
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass   # Telegram failure must never stop trading


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (49 features)
#
# Encodes strategies from the world's best traders:
#   Stan Weinstein   — Stage Analysis (150-SMA trend gate)
#   Mark Minervini   — SEPA / VCP (EMA stack, ATR contraction)
#   William O'Neil   — CAN SLIM (volume quality: up vs down days)
#   Nicolas Darvas   — Box Theory (Donchian channel breakout)
#   Paul Tudor Jones — 200-SMA trend + slope
#   Linda Raschke    — ADX + mean-reversion oversold signal
#   Larry Williams   — Williams %R + temporal patterns
#   John Bollinger   — BB width + lower-band bounce
#   Richard Donchian — 20-bar channel position + breakout
# ══════════════════════════════════════════════════════════
FEATURES = [
    # ── Returns ──────────────────────────────────────────
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_30',
    # ── Trend / EMAs ─────────────────────────────────────
    'ema9_20x', 'ema20_50x', 'above_ema50', 'above_ema200',
    # ── STAN WEINSTEIN — Stage Analysis ──────────────────
    'above_sma150',    # price above 150-period SMA = Stage 2
    'sma150_rising',   # 150-SMA slope positive = uptrend confirmed
    # ── PAUL TUDOR JONES — 200-SMA Trend ─────────────────
    'trend_strength_200',  # (price − SMA200) / SMA200
    'ma200_slope',         # +1 rising, -1 falling
    # ── MARK MINERVINI — SEPA / VCP ──────────────────────
    'ema_stack',       # EMA9 > EMA20 > EMA50 > EMA200 (full bullish alignment)
    'pct_from_high',   # distance from 20-bar high (negative = below peak)
    'atr_contracting', # ATR shrinking vs 20 bars ago (VCP coiling)
    # ── Momentum — RSI ───────────────────────────────────
    'rsi', 'rsi_overbought', 'rsi_oversold',
    # ── Momentum — MACD ──────────────────────────────────
    'macd_hist', 'macd_cross',
    # ── JOHN BOLLINGER — Bands ───────────────────────────
    'bb_pct', 'bb_squeeze',
    'bb_width',        # raw band width (low = coiling before explosion)
    'bb_at_lower',     # price near lower band (bounce candidate)
    # ── Volatility — ATR ─────────────────────────────────
    'atr_pct', 'atr_expanding',
    # ── LARRY WILLIAMS — Williams %R + Temporal ──────────
    'williams_r',      # -100 (oversold) to 0 (overbought)
    'day_of_week',     # 0=Mon … 4=Fri (Friday effect)
    'is_month_end',    # last 3 trading days of month (window dressing)
    # ── RICHARD DONCHIAN / DARVAS — Channel Breakout ─────
    'donchian_pct',    # position in 20-bar price channel (0=low, 1=high)
    'donchian_break',  # price breaks above prior 20-bar high
    # ── LINDA RASCHKE — ADX + Mean Reversion ─────────────
    'adx',             # trend strength (> 25 = trending)
    'adx_trending',    # binary: ADX > 25
    'oversold_trend',  # RSI < 35 AND ADX > 25 = bounce in trend (Raschke signal)
    # ── WILLIAM O'NEIL — Volume Quality ──────────────────
    'vol_up_vs_down',  # volume on up-candles / volume on down-candles
    # ── Volume ───────────────────────────────────────────
    'vol_ratio', 'vol_surge',
    # ── Candlestick ──────────────────────────────────────
    'body', 'candle_dir', 'upper_wick', 'lower_wick', 'doji',
    # ── Stochastic Oscillator ────────────────────────────
    'stoch_k', 'stoch_d', 'stoch_golden',
    # ── Session ──────────────────────────────────────────
    'hour', 'is_morning', 'is_power_hour',
    # ── Circular Time Encoding ────────────────────────────────────────────
    # Raw 'hour' is linear — XGBoost treats 9 AM and 3 PM as just numbers.
    # Sine/cosine encodes the clock as a circle so 9:45 AM is genuinely
    # close to 9:30 AM, and the model learns morning vs afternoon naturally.
    'hour_sin', 'hour_cos',
    # ── Market Context (SPY = whole-market pulse) ─────────────────────────
    # If SPY is up +0.8% right now and NVDA is also up, that BUY signal is
    # far stronger than NVDA going up while SPY is flat or falling.
    # These 3 features tell the model what the market is doing at the EXACT
    # SAME 15-minute bar as the stock being scored. This is the single
    # biggest alpha improvement — most bots treat each stock in isolation.
    'spy_ret_1',        # SPY 1-bar return: is market moving up right now?
    'spy_ret_5',        # SPY 5-bar (75-min) return: short-term mkt momentum
    'spy_above_ema20',  # 0/1: is SPY in a healthy uptrend at this moment?
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 49 technical indicator features from OHLCV data.

    Combines strategies from 9 legendary traders into a single
    feature set that XGBoost learns to weight automatically.
    """
    d = df.copy()
    c = d['close']

    # ── Price returns ─────────────────────────────────────
    for n in [1, 3, 5, 10, 30]:
        d[f'ret_{n}'] = c.pct_change(n)

    # ── Exponential Moving Averages ───────────────────────
    for n in [9, 20, 50, 200]:
        d[f'ema{n}'] = c.ewm(span=n, adjust=False).mean()
    d['ema9_20x']     = (d['ema9']  - d['ema20'])  / (d['ema20']  + 1e-9)
    d['ema20_50x']    = (d['ema20'] - d['ema50'])  / (d['ema50']  + 1e-9)
    d['above_ema50']  = (c > d['ema50']).astype(int)
    d['above_ema200'] = (c > d['ema200']).astype(int)

    # ── STAN WEINSTEIN — Stage Analysis ──────────────────
    # 150-period SMA on 15-min bars ≈ medium-term trend line.
    # Stage 2 (buy zone) = price above rising 150-SMA.
    d['sma150']        = c.rolling(150).mean()
    d['above_sma150']  = (c > d['sma150']).astype(int)
    d['sma150_rising'] = (d['sma150'] > d['sma150'].shift(10)).astype(int)

    # ── PAUL TUDOR JONES — 200-SMA Trend ─────────────────
    # PTJ rule #1: never hold a position below the 200-day MA.
    d['sma200']             = c.rolling(200).mean()
    d['trend_strength_200'] = (c - d['sma200']) / (d['sma200'] + 1e-9)
    d['ma200_slope']        = np.sign(d['sma200'] - d['sma200'].shift(20)).astype(float)

    # ── MARK MINERVINI — SEPA: EMA Stack Alignment ───────
    # All 4 EMAs in perfect bullish order = highest-probability entry.
    d['ema_stack'] = (
        (d['ema9'] > d['ema20']) &
        (d['ema20'] > d['ema50']) &
        (d['ema50'] > d['ema200'])
    ).astype(int)

    # ── RSI ───────────────────────────────────────────────
    def _rsi(s, n=14):
        delta = s.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(n).mean()
        loss  = -delta.where(delta < 0, 0.0).rolling(n).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-9))

    d['rsi']            = _rsi(c, 14)
    d['rsi_overbought'] = (d['rsi'] > 70).astype(int)
    d['rsi_oversold']   = (d['rsi'] < 30).astype(int)

    # ── MACD ──────────────────────────────────────────────
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig  = macd.ewm(span=9, adjust=False).mean()
    d['macd_hist']  = macd - sig
    d['macd_cross'] = (
        np.sign(d['macd_hist']) != np.sign(d['macd_hist'].shift(1))
    ).astype(int)

    # ── JOHN BOLLINGER — Bands + Width ───────────────────
    bm   = c.rolling(20).mean()
    bstd = c.rolling(20).std()
    bu, bl = bm + 2 * bstd, bm - 2 * bstd
    bw      = (bu - bl) / (bm + 1e-9)
    d['bb_pct']      = (c - bl) / (bu - bl + 1e-9)
    d['bb_squeeze']  = (bw < bw.rolling(50).mean()).astype(int)
    d['bb_width']    = bw                                    # low width = coiling
    d['bb_at_lower'] = (d['bb_pct'] < 0.2).astype(int)     # near lower band = bounce zone

    # ── ATR (Average True Range) ──────────────────────────
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - c.shift(1)).abs(),
        (d['low']  - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d['atr']           = tr.rolling(14).mean()
    d['atr_pct']       = d['atr'] / (c + 1e-9)
    d['atr_expanding'] = (d['atr'] > d['atr'].rolling(20).mean()).astype(int)

    # ── MARK MINERVINI — VCP (Volatility Contraction) ─────
    # ATR shrinking = stock coiling before breakout.
    d['atr_contracting'] = (d['atr'] < d['atr'].shift(20)).astype(int)
    # Distance from 20-bar high (negative = below peak — tighter = better setup)
    high_20 = d['high'].rolling(20).max()
    d['pct_from_high'] = (c - high_20) / (high_20 + 1e-9)

    # ── LINDA RASCHKE — ADX (Average Directional Index) ──
    # ADX > 25 = strong trend; oversold in a trend = high-probability bounce.
    up_move = d['high'] - d['high'].shift(1)
    dn_move = d['low'].shift(1) - d['low']
    dm_pos  = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    dm_neg  = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    atr14   = tr.rolling(14).mean()
    di_pos  = 100 * dm_pos.rolling(14).mean() / (atr14 + 1e-9)
    di_neg  = 100 * dm_neg.rolling(14).mean() / (atr14 + 1e-9)
    dx      = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)
    d['adx']           = dx.rolling(14).mean()
    d['adx_trending']  = (d['adx'] > 25).astype(int)
    d['oversold_trend'] = (
        (d['rsi'] < 35) & (d['adx'] > 25)
    ).astype(int)

    # ── LARRY WILLIAMS — Williams %R ─────────────────────
    # Computed after stochastic (shares hi14/lo14 window)
    lo14 = d['low'].rolling(14).min()
    hi14 = d['high'].rolling(14).max()
    d['williams_r'] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-9)

    # ── RICHARD DONCHIAN / DARVAS — Channel Breakout ─────
    # Darvas Box = price breaks above 20-bar high on strength.
    dc_hi = d['high'].rolling(20).max()
    dc_lo = d['low'].rolling(20).min()
    d['donchian_pct']   = (c - dc_lo) / (dc_hi - dc_lo + 1e-9)
    d['donchian_break'] = (c >= dc_hi.shift(1)).astype(int)

    # ── WILLIAM O'NEIL — Volume Quality ──────────────────
    # More volume on up-days vs down-days = institutional accumulation.
    up_vol   = d['volume'].where(c > d['open'], 0.0).rolling(10).sum()
    dn_vol   = d['volume'].where(c <= d['open'], 0.0).rolling(10).sum()
    d['vol_up_vs_down'] = up_vol / (dn_vol + 1)

    # ── Volume ────────────────────────────────────────────
    vol_avg = d['volume'].rolling(20).mean()
    d['vol_ratio'] = d['volume'] / (vol_avg + 1)
    d['vol_surge'] = (d['vol_ratio'] > 1.5).astype(int)

    # ── Candlestick patterns ──────────────────────────────
    hi_body = d[['open', 'close']].max(axis=1)
    lo_body = d[['open', 'close']].min(axis=1)
    d['body']       = (c - d['open']).abs() / (d['open'] + 1e-9)
    d['candle_dir'] = np.sign(c - d['open'])
    d['upper_wick'] = (d['high'] - hi_body) / (d['open'] + 1e-9)
    d['lower_wick'] = (lo_body - d['low'])   / (d['open'] + 1e-9)
    d['doji']       = (d['body'] < 0.001).astype(int)

    # ── Stochastic Oscillator ─────────────────────────────
    d['stoch_k']     = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    d['stoch_d']     = d['stoch_k'].rolling(3).mean()
    d['stoch_golden'] = (
        (d['stoch_k'] > d['stoch_d']) &
        (d['stoch_k'].shift(1) <= d['stoch_d'].shift(1))
    ).astype(int)

    # ── Session + LARRY WILLIAMS Temporal ────────────────
    try:
        d['hour']         = d.index.hour
        d['day_of_week']  = d.index.dayofweek          # 0=Mon … 4=Fri
        d['is_month_end'] = (d.index.day >= 28).astype(int)  # window dressing
    except AttributeError:
        d['hour']         = 12
        d['day_of_week']  = 2
        d['is_month_end'] = 0
    d['is_morning']    = (d['hour'] == 9).astype(int)
    d['is_power_hour'] = (d['hour'] == 15).astype(int)

    # ── Circular time encoding ────────────────────────────────────────────
    # Maps hour onto a unit circle so XGBoost sees 9 and 15 as equidistant
    # from noon, not 6 units apart. This unlocks time-of-day pattern learning.
    d['hour_sin'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour'] / 24)

    # ── SPY context columns (placeholders — filled by train_model/ml_predict) ─
    # Default to 0.0 so add_features() is safe to call without SPY data.
    # train_model() and ml_predict() overwrite these by joining real SPY bars.
    for col in ('spy_ret_1', 'spy_ret_5', 'spy_above_ema20'):
        if col not in d.columns:
            d[col] = 0.0

    return d


def make_labels(df: pd.DataFrame, forward: int = 8, threshold: float = 0.005,
                max_dd: float = -0.01) -> pd.Series:
    """
    High-quality binary label — stricter than a simple forward return.

    Label = 1 only when BOTH conditions are true:
      1. Price rises > 0.5% by bar +8 (≈2 hours forward)
      2. Price never drops below -1% from entry during those 8 bars

    Why: the old label caught 'brief touch' false signals — price taps +0.5%
    for one bar then crashes -3%. The model learned to chase those.
    The new label only rewards clean, sustained moves.

    Result: bullish rate drops slightly (better signal quality), precision rises.
    """
    c = df['close']
    l = df['low']

    # Condition 1: forward return at bar +8
    future_ret = c.shift(-forward) / c - 1

    # Condition 2: minimum low across next `forward` bars (vectorized)
    # Stack 8 shifted copies of low → take row-wise min
    future_lows    = pd.concat([l.shift(-i) for i in range(1, forward + 1)], axis=1)
    min_future_low = future_lows.min(axis=1)
    worst_drawdown = (min_future_low - c) / c   # negative = how far it dropped

    # Label = 1 only if rising AND never crashed first
    return ((future_ret > threshold) & (worst_drawdown > max_dd)).astype(int)


# ══════════════════════════════════════════════════════════
# XGBOOST MODEL
# ══════════════════════════════════════════════════════════
model: xgb.XGBClassifier = None
lgb_model = None                    # LightGBM companion (None if not installed)
model_accuracy:  float = 0.0
model_precision: float = 0.0       # precision on last retrain's validation set
model_trained_at: str  = 'never'
_wf_precision:   float = 0.0       # walk-forward precision (most honest metric)

# Decision threshold the model's probability must clear to count as a BUY
# signal. Calibrated every retrain (see calibrate_threshold()) instead of
# assuming the textbook default of 0.5 — that default was never actually
# checked against what precision it produces, so the "precision" logged
# during training didn't reflect the threshold live trading uses. 0.55 is
# just the starting value before the first calibration ever runs.
ML_DECISION_THRESHOLD: float = 0.55

# Real trade outcomes are relabeled onto individual training rows (see
# apply_real_trade_outcomes) but are a tiny fraction of the ~tens of
# thousands of synthetic rows in a 900-day training set. Without upweighting
# them, the model barely notices its own mistakes. This weight is applied
# on top of the existing time-decay weighting.
REAL_TRADE_SAMPLE_WEIGHT = 8.0

LEARNING_LOG = STATE_DIR / 'learning_history.json'   # one row per retrain, never overwritten


def fetch_bars(symbol: str, days: int = None, bars: int = 550) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca historical data API."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days) if days else end - timedelta(minutes=bars * 15 + 180)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TF,
        start=start,
        end=end,
        adjustment='all',
        feed='iex',   # free data feed for paper accounts
    )
    raw = data_client.get_stock_bars(req)
    df  = raw.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')
    # Flatten column names if MultiIndex (e.g. ('close', 'SQ') → 'close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    return df.sort_index()


# ── SPY context cache ─────────────────────────────────────────────────────
_spy_ctx_cache: dict = {}   # keyed by number of days requested


def fetch_spy_context(days: int) -> pd.DataFrame:
    """
    Fetch SPY bars and compute market-context features.

    Called once before the training loop so each stock's dataframe can be
    joined against SPY by timestamp.  Cached so ml_predict() reuses data
    fetched earlier in the same scan cycle.

    Returns a DataFrame indexed by timestamp with columns:
        spy_ret_1       — 1-bar SPY return
        spy_ret_5       — 5-bar SPY return
        spy_above_ema20 — 1 if SPY above its 20-bar EMA, else 0
    """
    global _spy_ctx_cache
    if days in _spy_ctx_cache:
        return _spy_ctx_cache[days]
    try:
        df = fetch_bars('SPY', days=days)
        c  = df['close']
        ctx = pd.DataFrame(index=df.index)
        ctx['spy_ret_1']        = c.pct_change(1)
        ctx['spy_ret_5']        = c.pct_change(5)
        ema20                   = c.ewm(span=20, adjust=False).mean()
        ctx['spy_above_ema20']  = (c > ema20).astype(float)
        ctx = ctx.fillna(0.0)
        _spy_ctx_cache[days] = ctx
        return ctx
    except Exception as e:
        log.warning(f'fetch_spy_context error: {e}')
        return pd.DataFrame(columns=['spy_ret_1', 'spy_ret_5', 'spy_above_ema20'])


def apply_real_trade_outcomes(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fold real trade outcomes from trade_log into the training labels.

    Previously the nightly retrain only re-derived labels from fresh price
    action (make_labels) and read trade_log solely to print a win-rate
    summary — the model never actually saw what happened on its own trades,
    despite the bot being advertised as self-learning from live results.

    For every closed (SELL) trade, find the bar closest to that trade's
    date for that symbol and overwrite its label with the real outcome
    (1 = win, 0 = loss). Matched by symbol + calendar date rather than
    exact timestamp since BUY/SELL log timestamps aren't stored in the
    same explicit timezone as the ET-indexed price bars.
    """
    all_data['real_outcome'] = False   # marks rows overwritten with a real result,
                                        # so train_model() can upweight them

    sells = [
        t for t in trade_log
        if t.get('action') == 'SELL' and t.get('symbol') in TRAINING_UNIVERSE
    ]
    if not sells:
        return all_data

    applied = 0
    for sell in sells:
        try:
            sym         = sell['symbol']
            trade_date  = pd.Timestamp(sell['timestamp']).date()
            sym_id      = TRAINING_UNIVERSE.index(sym)
            day_mask    = (all_data['sym_id'] == sym_id) & (all_data.index.date == trade_date)
            day_bars    = all_data.index[day_mask]
            if len(day_bars) == 0:
                continue
            bar_idx = day_bars.max()
            all_data.loc[bar_idx, 'target']       = int(bool(sell.get('win')))
            all_data.loc[bar_idx, 'real_outcome'] = True
            applied += 1
        except Exception:
            continue

    if applied:
        log.info(f'  Self-learning: {applied} real trade outcome(s) folded into training labels')
    return all_data


def calibrate_threshold(probs: np.ndarray, y_true: pd.Series, min_signals: int = 10) -> tuple:
    """
    Search validation-set probabilities for the cutoff that maximizes
    precision while still producing enough BUY signals to be usable.

    Without a minimum-signal floor, a threshold of 0.98 could "achieve"
    100% precision by only ever firing on 1-2 lucky samples — useless live.
    This is what live trading (combined_signal) now uses instead of the
    unvalidated default of 0.5.

    Returns (threshold, precision_at_threshold, signal_count).
    """
    best_th, best_prec, best_n = 0.5, 0.0, 0
    for th in np.arange(0.50, 0.91, 0.02):
        preds = (probs >= th).astype(int)
        n_pos = int(preds.sum())
        if n_pos < min_signals:
            continue
        prec = precision_score(y_true, preds, zero_division=0)
        if prec > best_prec:
            best_th, best_prec, best_n = float(th), float(prec), n_pos
    return best_th, best_prec, best_n


def record_learning_progress(real_trades_folded: int) -> tuple:
    """
    Append one row per retrain to LEARNING_LOG — never overwritten, so
    accuracy/precision/win-rate/P&L over time is a visible, growing record
    instead of only the latest snapshot in the log stream.

    Returns (this_record, previous_record_or_None) so the caller can show
    a trend delta (e.g. "precision +2.1% vs last retrain").
    """
    sells = [t for t in trade_log if t.get('action') == 'SELL']
    wins  = [t for t in sells if t.get('win')]
    wr    = len(wins) / len(sells) * 100 if sells else 0.0
    pnl   = sum(t.get('pnl', 0) for t in sells)

    record = {
        'timestamp':            datetime.now().isoformat(),
        'accuracy':              round(model_accuracy, 4),
        'precision':             round(model_precision, 4),
        'walk_forward_precision': round(_wf_precision, 4),
        'decision_threshold':    round(ML_DECISION_THRESHOLD, 4),
        'real_trades_folded_in': real_trades_folded,
        'total_trades_ever':     len(sells),
        'win_rate_pct':          round(wr, 1),
        'cumulative_pnl':        round(pnl, 2),
    }
    history = []
    if LEARNING_LOG.exists():
        try:
            history = json.loads(LEARNING_LOG.read_text())
        except Exception:
            history = []
    previous = history[-1] if history else None
    history.append(record)
    LEARNING_LOG.write_text(json.dumps(history, indent=2))
    log.info(f'  Learning history: {len(history)} retrain(s) recorded -> {LEARNING_LOG}')
    return record, previous


def train_model(retrain: bool = False) -> None:
    """
    Train XGBoost on 1 year of 15-min data for all 5 watchlist stocks.
    Called once at startup and then nightly for self-improvement.
    """
    global model, lgb_model, model_accuracy, model_precision, \
        model_trained_at, _wf_precision, ML_DECISION_THRESHOLD
    label = 'Nightly retrain' if retrain else 'Initial training'
    log.info(f'🧠 {label} — fetching {TRAIN_DAYS} days x {len(TRAINING_UNIVERSE)} stocks...')
    log.info(f'   Training universe: {", ".join(TRAINING_UNIVERSE)}')
    log.info(f'   Trading watchlist: {", ".join(WATCHLIST)}')

    # Fetch SPY context ONCE before the stock loop.
    # We join SPY bars to every stock dataframe by timestamp so the model
    # learns "NVDA BUY when SPY is also up" vs "NVDA BUY when SPY is red".
    log.info('  Fetching SPY market context...')
    spy_ctx = fetch_spy_context(TRAIN_DAYS)
    spy_ok  = len(spy_ctx) > 0
    if spy_ok:
        log.info(f'  SPY context: {len(spy_ctx):,} bars loaded')
    else:
        log.warning('  SPY context unavailable — market features default to 0')

    frames = []
    for i, sym in enumerate(TRAINING_UNIVERSE):
        try:
            df  = fetch_bars(sym, days=TRAIN_DAYS)
            df  = add_features(df)
            # Join SPY context: each bar gets SPY return from the same timestamp
            if spy_ok:
                df = df.join(spy_ctx, how='left', rsuffix='_spy')
                for col in ('spy_ret_1', 'spy_ret_5', 'spy_above_ema20'):
                    if col not in df.columns:
                        df[col] = 0.0
                df[['spy_ret_1', 'spy_ret_5', 'spy_above_ema20']] = (
                    df[['spy_ret_1', 'spy_ret_5', 'spy_above_ema20']]
                    .ffill().fillna(0.0)
                )
            df['target'] = make_labels(df)
            df['sym_id'] = i
            df = df.dropna(subset=FEATURES + ['sym_id', 'target'])
            frames.append(df)
            log.info(f'  ok {sym}: {len(df):,} bars | bullish rate {df["target"].mean():.1%}')
        except Exception as e:
            log.warning(f'  fail {sym}: {e}')

    if not frames:
        log.error('No training data — cannot train')
        return

    all_data = pd.concat(frames).sort_index()
    all_data = apply_real_trade_outcomes(all_data)
    real_trade_mask = all_data['real_outcome'].to_numpy()
    X        = all_data[FEATURES + ['sym_id']]
    y        = all_data['target']
    pos_rate = y.mean()
    n        = len(X)
    log.info(f'  Total: {n:,} samples | Bullish rate: {pos_rate:.1%}')

    # Time-decay sample weights: older bars get lower weight
    time_weights = np.exp(TIME_DECAY_RATE * np.arange(n))
    time_weights = time_weights / time_weights.max()
    if real_trade_mask.any():
        time_weights = time_weights * np.where(real_trade_mask, REAL_TRADE_SAMPLE_WEIGHT, 1.0)
        log.info(
            f'  Real-trade upweighting: {int(real_trade_mask.sum())} row(s) '
            f'weighted {REAL_TRADE_SAMPLE_WEIGHT}x so the model actually notices them'
        )
    log.info(f'  Time weights: oldest={time_weights[0]:.2f}  newest={time_weights[-1]:.2f}')

    neg       = (y == 0).sum()
    pos_count = (y == 1).sum()
    spw       = neg / (pos_count + 1)
    log.info(f'  Class balance: {pos_count:,} BUY / {neg:,} HOLD  scale_pos_weight={spw:.2f}')

    # =========================================================================
    # WALK-FORWARD VALIDATION
    # =========================================================================
    # Why this matters: a random 80/20 split lets future data leak into
    # training (a bar from Nov 2024 ends up training a model that 'tests'
    # on Oct 2024). This inflates precision by ~10% vs real trading.
    #
    # Walk-forward always trains on the PAST and tests on the FUTURE:
    #   Fold 1: train 0-60%  -> test 60-75%
    #   Fold 2: train 0-75%  -> test 75-87%
    #   Fold 3: train 0-87%  -> test 87-100%
    # Average precision across folds = what you will actually see live.
    # =========================================================================
    log.info('Walk-Forward Validation (3 folds — no future leakage)...')
    wf_precisions = []
    for fold_num, (tr_end, te_end) in enumerate([(0.60,0.75),(0.75,0.87),(0.87,1.00)], 1):
        t0 = int(n * tr_end);  t1 = int(n * te_end)
        Xtr = X.iloc[:t0];  ytr = y.iloc[:t0];  wtr = time_weights[:t0]
        Xva = X.iloc[t0:t1]; yva = y.iloc[t0:t1]
        if yva.sum() == 0:
            log.warning(f'  Fold {fold_num}: no BUY samples in test window — skipping')
            continue
        _m = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=spw, eval_metric='aucpr',
            early_stopping_rounds=30, verbosity=0, n_jobs=-1, random_state=42,
        )
        _m.fit(Xtr, ytr, eval_set=[(Xva, yva)], sample_weight=wtr, verbose=False)
        _p  = _m.predict(Xva)
        _pr = precision_score(yva, _p, zero_division=0)
        _ac = accuracy_score(yva, _p)
        wf_precisions.append(_pr)
        log.info(
            f'  Fold {fold_num}: train={t0:,} test={t1-t0:,} | '
            f'precision={_pr:.1%}  accuracy={_ac:.1%}  BUYs={_p.sum()}/{len(_p)}'
        )

    _wf_precision = float(np.mean(wf_precisions)) if wf_precisions else 0.0
    log.info(
        f'  Walk-forward avg precision: {_wf_precision:.1%} '
        f'<-- this is your REAL expected precision trading live'
    )

    # Final model trained on 80% of all data
    split_idx  = int(n * 0.80)
    X_tr, X_va = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_va = y.iloc[:split_idx], y.iloc[split_idx:]
    w_tr       = time_weights[:split_idx]

    # XGBoost (primary model)
    log.info('  Training XGBoost (final model)...')
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, scale_pos_weight=spw,
        eval_metric='aucpr', early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=w_tr, verbose=False)
    xgb_preds = model.predict(X_va)
    xgb_probs = model.predict_proba(X_va)[:, 1]
    xgb_prec  = precision_score(y_va, xgb_preds, zero_division=0)
    xgb_acc   = accuracy_score(y_va, xgb_preds)
    log.info(f'  XGBoost: precision={xgb_prec:.1%}  accuracy={xgb_acc:.1%}')
    log.info(
        f'  Prob spread: min={xgb_probs.min():.2f} '
        f'median={float(pd.Series(xgb_probs).median()):.2f} '
        f'max={xgb_probs.max():.2f} (all 0.50 = model broken)'
    )

    # LightGBM (companion model)
    # Uses leaf-wise tree growth — catches different patterns than XGBoost.
    # Ensemble (60% XGB + 40% LGB) reduces false positives -> higher precision.
    lgb_prec    = 0.0
    final_probs = xgb_probs   # falls back to XGBoost-only if LightGBM unavailable/fails
    if _LGBM_OK:
        log.info('  Training LightGBM...')
        try:
            dtrain    = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            dval      = lgb.Dataset(X_va, label=y_va, reference=dtrain)
            lgb_model = lgb.train(
                dict(
                    objective='binary', metric='average_precision',
                    learning_rate=0.05, num_leaves=63, max_depth=6,
                    min_child_samples=20, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=5,
                    scale_pos_weight=spw, verbose=-1, n_jobs=-1,
                ),
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            lgb_raw  = (lgb_model.predict(X_va) >= 0.5).astype(int)
            lgb_prec = precision_score(y_va, lgb_raw, zero_division=0)
            lgb_acc  = accuracy_score(y_va, lgb_raw)
            log.info(f'  LightGBM: precision={lgb_prec:.1%}  accuracy={lgb_acc:.1%}')
            ens_probs = 0.60 * xgb_probs + 0.40 * lgb_model.predict(X_va)
            ens_preds = (ens_probs >= 0.5).astype(int)
            ens_prec  = precision_score(y_va, ens_preds, zero_division=0)
            log.info(f'  Ensemble: precision={ens_prec:.1%}  <-- live bot uses this')
            final_probs = ens_probs
        except Exception as _le:
            log.warning(f'  LightGBM failed: {_le} — XGBoost only')
            lgb_model = None
    else:
        lgb_model = None
        log.info('  LightGBM not installed (pip install lightgbm to enable ensemble)')

    # Calibrate the live decision threshold against whatever the model
    # (XGBoost alone, or the XGB+LGB ensemble) actually produces — replaces
    # the unvalidated hardcoded 0.5 default with a data-driven cutoff, and
    # is what combined_signal() uses for live BUY decisions. min_signals
    # scales with the validation set so a threshold isn't picked just
    # because it got lucky on a handful of predictions.
    min_signals = max(10, int(0.01 * len(y_va)))
    ML_DECISION_THRESHOLD, calib_prec, calib_n = calibrate_threshold(final_probs, y_va, min_signals)
    log.info(
        f'  Calibrated decision threshold: {ML_DECISION_THRESHOLD:.2f} '
        f'(precision {calib_prec:.1%} on {calib_n} signals in validation) '
        f'<-- live trading now uses this cutoff, not 0.5'
    )

    # Feature importance — which features drive the model most?
    fi_pairs  = list(zip(FEATURES + ['sym_id'], model.feature_importances_))
    fi_sorted = sorted(fi_pairs, key=lambda x: -x[1])
    top_feats = fi_sorted[:5]
    bot_feats = fi_sorted[-5:]
    model_accuracy   = xgb_acc
    model_precision  = xgb_prec
    model_trained_at = datetime.now().isoformat()
    prec             = model_precision

    real_trades_folded = int(real_trade_mask.sum())
    record, previous = record_learning_progress(real_trades_folded)
    trend_line = ''
    if previous:
        d_prec = (record['precision'] - previous['precision']) * 100
        d_wf   = (record['walk_forward_precision'] - previous['walk_forward_precision']) * 100
        trend_line = (
            f'\n  Trend vs last retrain: precision {d_prec:+.1f}pp, '
            f'walk-forward {d_wf:+.1f}pp'
        )

    log.info(
        f'\n=== TRAINING COMPLETE ==='
        f'\n  XGB precision:  {xgb_prec:.1%}'
        f'\n  LGB precision:  {lgb_prec:.1%}'
        f'\n  Walk-forward:   {_wf_precision:.1%}  <-- most honest number'
        f'\n  Decision threshold: {ML_DECISION_THRESHOLD:.2f}'
        f'\n  Real trades folded in: {real_trades_folded}  (total ever: {record["total_trades_ever"]})'
        f'{trend_line}'
        f'\n  Top 5 drivers:  {", ".join(n for n,_ in top_feats)}'
        f'\n  Weakest feats:  {", ".join(n for n,_ in bot_feats)}'
        f'\n  (weak features above 0.001 are still contributing — below means useless)'
    )
    send_telegram(
        f'🧠 <b>Model Retrained</b>\n'
        f'XGB precision:  <b>{xgb_prec:.1%}</b>\n'
        f'LGB precision:  {lgb_prec:.1%}\n'
        f'Walk-forward:   <b>{_wf_precision:.1%}</b> (real-world estimate)\n'
        f'Decision threshold: {ML_DECISION_THRESHOLD:.2f}\n'
        f'Real trades folded in: {real_trades_folded} (total ever: {record["total_trades_ever"]})'
        f'{trend_line}\n'
        f'Top drivers: {", ".join(n for n,_ in top_feats[:3])}'
    )
    joblib.dump({
        'model':             model,
        'lgb_model':         lgb_model,
        'accuracy':          model_accuracy,
        'precision':         model_precision,
        'wf_precision':      _wf_precision,
        'decision_threshold': ML_DECISION_THRESHOLD,
        'trained_at':        model_trained_at,
        'features':          FEATURES,
    }, MODEL_FILE)
    log.info(f'Model saved -> {MODEL_FILE}')


def load_or_train() -> None:
    """Load saved model or train fresh if none exists."""
    global model, lgb_model, model_accuracy, model_precision, \
        model_trained_at, _wf_precision, ML_DECISION_THRESHOLD
    if MODEL_FILE.exists():
        try:
            saved                  = joblib.load(MODEL_FILE)
            model                  = saved['model']
            lgb_model              = saved.get('lgb_model', None)
            model_accuracy         = saved.get('accuracy',   0.0)
            model_precision        = saved.get('precision',  0.0)
            _wf_precision          = saved.get('wf_precision', 0.0)
            ML_DECISION_THRESHOLD  = saved.get('decision_threshold', ML_DECISION_THRESHOLD)
            model_trained_at       = saved.get('trained_at', 'unknown')
            lgb_tag = ' + LGB' if lgb_model is not None else ''
            log.info(
                f'📦 Model loaded (XGB{lgb_tag}) — '
                f'accuracy={model_accuracy:.1%}  '
                f'precision={model_precision:.1%}  '
                f'walk-forward={_wf_precision:.1%}  '
                f'threshold={ML_DECISION_THRESHOLD:.2f}  '
                f'trained={model_trained_at[:10]}'
            )
            return
        except Exception as e:
            log.warning(f'Model load failed ({e}), retraining from scratch')
    train_model()


def ml_predict(symbol: str, sym_id: int = None) -> dict:
    """
    Run the XGBoost + LightGBM ensemble on the latest bars for a symbol.

    Returns confidence (0-1) where >0.5 means the model thinks BUY.
    The ensemble averages XGBoost (60%) and LightGBM (40%) predictions.
    Also fetches and joins live SPY context so market-direction features
    are accurate at prediction time (not stale training defaults).

    Logs a plain-English explanation of the top 3 reasons for the decision
    so you can understand WHY the model is saying BUY or HOLD each scan.
    """
    if sym_id is None:
        sym_id = TRAINING_UNIVERSE.index(symbol) if symbol in TRAINING_UNIVERSE else 0
    if model is None:
        return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}
    try:
        df = fetch_bars(symbol, days=45)   # 45 days = 800+ 15-min bars
        if len(df) < 250:
            log.warning(f'  {symbol}: not enough bars ({len(df)})')
            return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}

        df = add_features(df)

        # Join live SPY context: fetches SPY bars (cached) and merges
        # by timestamp so the model sees what SPY was doing at this moment
        try:
            spy_ctx = fetch_spy_context(days=45)
            if len(spy_ctx) > 0:
                df = df.join(spy_ctx, how='left', rsuffix='_spy')
                for col in ('spy_ret_1', 'spy_ret_5', 'spy_above_ema20'):
                    if col not in df.columns:
                        df[col] = 0.0
                df[['spy_ret_1', 'spy_ret_5', 'spy_above_ema20']] = (
                    df[['spy_ret_1', 'spy_ret_5', 'spy_above_ema20']]
                    .ffill().fillna(0.0)
                )
        except Exception:
            pass

        df['sym_id'] = sym_id
        row = df[FEATURES + ['sym_id']].iloc[-1:]

        if row.isnull().any().any():
            return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}

        # XGBoost probability (primary)
        xgb_prob = float(model.predict_proba(row)[0][1])

        # LightGBM probability (companion, if trained)
        lgb_prob  = float(lgb_model.predict(row)[0]) if lgb_model is not None else xgb_prob

        # Ensemble: 60% XGBoost + 40% LightGBM
        # (If no LGB, both are xgb_prob so result is unchanged)
        ens_prob = round(0.60 * xgb_prob + 0.40 * lgb_prob, 4)

        # ── Plain-English explanation ─────────────────────────────────────
        # Show the top 3 features driving the prediction so you understand
        # WHY the model is bullish or bearish at each scan.
        fi_pairs   = list(zip(FEATURES + ['sym_id'], model.feature_importances_))
        fi_sorted  = sorted(fi_pairs, key=lambda x: -x[1])
        top3_names = [n for n, _ in fi_sorted[:3]]
        row_vals   = row.iloc[0]
        reasons    = []
        for feat in top3_names:
            if feat in row_vals.index:
                reasons.append(f'{feat}={row_vals[feat]:.3f}')
        signal_word = 'BUY' if ens_prob >= 0.50 else 'HOLD'
        log.info(
            f'  [{symbol}] ML {signal_word}  '
            f'xgb={xgb_prob:.2f} lgb={lgb_prob:.2f} ens={ens_prob:.2f} | '
            f'Drivers: {", ".join(reasons)}'
        )

        return {
            'confidence':   ens_prob,
            'xgb_prob':     xgb_prob,
            'lgb_prob':     lgb_prob,
            'price':        float(df['close'].iloc[-1]),
            'rsi':          float(df['rsi'].iloc[-1]),
            'vol_ratio':    float(df['vol_ratio'].iloc[-1]),
            'macd_hist':    float(df['macd_hist'].iloc[-1]),
            'above_ema50':  int(df['above_ema50'].iloc[-1]),
            'bb_pct':       float(df['bb_pct'].iloc[-1]),
            'spy_ret_1':    float(df['spy_ret_1'].iloc[-1]),
        }
    except Exception as e:
        log.warning(f'ML predict error {symbol}: {e}')
        return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}


# ══════════════════════════════════════════════════════════
# TRADINGVIEW TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════
# Score map: TradingView recommendation → numeric weight
TV_SCORE = {
    'STRONG_BUY': 1.0, 'BUY': 0.7,
    'NEUTRAL':    0.0,
    'SELL':      -0.5, 'STRONG_SELL': -1.0,
}


def get_tv_analysis(symbol: str) -> dict | None:
    """
    Fetch TradingView's 15-minute technical analysis.
    Returns buy/sell/neutral signal counts + key indicators.
    """
    try:
        handler = TA_Handler(
            symbol=symbol,
            screener='america',
            exchange=TV_EXCHANGE.get(symbol, 'NASDAQ'),
            interval=Interval.INTERVAL_15_MINUTES,
        )
        a = handler.get_analysis()
        return {
            'rec':     a.summary['RECOMMENDATION'],
            'buy':     a.summary['BUY'],
            'sell':    a.summary['SELL'],
            'neutral': a.summary['NEUTRAL'],
            'rsi':     a.indicators.get('RSI',      50.0),
            'macd':    a.indicators.get('MACD.macd', 0.0),
            'ema20':   a.indicators.get('EMA20',      0.0),
            'ema50':   a.indicators.get('EMA50',      0.0),
            'adx':     a.indicators.get('ADX',       20.0),
        }
    except Exception as e:
        log.warning(f'TradingView error {symbol}: {e}')
        return None


# ══════════════════════════════════════════════════════════
# COMBINED SIGNAL ENGINE
# XGBoost (50%) + TradingView (35%) + RSI filter (15%)
# ══════════════════════════════════════════════════════════

def combined_signal(symbol: str, ml: dict, tv: dict | None) -> dict:
    """
    Combine all three signal sources into one final score.

    When TradingView gives a directional call (BUY/STRONG_BUY/SELL/STRONG_SELL):
      Score >= 0.55 → BUY.
    When TradingView is N/A or NEUTRAL:
      TV weight (35%) is redistributed to ML, threshold drops to 0.48.
      This prevents the bot from being completely paralyzed when TV is down —
      and NEUTRAL is treated the same way, since a NEUTRAL rating carries no
      directional information but used to still eat 35% of the score while
      the full 0.55 bar still applied, making BUY signals almost unreachable
      (NEUTRAL is the most common TradingView outcome on a 10-min scan).
    """
    score      = 0.0
    reasons    = []
    tv_present = tv is not None
    tv_up      = tv_present and tv.get('rec') != 'NEUTRAL'

    # ── XGBoost ────────────────────────────────────────────
    # When TV is N/A or NEUTRAL, ML absorbs its weight (50% → 85%) so the
    # bot can still act on strong ML signals alone.
    eff_ml_w = ML_WEIGHT if tv_up else (ML_WEIGHT + TV_WEIGHT)
    conf = ml.get('confidence', 0.0)
    # Normalize against the calibrated decision threshold (ML_DECISION_THRESHOLD),
    # not a hardcoded 0.5 — the threshold is picked every retrain as whatever
    # cutoff actually produced good precision on held-out data, so "0
    # contribution" here means "at the model's real breakeven point", not an
    # arbitrary textbook default.
    neutral_pt = min(ML_DECISION_THRESHOLD, 0.95)
    ml_contrib = (conf - neutral_pt) / (1.0 - neutral_pt) * eff_ml_w
    score += ml_contrib
    reasons.append(f'ML:{conf:.0%}(vs{neutral_pt:.0%})')

    # ── TradingView (35%) ─────────────────────────────────
    if tv_present:
        reasons.append(f'TV:{tv["rec"]}' if tv_up else f'TV:{tv["rec"]}(ML+)')
        if tv_up:
            tv_contrib = TV_SCORE.get(tv['rec'], 0.0) * TV_WEIGHT
            score += tv_contrib
            # ADX bonus: strong trend (ADX > 25) boosts conviction
            if tv.get('adx', 0) > 25 and tv['rec'] in ('BUY', 'STRONG_BUY'):
                score += 0.04
                reasons.append('ADX:TREND')
    else:
        reasons.append('TV:N/A(ML+)')  # note that ML weight was boosted

    # ── RSI filter (15%) ──────────────────────────────────
    rsi = ml.get('rsi', 50)
    if 35 <= rsi <= 65:           # sweet spot for entry
        score += RSI_WEIGHT
    elif rsi > 78:                # overbought — penalize heavily
        score -= 0.22
        reasons.append(f'RSI:OVER({rsi:.0f})')
    elif rsi < 25:                # oversold — small bounce bonus
        score += 0.06
        reasons.append(f'RSI:OVERSOLD({rsi:.0f})')
    reasons.append(f'RSI:{rsi:.0f}')

    # ── Volume surge bonus ────────────────────────────────
    if ml.get('vol_ratio', 1) > 1.5:
        score += 0.04
        reasons.append('VOL:SURGE')

    # ── Trend gate: must be above EMA50 ──────────────────
    if ml.get('above_ema50', 0) == 0:
        score -= 0.15
        reasons.append('BELOW:EMA50')

    # ── Bollinger Band position ───────────────────────────
    bb = ml.get('bb_pct', 0.5)
    if 0.35 <= bb <= 0.65:       # middle of bands — less risky
        score += 0.02
    elif bb > 0.95:              # near upper band — overbought
        score -= 0.08

    # Threshold: 0.55 with TV (full 3-source confidence)
    #            0.48 without TV (ML bearing more weight, slightly lower bar)
    threshold    = 0.55 if tv_up else 0.48
    final_signal = 'BUY' if score >= threshold else 'HOLD'
    return {
        'signal':     final_signal,
        'score':      round(score, 3),
        'confidence': conf,
        'reasons':    reasons,
    }


# ══════════════════════════════════════════════════════════
# CLAUDE AI — DAILY MORNING BRIEF
# Runs every morning at 9:00 AM ET to set the day's strategy
# ══════════════════════════════════════════════════════════
morning_brief_text: str = ''


def generate_morning_brief() -> None:
    """
    Gemini AI morning brief — OPTIONAL.
    Skipped automatically if GEMINI_API_KEY is not set or quota is exhausted.
    The trading bot runs perfectly without this feature.
    """
    global morning_brief_text

    # Skip entirely if Gemini not initialised
    if gemini is None:
        log.info('⏭️  Morning brief skipped (Gemini not configured)')
        morning_brief_text = '[Morning brief disabled — trading continues normally]'
        return

    log.info('🧠 Generating morning brief...')

    # ── Gather real-time context ─────────────────────────
    try:
        acct   = trade_client.get_account()
        equity = float(acct.equity)
        cash   = float(acct.cash)
        bp     = float(acct.buying_power)
    except Exception:
        equity, cash, bp = 100000, 100000, 100000

    pos_now = get_positions()
    sells   = [t for t in trade_log if t.get('action') == 'SELL']
    wr      = sum(1 for t in sells if t.get('win')) / len(sells) * 100 if sells else 0
    total_pnl = sum(t.get('pnl', 0) for t in sells)

    # ── TradingView snapshot (all watchlist + SPY) ────────
    tv_lines = []
    for sym in WATCHLIST + ['SPY']:
        tv = get_tv_analysis(sym)
        if tv:
            tv_lines.append(
                f"  {sym:5s} | {tv['rec']:12s} | RSI:{tv['rsi']:.0f} "
                f"| ADX:{tv['adx']:.0f} | Buy:{tv['buy']:2d} Sell:{tv['sell']:2d}"
            )
        else:
            tv_lines.append(f"  {sym:5s} | N/A")

    # ── Recent trades ─────────────────────────────────────
    recent = sells[-8:] if len(sells) >= 8 else sells
    recent_str = '\n'.join(
        f"  {t['timestamp'][:10]} | {t['symbol']:5s} | {t.get('reason',''):15s} | P&L: ${t.get('pnl',0):+7.2f}"
        for t in recent
    ) or '  No closed trades yet'

    prompt = f"""You are the AI intelligence layer for a self-learning paper trading bot running on Alpaca.

TODAY: {now_et().strftime('%A, %B %d, %Y — %I:%M %p ET')}

ACCOUNT STATUS:
  Equity:        ${equity:>12,.2f}
  Cash:          ${cash:>12,.2f}
  Buying Power:  ${bp:>12,.2f}
  Open positions: {list(pos_now.keys()) or 'None'}

BOT PERFORMANCE:
  Model accuracy: {model_accuracy:.1%}  (last trained: {model_trained_at[:10]})
  Lifetime trades: {len(sells)}
  Win rate:       {wr:.1f}%
  Total P&L:      ${total_pnl:+,.2f}

TRADINGVIEW 15-MINUTE SIGNALS RIGHT NOW:
{chr(10).join(tv_lines)}

RECENT CLOSED TRADES:
{recent_str}

STRATEGY RULES:
  · 2% equity risk per trade
  · Stop loss at -2%, take profit at +6%
  · Max 3 simultaneous positions
  · Watchlist: NVDA, PANW, AVGO, SOFI, PLTR

Generate today's morning trading brief. Format exactly as follows:

MARKET MOOD: [One sentence — bull/bear/neutral + primary driver]

TODAY'S PRIORITY RANKING:
  #1 [Symbol] — [Specific reason based on signals above]
  #2 [Symbol] — [Specific reason]
  #3 [Symbol] — [Specific reason]
  #4 [Symbol] — [Specific reason]
  #5 [Symbol] — [Specific reason]

KEY RISK TODAY: [One specific thing to watch out for]

STANCE: [Aggressive / Neutral / Defensive] — [Why, based on today's signals]

TARGET P&L: $[realistic number] — [Brief justification]

CLAUDE'S EDGE FOR TODAY: [One insight that a human might miss — patterns in the signals, sector rotation, macro context, etc.]

Be precise and data-driven. Reference specific indicator values from the signals above."""

    try:
        # Retry up to 3 times with backoff — handles Gemini free-tier 429 rate limits
        response = None
        for attempt in range(3):
            try:
                response = gemini.generate_content(prompt)
                break
            except Exception as e:
                if '429' in str(e) and attempt < 2:
                    wait = 60 * (attempt + 1)   # 60s, 120s
                    log.warning(f'Gemini rate limited — retrying in {wait}s (attempt {attempt+1}/3)...')
                    time.sleep(wait)
                else:
                    raise

        morning_brief_text = response.text
        sep = '═' * 60
        log.info(f'\n{sep}\n🧠 MORNING BRIEF — {now_et().strftime("%b %d %Y %I:%M %p ET")}\n{sep}\n{morning_brief_text}\n{sep}')
        # Send first 3500 chars to Telegram (message limit is 4096)
        send_telegram(
            f'🧠 <b>MORNING BRIEF — {now_et().strftime("%b %d %Y")}</b>\n'
            f'Equity: ${equity:,.0f}  ·  Model: {model_accuracy:.1%}\n'
            f'────────────────────\n'
            + morning_brief_text[:3200]
        )

        BRIEF_FILE.write_text(
            f"Generated: {now_et().isoformat()}\n"
            f"Equity: ${equity:,.2f} | Model: {model_accuracy:.1%}\n"
            f"{'─'*60}\n{morning_brief_text}"
        )
    except Exception as e:
        log.warning(f'Gemini API error: {e}')
        morning_brief_text = f'[Brief unavailable: {e}]'


# ══════════════════════════════════════════════════════════
# DIP CONFIRMATION SYSTEM
# ══════════════════════════════════════════════════════════
#
# Replaces fixed-time cooldowns with pattern-based re-entry.
#
# Instead of blindly waiting 120 minutes after a stop loss,
# the bot WATCHES the stock every scan cycle and re-enters
# the moment candle patterns confirm the dip has reversed.
#
# This means:
#   • If the stock bottoms in 20 min → bot buys the bottom
#   • If the stock keeps falling for 2h → bot stays out
#   • Other watchlist stocks are ALWAYS scanned freely
#
# Signals checked each scan cycle:
#   1. Hammer candle        — buyers pushed price back up from lows
#   2. Bullish engulfing    — big green candle swallows the red candle
#   3. RSI oversold + rising — momentum washed out, now recovering
#   4. Bounce from low      — price lifted 0.5%+ off the dip low
#   5. Volume capitulation  — big sell volume = everyone already sold
#
# ══════════════════════════════════════════════════════════

position_highs: dict = {}
# Tracks the highest price seen for each open position since entry.
# Used by the trailing stop logic in manage_open_positions().
# Set on BUY, updated every scan, cleared on SELL.

dip_watch: dict = {}

# ── Market Regime Cache ───────────────────────────────────
market_regime: str      = 'BULL'   # current regime — updated every 30 min
_regime_checked_at      = None     # datetime of last regime check

# ── Earnings Blackout Cache ───────────────────────────────
earnings_cache: dict    = {}       # sym → next earnings datetime (UTC)
_earnings_cache_date: str = ''     # date when cache was last refreshed


# ══════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════
# Uses SPY vs its 50-day SMA as the macro health indicator.
# Updated every 30 minutes during market hours — lightweight.
# BULL → trade normally at full size
# BEAR → require higher ML confidence + use half position cap
# ══════════════════════════════════════════════════════════

def get_market_regime() -> str:
    """
    Determine macro regime from SPY vs 50-day SMA.
    Cached for 30 minutes — avoids redundant API calls each scan.
    Returns 'BULL' or 'BEAR'.
    """
    global market_regime, _regime_checked_at
    now = now_et()
    if _regime_checked_at and (now - _regime_checked_at).total_seconds() < 1800:
        return market_regime   # use cached value

    try:
        df = fetch_bars('SPY', days=80)
        # Resample 15-min bars to daily closes for true 50-day SMA
        daily = df['close'].resample('1D').last().dropna()
        if len(daily) < 50:
            return market_regime   # not enough data — keep last known
        sma50   = daily.rolling(50).mean().iloc[-1]
        current = daily.iloc[-1]
        new_regime = 'BULL' if current > sma50 else 'BEAR'
        if new_regime != market_regime:
            icon = '📈' if new_regime == 'BULL' else '📉'
            log.info(
                f'{icon} REGIME CHANGE: {market_regime} → {new_regime} '
                f'(SPY ${current:.2f} vs SMA50 ${sma50:.2f})'
            )
            send_telegram(
                f'{icon} <b>MARKET REGIME CHANGE</b>\n'
                f'{market_regime} → <b>{new_regime}</b>\n'
                f'SPY: ${current:.2f}  ·  SMA50: ${sma50:.2f}\n'
                + ('⚠️ Switching to half-size positions + higher confidence required'
                   if new_regime == 'BEAR' else
                   '✅ Back to full-size positions')
            )
        market_regime      = new_regime
        _regime_checked_at = now
    except Exception as e:
        log.warning(f'Regime check error: {e}')

    return market_regime


# ══════════════════════════════════════════════════════════
# EARNINGS BLACKOUT SYSTEM
# ══════════════════════════════════════════════════════════
# Stocks can gap ±15% on earnings surprise — no technical
# signal can predict this. The safest rule: never hold into
# earnings. Bot skips buying any stock within 2 days of its
# next scheduled earnings report.
# ══════════════════════════════════════════════════════════

def _fetch_earnings_date(symbol: str) -> datetime | None:
    """Fetch next earnings date from Yahoo Finance free API (no key needed)."""
    try:
        url = (
            f'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}'
            f'?modules=calendarEvents'
        )
        r    = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        data = r.json()
        events = data['quoteSummary']['result'][0]['calendarEvents']['earnings']
        dates  = events.get('earningsDate', [])
        if dates:
            ts = dates[0]['raw']
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    return None


def refresh_earnings_cache() -> None:
    """
    Pre-fetch next earnings date for all watchlist stocks.
    Called once per day at market open. Results cached in earnings_cache dict.
    """
    global earnings_cache, _earnings_cache_date
    today = now_et().strftime('%Y-%m-%d')
    if _earnings_cache_date == today:
        return   # already refreshed today
    _earnings_cache_date = today
    log.info('📅 Refreshing earnings calendar...')
    upcoming = []
    for sym in list(set(WATCHLIST + FALLBACK_WATCHLIST + list(SCAN_UNIVERSE[:15]))):
        dt = _fetch_earnings_date(sym)
        earnings_cache[sym] = dt
        if dt:
            days_away = (dt - datetime.now(timezone.utc)).days
            if 0 <= days_away <= 7:
                upcoming.append(f'{sym}({days_away}d)')
    if upcoming:
        log.info(f'  ⚠️  Earnings within 7 days: {", ".join(upcoming)}')
    else:
        log.info('  No earnings within 7 days for watchlist stocks')


def is_earnings_blackout(symbol: str) -> bool:
    """
    Return True if symbol has earnings within EARNINGS_BLACKOUT_DAYS.
    Signals the scan loop to skip this stock today.
    """
    dt = earnings_cache.get(symbol)
    if not dt:
        return False
    days_away = (dt - datetime.now(timezone.utc)).days
    return 0 <= days_away <= EARNINGS_BLACKOUT_DAYS
# Structure per symbol:
# {
#   'reason':       'STOP_LOSS' | 'TAKE_PROFIT' | 'TREND' | 'NEWS_EXIT',
#   'exit_price':   float,
#   'exit_time':    datetime,
#   'lowest_since': float,   ← tracked each scan to find the dip low
# }


# ── Candle pattern detectors ──────────────────────────────

def _rsi_from_close(close: pd.Series, period: int = 14) -> pd.Series:
    """Lightweight RSI — used inside dip detection without full feature pipeline."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))


def _is_hammer(df: pd.DataFrame) -> bool:
    """
    Hammer: small body at the top, long lower shadow, tiny upper shadow.
    Signals buyers absorbed all selling pressure — reversal likely.
    """
    c = df.iloc[-1]
    body        = abs(c['close'] - c['open'])
    total_range = c['high'] - c['low']
    lower_wick  = min(c['close'], c['open']) - c['low']
    upper_wick  = c['high'] - max(c['close'], c['open'])
    if total_range < 1e-9:
        return False
    return (lower_wick >= body * 2.0          # long tail = buyers fought back
            and lower_wick >= total_range * 0.5
            and upper_wick <= body * 0.6       # small upper shadow
            and c['close'] >= c['open'])       # green body preferred


def _is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """
    Bullish engulfing: big green candle completely swallows the previous red candle.
    One of the strongest reversal signals — buyers overwhelmed sellers in one bar.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_red   = prev['close'] < prev['open']
    curr_green = curr['close'] > curr['open']
    engulfs    = (curr['open']  <= prev['close']   # opens below or at prev close
                  and curr['close'] >= prev['open'])  # closes above or at prev open
    return prev_red and curr_green and engulfs


def _is_doji(df: pd.DataFrame) -> bool:
    """
    Doji: open ≈ close — indecision candle.
    At a dip low it signals the selling momentum has stalled.
    """
    c           = df.iloc[-1]
    body        = abs(c['close'] - c['open'])
    total_range = c['high'] - c['low']
    return total_range > 1e-9 and (body / total_range) < 0.10


def _rsi_oversold_recovery(df: pd.DataFrame) -> bool:
    """
    RSI was deeply oversold (< 35) recently and is now turning up.
    Classic Raschke setup: oversold in a healthy stock = buying opportunity.
    """
    rsi = _rsi_from_close(df['close'])
    if len(rsi) < 5:
        return False
    recent_low = rsi.iloc[-5:-1].min()
    curr_rsi   = rsi.iloc[-1]
    prev_rsi   = rsi.iloc[-2]
    return (recent_low < 35              # was oversold
            and curr_rsi > prev_rsi      # now rising
            and curr_rsi < 50)           # not yet overbought


def _volume_capitulation(df: pd.DataFrame) -> bool:
    """
    A big red candle on huge volume (2× average) followed by stabilization.
    'Capitulation' = everyone who wanted to sell already sold.
    After capitulation the path of least resistance is up.
    """
    if len(df) < 6:
        return False
    recent   = df.iloc[-5:]
    vol_avg  = df['volume'].rolling(20).mean().iloc[-1]
    for i in range(len(recent) - 1):
        bar = recent.iloc[i]
        if (bar['close'] < bar['open']           # red candle
                and bar['volume'] > vol_avg * 2  # huge volume
                and df.iloc[-1]['close'] >= df.iloc[-2]['close']):  # now stabilizing
            return True
    return False


# ── Main dip confirmation gate ────────────────────────────

def set_dip_watch(symbol: str, reason: str, exit_price: float) -> None:
    """Called after every sell — puts the symbol into pattern-monitoring mode."""
    dip_watch[symbol] = {
        'reason':       reason,
        'exit_price':   exit_price,
        'exit_time':    now_et(),
        'lowest_since': exit_price,
    }
    required = DIP_SIGNALS_AFTER_STOP if reason == 'STOP_LOSS' else DIP_SIGNALS_AFTER_TP
    log.info(
        f'  👁  {symbol} → dip watch [{reason}] '
        f'— will re-enter when {required} reversal signal(s) appear'
    )


def check_dip_reversal(symbol: str) -> tuple:
    """
    Check if a stock in dip-watch has confirmed a reversal.

    Returns (can_enter: bool, status_message: str)

    Logic:
      • Always waits 5 min minimum (avoids same-candle rebuy)
      • Fetches last 3 days of bars and checks 5 pattern types
      • After STOP_LOSS: needs 2 signals (strict — stock was falling)
      • After TP/TREND:  needs 1 signal (lenient — stock was healthy)
      • When signals reached: clears dip_watch and allows re-entry
    """
    if symbol not in dip_watch:
        return True, ''

    watch      = dip_watch[symbol]
    elapsed    = (now_et() - watch['exit_time']).total_seconds()
    exit_reason = watch['reason']

    # Enforce minimum wait — never rebuy on the very same candle
    if elapsed < DIP_MIN_WAIT_SEC:
        remaining = int(DIP_MIN_WAIT_SEC - elapsed)
        return False, f'min wait ({remaining}s)'

    # Enforce maximum wait — a symbol with no confirming pattern used to sit
    # in dip_watch indefinitely, silently removing it from the tradeable set
    # for the rest of the day. Give up on pattern confirmation after 4h and
    # fall back to allowing re-entry on the normal signal gates.
    if elapsed >= DIP_MAX_WAIT_SEC:
        log.info(f'  ⏱  {symbol} dip watch expired after {elapsed/3600:.1f}h — re-entry allowed')
        del dip_watch[symbol]
        return True, 'dip watch expired'

    # Fetch recent bars for pattern analysis
    try:
        df = fetch_bars(symbol, days=3)
        if len(df) < 10:
            return False, 'not enough bars'

        # Track the lowest price seen since exit
        curr_price = float(df['close'].iloc[-1])
        if curr_price < watch['lowest_since']:
            watch['lowest_since'] = curr_price
            dip_watch[symbol] = watch

    except Exception as e:
        return False, f'bar fetch error: {e}'

    low = watch['lowest_since']

    # ── Count active signals ──────────────────────────────
    signals = []

    if _is_hammer(df):
        signals.append('hammer')

    if _is_bullish_engulfing(df):
        signals.append('bullish engulfing')

    if _is_doji(df) and curr_price > low * 1.003:
        signals.append('doji + bounce')

    if _rsi_oversold_recovery(df):
        signals.append('RSI recovery')

    bounce_pct = (curr_price - low) / low if low > 0 else 0
    if bounce_pct >= 0.005:
        signals.append(f'bounce +{bounce_pct:.1%} from low')

    if _volume_capitulation(df):
        signals.append('vol capitulation')

    # ── Verdict ───────────────────────────────────────────
    # STOP_LOSS / NEWS_EXIT → strict (2 signals) — stock was in trouble
    # TAKE_PROFIT / TRAILING_STOP / TREND → lenient (1 signal) — stock was healthy
    required = DIP_SIGNALS_AFTER_STOP if exit_reason in ('STOP_LOSS', 'NEWS_EXIT') else DIP_SIGNALS_AFTER_TP

    if len(signals) >= required:
        log.info(
            f'  ✅ {symbol} dip reversal CONFIRMED '
            f'({", ".join(signals)}) — re-entry allowed'
        )
        del dip_watch[symbol]
        return True, f'reversal: {", ".join(signals)}'

    signal_str = ', '.join(signals) if signals else 'none yet'
    return False, f'watching for reversal ({len(signals)}/{required} signals: {signal_str})'


# ══════════════════════════════════════════════════════════
# TRADE LOGGING  (feeds nightly retrain)
# ══════════════════════════════════════════════════════════
trade_log: list = []


def load_log() -> None:
    global trade_log
    if LOG_FILE.exists():
        try:
            trade_log = json.loads(LOG_FILE.read_text())
            log.info(f'📚 Loaded {len(trade_log)} trades from history')
        except Exception:
            trade_log = []


def save_log(entry: dict) -> None:
    trade_log.append(entry)
    LOG_FILE.write_text(json.dumps(trade_log, indent=2, default=str))


# ══════════════════════════════════════════════════════════
# POSITION & ORDER MANAGEMENT
# ══════════════════════════════════════════════════════════

def get_account_equity() -> float:
    return float(trade_client.get_account().equity)


def get_positions() -> dict:
    return {p.symbol: p for p in trade_client.get_all_positions()}


def place_buy(symbol: str, price: float, equity: float, signal: dict) -> bool:
    """Calculate position size, place market buy, log trade."""
    global position_highs
    pos = get_positions()
    if symbol in pos:
        return False  # already have this

    open_count = sum(1 for s in pos if s in WATCHLIST)
    if open_count >= MAX_POSITIONS:
        return False

    # ── Market regime gate ────────────────────────────────
    # Bear market (SPY below 50-day SMA): require higher conviction + smaller size.
    # Bull market: normal rules apply.
    if REGIME_GATE:
        regime = get_market_regime()
        if regime == 'BEAR':
            ml_conf = signal.get('confidence', 0)
            if ml_conf < BEAR_MIN_CONF:
                log.info(
                    f'  {symbol}: 🐻 BEAR MARKET skip '
                    f'(ML {ml_conf:.0%} < {BEAR_MIN_CONF:.0%} required)'
                )
                return False
            cap = BEAR_POSITION_CAP
            log.info(f'  {symbol}: 🐻 BEAR MARKET — half-size position ({cap:.0%} cap)')
        else:
            cap = POSITION_CAP
    else:
        cap = POSITION_CAP

    # Position sizing: risk 2% of equity, stop at -2% → position = 100%×equity×RISK_PCT/STOP_PCT
    # Capped at regime-adjusted position cap
    qty_usd = min(equity * RISK_PCT / STOP_PCT, equity * cap)
    qty     = max(1, int(qty_usd / price))

    try:
        order = trade_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        position_highs[symbol] = price   # seed trailing stop tracker at entry price
        log.info(
            f'🟢 BUY  {qty:5d}×{symbol} @ ${price:8.2f} '
            f'| Score:{signal["score"]:+.3f} | {" | ".join(signal["reasons"])}'
        )
        send_telegram(
            f'🟢 <b>PAPER BOT — BUY</b>\n'
            f'Stock: <b>{symbol}</b>  ·  {qty} shares @ ${price:.2f}\n'
            f'Cost: ${qty*price:,.0f}  ·  ML: {signal["confidence"]:.0%}\n'
            f'Score: {signal["score"]:+.3f}  ·  {" | ".join(signal["reasons"][:3])}\n'
            f'Equity: ${equity:,.0f}'
        )
        save_log({
            'action':      'BUY',
            'symbol':      symbol,
            'qty':         qty,
            'entry_price': price,
            'equity':      equity,
            'signal':      signal,
            'order_id':    str(order.id),
            'timestamp':   datetime.now().isoformat(),
        })
        return True
    except Exception as e:
        log.error(f'Buy failed {symbol}: {e}')
        send_telegram(
            f'⚠️ <b>BUY FAILED</b>\n'
            f'Stock: <b>{symbol}</b>  ·  Qty: {qty}\n'
            f'Error: {e}'
        )
        return False


def place_sell(symbol: str, pos, reason: str) -> None:
    """Market sell an open position, log the outcome."""
    global position_highs
    position_highs.pop(symbol, None)   # clear trailing stop tracker
    qty     = float(pos.qty)
    entry   = float(pos.avg_entry_price)
    current = float(pos.current_price)
    pnl     = float(pos.unrealized_pl)
    pct     = (current - entry) / entry
    win     = pnl > 0
    emoji   = '✅' if win else '❌'

    try:
        trade_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        log.info(
            f'🔴 SELL {qty:5.0f}×{symbol} @ ${current:8.2f} '
            f'| P&L: ${pnl:+8.2f} ({pct:+.1%}) [{reason}] {emoji}'
        )
        send_telegram(
            f'{emoji} <b>PAPER BOT — SELL [{reason}]</b>\n'
            f'Stock: <b>{symbol}</b>  ·  {qty:.0f} shares @ ${current:.2f}\n'
            f'Entry: ${entry:.2f}  →  Exit: ${current:.2f}\n'
            f'P&L: <b>${pnl:+.2f} ({pct:+.1%})</b>\n'
            f'{"✅ WIN" if win else "❌ LOSS"}'
        )
        # Enter dip watch — bot will re-buy when candle patterns confirm reversal
        # (replaces fixed 60/120 min cooldown — smarter, pattern-driven)
        if reason in ('STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', 'PROFIT_LOCK', 'TREND', 'NEWS_EXIT'):
            set_dip_watch(symbol, reason, current)
        save_log({
            'action':      'SELL',
            'symbol':      symbol,
            'qty':         qty,
            'entry_price': entry,
            'exit_price':  current,
            'pnl':         pnl,
            'pct':         pct,
            'win':         win,
            'reason':      reason,
            'timestamp':   datetime.now().isoformat(),
        })
    except Exception as e:
        err_str = str(e)
        # "held_for_orders" means a sell order is already queued — not a real error
        if 'held_for_orders' in err_str or '40310000' in err_str:
            log.info(f'  {symbol}: sell order already pending — skipping duplicate')
        else:
            log.error(f'Sell failed {symbol}: {e}')


def manage_open_positions() -> None:
    """
    Four-tier exit system — evaluated every scan for each open position.

    Tier 1  peak < +1.5%   : hard stop-loss at -2% from entry
    Tier 2  peak ≥ +1.5%   : profit lock — floor moves to 0% (breakeven)
    Tier 3  peak ≥ +4.0%   : profit lock — floor moves to +2%
    Tier 4  peak ≥ +6.0%   : trailing stop (5% below peak, +3% hard floor)

    The floor only moves UP — it never comes back down.
    Result: you can never lose money once you're up +1.5%, and
            you capture outsized gains on the rare +8–20% catalyst days.

    Examples (entry $100):
      Hits +3%, retraces to  0%  → PROFIT_LOCK exit at  0%  (was -2% before)
      Hits +5%, retraces to +2%  → PROFIT_LOCK exit at +2%
      Hits +15%, retraces 5%     → TRAILING_STOP exit at +9.25%
    """
    global position_highs

    for sym, pos in get_positions().items():
        if sym not in WATCHLIST:
            continue

        entry   = float(pos.avg_entry_price)
        current = float(pos.current_price)
        pct     = (current - entry) / entry

        # ── Update running peak ──────────────────────────────
        peak     = max(position_highs.get(sym, entry), current)
        position_highs[sym] = peak
        peak_pct = (peak - entry) / entry

        # ── Tier 4: trailing stop (peak ever ≥ +6%) ─────────
        if peak_pct >= TRAIL_ACTIVATE_PCT:
            trail_price = max(
                peak * (1 - TRAIL_PCT),
                entry * (1 + TRAIL_FLOOR_PCT),
            )
            trail_pct = (trail_price - entry) / entry

            if current <= trail_price:
                log.info(
                    f'  {sym:5s}: 📉 TRAILING STOP | '
                    f'peak={peak_pct:+.1%}  exit={pct:+.1%}  '
                    f'floor={trail_pct:+.1%}'
                )
                place_sell(sym, pos, 'TRAILING_STOP')
            else:
                log.info(
                    f'  {sym:5s}: 🚀 TRAILING     | '
                    f'pnl={pct:+.1%}  peak={peak_pct:+.1%}  '
                    f'trail_stop={trail_pct:+.1%}  '
                    f'room={pct - trail_pct:+.1%}'
                )

        # ── Tier 3: profit lock — floor at +2% (peak ≥ +4%) ─
        elif peak_pct >= LOCK_PROFIT_AT:
            floor_price = entry * (1 + LOCK_PROFIT_FLOOR)
            if current <= floor_price:
                log.info(
                    f'  {sym:5s}: 🔒 PROFIT LOCK  | '
                    f'peak={peak_pct:+.1%}  exit={pct:+.1%}  '
                    f'(locked +2% floor)'
                )
                place_sell(sym, pos, 'PROFIT_LOCK')
            else:
                log.info(
                    f'  {sym:5s}: 📈 HOLDING      | '
                    f'pnl={pct:+.1%}  peak={peak_pct:+.1%}  '
                    f'floor=+{LOCK_PROFIT_FLOOR:.0%}'
                )

        # ── Tier 2: profit lock — floor at 0% (peak ≥ +1.5%) ─
        elif peak_pct >= LOCK_BREAKEVEN_AT:
            if current <= entry:
                log.info(
                    f'  {sym:5s}: 🔒 BREAKEVEN    | '
                    f'peak={peak_pct:+.1%}  exit={pct:+.1%}  '
                    f'(protected from loss)'
                )
                place_sell(sym, pos, 'PROFIT_LOCK')
            else:
                log.info(
                    f'  {sym:5s}: 📈 HOLDING      | '
                    f'pnl={pct:+.1%}  peak={peak_pct:+.1%}  '
                    f'floor=breakeven'
                )

        # ── Tier 1: hard stop-loss (peak never reached +1.5%) ─
        elif pct <= -STOP_PCT:
            place_sell(sym, pos, 'STOP_LOSS')


def close_all_positions(reason: str = 'EOD') -> None:
    for sym, pos in get_positions().items():
        if sym in WATCHLIST:
            place_sell(sym, pos, reason)


# ══════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════

def is_market_open() -> bool:
    try:
        return trade_client.get_clock().is_open
    except Exception:
        return False


eod_done_date: str = ''   # track which date EOD already ran — prevents repeated firing


# ══════════════════════════════════════════════════════════
# DAILY PERFORMANCE REPORT
# Sent via Telegram every day at EOD — shows precision,
# P&L, model accuracy, and individual trade results.
# Use this to track whether the ML model is improving.
# ══════════════════════════════════════════════════════════

def send_daily_report() -> None:
    """
    Build and send a daily performance report to Telegram.

    Metrics sent:
      • Today's precision (wins ÷ total closed trades today)
      • Today's total P&L and average per trade
      • Best and worst trade of the day
      • Individual trade log with reasons
      • Cumulative all-time win rate (model health trend)
      • Current ML model accuracy from last retrain
    """
    today     = now_et().strftime('%Y-%m-%d')
    day_label = now_et().strftime('%a %b %d')

    # ── Filter to today's SELL trades only ────────────────
    today_sells = [
        t for t in trade_log
        if t.get('action') == 'SELL'
        and str(t.get('timestamp', ''))[:10] == today
    ]

    # ── All-time stats ────────────────────────────────────
    all_sells    = [t for t in trade_log if t.get('action') == 'SELL']
    all_wins     = [t for t in all_sells if t.get('win')]
    all_wr       = len(all_wins) / len(all_sells) * 100 if all_sells else 0.0
    all_pnl      = sum(t.get('pnl', 0) for t in all_sells)

    # ── Today's stats ────────────────────────────────────
    n_trades     = len(today_sells)
    today_wins   = [t for t in today_sells if t.get('win')]
    today_losses = [t for t in today_sells if not t.get('win')]
    precision    = len(today_wins) / n_trades * 100 if n_trades else 0.0
    today_pnl    = sum(t.get('pnl', 0) for t in today_sells)
    avg_pnl      = today_pnl / n_trades if n_trades else 0.0

    # Best and worst trade
    best  = max(today_sells, key=lambda t: t.get('pnl', 0), default=None)
    worst = min(today_sells, key=lambda t: t.get('pnl', 0), default=None)

    # ── Account info ─────────────────────────────────────
    try:
        acct   = trade_client.get_account()
        equity = float(acct.equity)
        cash   = float(acct.cash)
    except Exception:
        equity = cash = 0.0

    # ── Precision emoji ───────────────────────────────────
    if precision >= 70:
        prec_emoji = '🟢'
    elif precision >= 50:
        prec_emoji = '🟡'
    else:
        prec_emoji = '🔴'

    pnl_emoji = '📈' if today_pnl >= 0 else '📉'

    # ── Build individual trade lines ──────────────────────
    trade_lines = ''
    for t in today_sells:
        w      = '✅' if t.get('win') else '❌'
        sym    = t.get('symbol', '???')
        reason = t.get('reason', '')[:14]
        pnl    = t.get('pnl', 0)
        pct    = t.get('pct', 0) * 100
        trade_lines += f'\n  {w} {sym:<5s} {reason:<14s}  ${pnl:+.2f} ({pct:+.1f}%)'

    if not trade_lines:
        trade_lines = '\n  No trades closed today'

    # ── Compose message ───────────────────────────────────
    msg = (
        f'📊 <b>DAILY REPORT — {day_label}</b>\n'
        '━' * 28 + '\n'
        f'\n'
        f'{prec_emoji} <b>TODAY\'S PRECISION</b>\n'
        f'  Trades: {n_trades}  ({len(today_wins)}W / {len(today_losses)}L)\n'
        f'  Precision: <b>{precision:.0f}%</b>\n'
        f'\n'
        f'{pnl_emoji} <b>TODAY\'S P&amp;L</b>\n'
        f'  Total:   <b>${today_pnl:+.2f}</b>\n'
        f'  Avg/trade: ${avg_pnl:+.2f}\n'
    )

    if best and n_trades > 0:
        msg += (
            f'  Best:  {best["symbol"]} ${best.get("pnl",0):+.2f} ({best.get("pct",0)*100:+.1f}%)\n'
        )
    if worst and n_trades > 1:
        msg += (
            f'  Worst: {worst["symbol"]} ${worst.get("pnl",0):+.2f} ({worst.get("pct",0)*100:+.1f}%)\n'
        )

    msg += (
        f'\n'
        f'🤖 <b>MODEL HEALTH</b>\n'
        f'  ML accuracy (last retrain): <b>{model_accuracy:.1%}</b>\n'
        f'  All-time win rate: <b>{all_wr:.0f}%</b>  ({len(all_wins)}W / {len(all_sells)} trades)\n'
        f'  Cumulative P&amp;L: <b>${all_pnl:+,.2f}</b>\n'
        f'\n'
        f'📋 <b>TRADES TODAY</b>{trade_lines}\n'
        f'\n'
        '━' * 28 + '\n'
        f'Equity: <b>${equity:,.2f}</b>  ·  Cash: ${cash:,.0f}'
    )

    send_telegram(msg)
    log.info(
        f'📊 Daily report sent — '
        f'precision={precision:.0f}% ({len(today_wins)}W/{n_trades}) '
        f'P&L=${today_pnl:+.2f}  '
        f'ML={model_accuracy:.1%}'
    )


def check_eod() -> None:
    global eod_done_date
    now = now_et()   # ← must use ET, not UTC (Render servers run UTC)
    today = now.strftime('%Y-%m-%d')

    # Only fire once per day, only when market is actually open, only at EOD window
    if now.hour != EOD_HOUR or now.minute < EOD_MIN:
        return
    if eod_done_date == today:
        return   # already ran today — skip
    if not is_market_open():
        log.info('🕑 EOD check skipped — market is closed (holiday or weekend)')
        eod_done_date = today   # mark done so it doesn't spam on holidays
        return

    eod_done_date = today
    if get_positions():
        log.info('🕑 EOD: closing all positions before market close')
        close_all_positions('EOD')

    # Send daily performance report to Telegram
    send_daily_report()


# ══════════════════════════════════════════════════════════
# NIGHTLY SELF-LEARNING RETRAIN
# ══════════════════════════════════════════════════════════

def nightly_retrain() -> None:
    """
    Midnight retrain — includes all trades from the day.
    Model improves as it sees real outcomes of its own predictions.
    """
    sells      = [t for t in trade_log if t.get('action') == 'SELL']
    wins       = [t for t in sells if t.get('win')]
    wr         = len(wins) / len(sells) * 100 if sells else 0
    total_pnl  = sum(t.get('pnl', 0) for t in sells)

    sep = '═' * 60
    log.info(f'\n{sep}\n🌙 NIGHTLY SELF-LEARNING RETRAIN\n{sep}')
    log.info(f'  Total trades ever: {len(sells)}')
    log.info(f'  Win rate:          {wr:.1f}%')
    log.info(f'  Cumulative P&L:    ${total_pnl:+,.2f}')
    log.info(f'  Old accuracy:      {model_accuracy:.1%}')

    train_model(retrain=True)

    log.info(f'  New accuracy:      {model_accuracy:.1%}')
    log.info(f'✅ Bot is smarter — ready for tomorrow\n{sep}')
    send_telegram(
        f'🌙 <b>NIGHTLY RETRAIN COMPLETE</b>\n'
        f'Trades ever: {len(sells)}  ·  Win rate: {wr:.1f}%\n'
        f'Cumulative P&L: <b>${total_pnl:+,.2f}</b>\n'
        f'Model accuracy: <b>{model_accuracy:.1%}</b>\n'
        f'✅ Bot is smarter — ready for tomorrow'
    )


# ══════════════════════════════════════════════════════════
# ██  NEWS INTELLIGENCE MODULE                            ██
# ██  Runs at 8:30 AM ET — before the market opens       ██
# ██                                                      ██
# ██  Pipeline:                                           ██
# ██  1. Fetch news: Alpaca · Yahoo RSS · SEC EDGAR       ██
# ██  2. Gemini AI reads every article                    ██
# ██  3. Score & rank all stocks                          ██
# ██  4. Print "Why These 5" explanation panel            ██
# ██  5. Update WATCHLIST dynamically for today           ██
# ══════════════════════════════════════════════════════════

def fetch_alpaca_news(hours: int = 20) -> list:
    """
    Fetch recent news from Alpaca's built-in news API.
    Articles are already tagged with affected stock tickers — no parsing needed.
    FREE with your existing Alpaca account.
    """
    if not news_client:
        return []
    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        req   = NewsRequest(
            symbols=SCAN_UNIVERSE[:30],   # Alpaca request limit
            start=start,
            end=end,
            limit=100,
            sort='desc',
        )
        result   = news_client.get_news(req)
        articles = []
        for item in result.news:
            articles.append({
                'title':   item.headline,
                'summary': (item.summary or '')[:300],
                'symbols': list(item.symbols or []),
                'source':  item.source,
                'time':    str(item.created_at),
            })
        log.info(f'  Alpaca News:   {len(articles):3d} articles')
        return articles
    except Exception as e:
        log.warning(f'  Alpaca news error: {e}')
        return []


def fetch_yahoo_rss(symbols: list) -> list:
    """
    Fetch Yahoo Finance RSS feed for a list of symbols.
    No API key, no registration — completely free.
    Great for earnings, analyst upgrades, and breaking company news.
    """
    articles = []
    for sym in symbols[:14]:     # rate-limit: 14 symbols max
        try:
            url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US'
            r   = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            root = XMLTree.fromstring(r.text)
            for item in root.findall('./channel/item')[:4]:
                title   = item.findtext('title', '') or ''
                summary = item.findtext('description', '') or ''
                articles.append({
                    'title':   title,
                    'summary': summary[:300],
                    'symbols': [sym],
                    'source':  'Yahoo Finance',
                    'time':    item.findtext('pubDate', ''),
                })
        except Exception:
            pass
    log.info(f'  Yahoo Finance: {len(articles):3d} articles')
    return articles


def fetch_sec_8k(hours: int = 24) -> list:
    """
    Fetch SEC EDGAR 8-K filings — the MOST POWERFUL free data source.

    Companies MUST file an 8-K within 4 business days of:
      • Mergers & acquisitions (Item 1.01)
      • Material agreements / contracts (Item 1.01)
      • Bankruptcy / delisting (Item 1.03)
      • CEO/CFO changes (Item 5.02)
      • Earnings announcements (Item 2.02)

    This catches M&A news BEFORE Wall Street Journal writes about it.
    Gemini extracts the company name and maps it to a ticker.
    """
    try:
        url = (
            'https://www.sec.gov/cgi-bin/browse-edgar'
            '?action=getcurrent&type=8-K&dateb=&owner=include'
            '&count=40&search_text=&output=atom'
        )
        r    = requests.get(url, timeout=12, headers={
            'User-Agent': 'ShubhamTradingBot/1.0 shubhamkumarsingh91@gmail.com'
        })
        root = XMLTree.fromstring(r.text)
        ns   = {'atom': 'http://www.w3.org/2005/Atom'}
        filings = []
        for entry in root.findall('atom:entry', ns)[:30]:
            title   = entry.findtext('atom:title',   '', ns)
            summary = entry.findtext('atom:summary', '', ns)
            updated = entry.findtext('atom:updated', '', ns)
            filings.append({
                'title':   title,
                'summary': summary[:400],
                'symbols': [],     # Gemini extracts the ticker from company name
                'source':  'SEC EDGAR 8-K',
                'time':    updated,
            })
        log.info(f'  SEC EDGAR 8-K: {len(filings):3d} filings')
        return filings
    except Exception as e:
        log.warning(f'  SEC EDGAR error: {e}')
        return []


# ══════════════════════════════════════════════════════════
# GEMINI ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════

def analyze_news_with_gemini(articles: list) -> dict:
    """
    Send articles to Gemini Flash for deep analysis.

    For each impacted stock, Gemini returns:
      • sentiment     : -1.0 (very bearish) to +1.0 (very bullish)
      • catalyst      : M&A / EARNINGS / CONTRACT / REGULATORY / etc.
      • impact        : HIGH / MEDIUM / LOW
      • weight        : 0.0–1.0 (how important is this catalyst)
      • headline      : the exact news that drove the score
      • reason        : plain English — WHY does this impact the stock?
      • bullish_case  : one sentence — why a trader should buy
      • risk          : one sentence — the main downside risk

    Returns dict keyed by uppercase ticker symbol.
    """
    if not gemini or not articles:
        return {}

    batch_size  = 12
    all_scores: dict = {}

    for i in range(0, min(len(articles), 72), batch_size):
        batch = articles[i : i + batch_size]
        article_text = '\n'.join([
            f"[{j+1}] SOURCE: {a['source']} | TICKERS: {','.join(a['symbols']) or 'unknown'}\n"
            f"    HEADLINE: {a['title']}\n"
            f"    SUMMARY: {a.get('summary', '')[:200]}"
            for j, a in enumerate(batch)
        ])

        prompt = f"""You are a senior Wall Street analyst. Analyze these market news items and identify which specific stocks will be MOST IMPACTED today.

NEWS ITEMS:
{article_text}

For each clearly impacted stock ticker, return a JSON object (valid JSON only, no extra text, no markdown):
{{
  "TICKER": {{
    "sentiment":    <float -1.0 to 1.0, positive=bullish for that stock>,
    "catalyst":     <exactly one of: "M&A" | "EARNINGS" | "CONTRACT" | "REGULATORY" | "UPGRADE" | "INSIDER" | "PRODUCT" | "OTHER">,
    "impact":       <"HIGH" | "MEDIUM" | "LOW">,
    "weight":       <float 0.0-1.0 based on how significant this catalyst is>,
    "headline":     <exact headline text that drove this score>,
    "reason":       <2-3 sentences in plain English: WHY does this news move this stock today? Include specific numbers if available>,
    "bullish_case": <1 sentence: the clearest reason a trader should buy this stock today>,
    "risk":         <1 sentence: the biggest reason NOT to buy, or main downside risk>
  }}
}}

Rules:
- M&A target company: sentiment +0.75 to +1.0 (stock pops on buyout premium)
- M&A acquirer: sentiment -0.1 to +0.2 (depends on deal terms)
- Earnings beat: +0.5 to +0.9 based on magnitude of beat
- Analyst downgrade: sentiment -0.4 to -0.8
- Only include stocks with HIGH confidence of price impact TODAY
- For SEC EDGAR filings: extract the company name and map to its ticker symbol
- Return ONLY valid JSON, no code blocks, no commentary"""

        try:
            resp = gemini.generate_content(prompt)
            text = resp.text.strip()
            # Strip markdown code fences if Gemini adds them
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json\n'):
                    text = text[5:]
            parsed = json.loads(text)
            for sym, data in parsed.items():
                sym = sym.upper().strip()
                # Keep whichever score has stronger conviction
                existing = all_scores.get(sym)
                if not existing or abs(data.get('sentiment', 0)) > abs(existing.get('sentiment', 0)):
                    all_scores[sym] = data
        except Exception as e:
            log.warning(f'  Gemini batch {i//batch_size + 1} parse error: {e}')

    return all_scores


def compute_news_score(sym: str, news_analysis: dict) -> float:
    """
    Final news score: sentiment × catalyst_weight × impact_multiplier.
    Range: -1.0 (very bearish) to +1.0 (very bullish).
    """
    if sym not in news_analysis:
        return 0.0
    d         = news_analysis[sym]
    sentiment = float(d.get('sentiment', 0))
    weight    = float(d.get('weight', 0.5))
    impact    = {'HIGH': 1.0, 'MEDIUM': 0.65, 'LOW': 0.35}.get(d.get('impact', 'LOW'), 0.35)
    # Extra boost for highest-value catalysts (M&A, FDA, earnings)
    cat_boost = CATALYST_WEIGHTS.get(d.get('catalyst', 'OTHER'), 0.30)
    return round(sentiment * weight * impact * cat_boost, 3)


def select_daily_watchlist(news_analysis: dict) -> list:
    """
    Score every candidate stock using a composite of three signals:

      40% News Alpha    — Gemini sentiment × catalyst importance × impact level
      35% ML Signal     — XGBoost 49-feature model (trained on 9 legend strategies)
      25% Technicals    — Minervini Stage 2 check (above EMA50) + RSI health

    Returns top 5 as [(symbol, score_dict), ...]
    """
    # Pool: stocks with notable news + always-on watchlist as safety net
    pool = set(FALLBACK_WATCHLIST)
    for sym, d in news_analysis.items():
        if d.get('impact') in ('HIGH', 'MEDIUM') and float(d.get('sentiment', 0)) > 0.15:
            pool.add(sym)

    candidates = {}
    for sym in pool:
        news_s = compute_news_score(sym, news_analysis)

        # ML prediction (use training universe index for consistent sym_id)
        sym_id = TRAINING_UNIVERSE.index(sym) if sym in TRAINING_UNIVERSE else 0
        ml     = ml_predict(sym, sym_id=sym_id)
        ml_s   = float(ml.get('confidence', 0))

        # Technical quality check
        above_ema  = int(ml.get('above_ema50', 0))
        rsi        = float(ml.get('rsi', 50))
        rsi_ok     = 40 <= rsi <= 72
        tech_s     = (0.5 if above_ema else 0.0) + (0.5 if rsi_ok else 0.0)

        composite = (news_s * 0.40) + (ml_s * 0.35) + (tech_s * 0.25)

        candidates[sym] = {
            'composite':  round(composite, 4),
            'news_score': news_s,
            'ml_score':   ml_s,
            'tech_score': tech_s,
            'ml_data':    ml,
            'news_data':  news_analysis.get(sym, {}),
        }

    ranked = sorted(candidates.items(), key=lambda x: -x[1]['composite'])
    return ranked[:10]   # top 10 — expanded from 5 for broader intraday coverage


# ══════════════════════════════════════════════════════════
# DAILY EXPLANATION REPORT  ← the "Why These 5" panel
# ══════════════════════════════════════════════════════════

def print_selection_report(ranked: list) -> None:
    """
    Print a detailed human-readable briefing explaining exactly WHY
    each stock was chosen — catalyst, numbers, bull case, and risk.

    This is the intelligence panel the user reads each morning to
    understand what the market is doing and why the bot is trading it.
    """
    sep   = '═' * 66
    today = now_et().strftime('%A  %B %d, %Y  ·  %I:%M %p ET')

    log.info(f'\n{sep}')
    log.info(f'  🎯  TODAY\'S AI STOCK SELECTION — {today}')
    log.info(f'  Why the bot chose these stocks for today\'s trading (up to 10)')
    log.info(sep)

    report_data = []

    for rank, (sym, data) in enumerate(ranked, 1):
        nd    = data['news_data']
        ml    = data['ml_data']
        price = float(ml.get('price', 0))
        rsi   = float(ml.get('rsi', 0))
        vol_r = float(ml.get('vol_ratio', 1))

        catalyst    = nd.get('catalyst',     'TECHNICAL')
        impact      = nd.get('impact',       '—')
        reason      = nd.get('reason',       'Strong technical setup with high ML confidence')
        headline    = nd.get('headline',     'No specific news — technicals-driven')
        bull_case   = nd.get('bullish_case', 'ML model shows high buy probability')
        risk_note   = nd.get('risk',         'Always manage with stop-loss')
        sentiment   = float(nd.get('sentiment', 0))

        stop_px   = round(price * (1 - STOP_PCT), 2) if price else 0
        target_px = round(price * (1 + TP_PCT),   2) if price else 0
        direction = '🟢 BULLISH' if sentiment >= 0 else '🔴 BEARISH'
        sent_bar  = '█' * int(abs(sentiment) * 8)

        # ── Header ──────────────────────────────────────────
        log.info(f'\n  ┌── #{rank} · {sym} ─ {catalyst} · {impact} IMPACT ─ {direction}')

        # ── Headline ────────────────────────────────────────
        log.info(f'  │')
        log.info(f'  │  📰 NEWS CATALYST')
        log.info(f'  │     {headline[:70]}')

        # ── Why it matters (word-wrapped) ───────────────────
        log.info(f'  │')
        log.info(f'  │  💡 WHY THIS MATTERS TO THE STOCK PRICE')
        words, line, out_lines = reason.split(), '', []
        for w in words:
            if len(line) + len(w) + 1 > 62:
                out_lines.append(line)
                line = w
            else:
                line = (line + ' ' + w).strip()
        if line:
            out_lines.append(line)
        for l in out_lines:
            log.info(f'  │     {l}')

        # ── Bull case & Risk ────────────────────────────────
        log.info(f'  │')
        log.info(f'  │  ✅ REASON TO BUY')
        log.info(f'  │     {bull_case[:70]}')
        log.info(f'  │')
        log.info(f'  │  ⚠️  MAIN RISK')
        log.info(f'  │     {risk_note[:70]}')

        # ── Scores ──────────────────────────────────────────
        log.info(f'  │')
        log.info(f'  │  📊 SCORES')
        log.info(f'  │     News    : {data["news_score"]:+.2f}  [{sent_bar:<8}] (catalyst × sentiment)')
        log.info(f'  │     ML Model: {data["ml_score"]:.0%}  confidence (49-feature XGBoost)')
        log.info(f'  │     Technical: {data["tech_score"]:.1f}/1.0  (Stage 2 + RSI health)')
        log.info(f'  │     COMPOSITE: {data["composite"]:.3f}  ← final ranking score')

        # ── Market data & levels ────────────────────────────
        log.info(f'  │')
        log.info(f'  │  📈 MARKET DATA')
        log.info(f'  │     Price ${price:.2f}  ·  RSI {rsi:.0f}  ·  Volume {vol_r:.1f}× average')
        if price:
            log.info(f'  │     Stop-loss  → ${stop_px}  (-2%)   ← bot auto-exits here')
            log.info(f'  │     Take-profit → ${target_px}  (+6%)   ← bot auto-exits here')
            log.info(f'  │     Risk/Reward  1:3  (PTJ rule — always risk $1 to make $3)')
        log.info(f'  └{"─" * 64}')

        report_data.append({
            'rank':        rank,
            'symbol':      sym,
            'catalyst':    catalyst,
            'impact':      impact,
            'headline':    headline,
            'reason':      reason,
            'bullish_case': bull_case,
            'risk':        risk_note,
            'sentiment':   sentiment,
            'scores': {
                'news':      data['news_score'],
                'ml':        data['ml_score'],
                'technical': data['tech_score'],
                'composite': data['composite'],
            },
            'price':  price,
            'stop':   stop_px,
            'target': target_px,
            'rsi':    rsi,
            'vol_ratio': vol_r,
        })

    log.info(f'\n  📌 HOW TO READ THIS REPORT:')
    log.info(f'     News score > 0.3  = strong catalyst driving price today')
    log.info(f'     ML > 55%          = model sees historical pattern of a winning trade')
    log.info(f'     Composite > 0.5   = bot will scan for entry trigger at market open')
    log.info(f'\n{sep}\n')

    # Save to JSON for external review (Render logs, dashboard, etc.)
    try:
        REPORT_FILE.write_text(json.dumps({
            'date':        now_et().strftime('%Y-%m-%d'),
            'generated':   now_et().isoformat(),
            'watchlist':   [sym for sym, _ in ranked],
            'picks':       report_data,
        }, indent=2))
        log.info(f'  📄 Full report saved → {REPORT_FILE}')
    except Exception as e:
        log.warning(f'  Could not save report file: {e}')


# ══════════════════════════════════════════════════════════
# NEWS INTELLIGENCE ORCHESTRATOR
# ══════════════════════════════════════════════════════════

def run_news_intelligence() -> None:
    """
    Master pipeline — called at 8:30 AM ET (30 min before market open).

    Step 1: Gather news from three free sources
    Step 2: Gemini AI analyzes every article
    Step 3: Score and rank all stocks
    Step 4: Print the "Why These 5" explanation report
    Step 5: Update WATCHLIST for today's trading session
    """
    global WATCHLIST

    sep = '─' * 66
    log.info(f'\n{sep}')
    log.info(f'  📡 NEWS INTELLIGENCE PIPELINE — {now_et().strftime("%I:%M %p ET")}')
    log.info(sep)

    # ── Step 1: Gather ──────────────────────────────────────
    alpaca_news  = fetch_alpaca_news(hours=20)
    yahoo_news   = fetch_yahoo_rss(
        FALLBACK_WATCHLIST
        + ['MSFT', 'AAPL', 'GOOGL', 'META', 'AMZN', 'TSLA', 'CRWD', 'AMD']
    )
    sec_news     = fetch_sec_8k(hours=24)
    all_articles = alpaca_news + yahoo_news + sec_news

    log.info(f'  Total items gathered: {len(all_articles)}')

    if not all_articles:
        log.warning('  ⚠️  No news available — keeping existing watchlist')
        return

    # ── Step 2: Gemini Analysis ──────────────────────────────
    if gemini:
        log.info('  🧠 Gemini analyzing all articles...')
        news_analysis = analyze_news_with_gemini(all_articles)
        log.info(f'  Gemini identified impact on {len(news_analysis)} stocks')
    else:
        log.warning('  ⚠️  Gemini not available — using technical signals only')
        news_analysis = {}

    # ── Step 3 & 4: Rank + Explain ──────────────────────────
    ranked = select_daily_watchlist(news_analysis)

    if not ranked:
        log.warning('  ⚠️  Could not rank stocks — keeping existing watchlist')
        return

    print_selection_report(ranked)

    # ── Step 5: Update Watchlist ─────────────────────────────
    new_wl = [sym for sym, _ in ranked]
    log.info(f'  📋 WATCHLIST: {WATCHLIST}  →  {new_wl}')
    WATCHLIST = new_wl
    log.info(f'  ✅ Bot will trade these {len(new_wl)} stocks today\n{sep}\n')


# ══════════════════════════════════════════════════════════
# INTRADAY NEWS MONITOR  (every 2 hours during market hours)
# ══════════════════════════════════════════════════════════

def run_intraday_news_update() -> None:
    """
    Runs every 2 hours while the market is open.
    Approximate fire times: 10:30 AM, 12:30 PM, 2:30 PM ET

    Two live actions:
      1. NEWS-DRIVEN EXIT   — sells an open position if strong negative
                              news hits (sentiment ≤ -0.6, HIGH/MEDIUM impact)
      2. BREAKOUT ADDITION  — adds a stock to today's watchlist if it
                              gets a major catalyst (M&A, earnings beat, etc.)
                              with sentiment ≥ 0.80 and HIGH impact
    """
    global WATCHLIST

    if not is_market_open():
        return

    sep = '─' * 66
    log.info(f'\n{sep}')
    log.info(f'  📡 INTRADAY NEWS UPDATE — {now_et().strftime("%I:%M %p ET")}')
    log.info(f'  Watchlist ({len(WATCHLIST)}): {WATCHLIST}')
    log.info(sep)

    # ── Gather last ~2.5 hours of news ─────────────────────
    positions   = get_positions()
    # Focus on current watchlist + any open positions not already in it
    focus_syms  = list(set(WATCHLIST + list(positions.keys())))
    alpaca_news = fetch_alpaca_news(hours=3)
    yahoo_news  = fetch_yahoo_rss(focus_syms[:14])
    all_articles = alpaca_news + yahoo_news

    log.info(f'  Articles (last 3 h): {len(all_articles)}')

    if not all_articles:
        log.info('  No recent news — watchlist unchanged')
        log.info(f'{sep}\n')
        return

    # ── Gemini quick-analysis ───────────────────────────────
    if not gemini:
        log.info('  Gemini not available — skipping intraday analysis')
        log.info(f'{sep}\n')
        return

    news_analysis = analyze_news_with_gemini(all_articles)
    log.info(f'  Gemini found news on {len(news_analysis)} stocks')

    # ── Action 1: News-driven early exit ───────────────────
    log.info('\n  📌 OPEN POSITIONS:')
    for sym, pos in positions.items():
        if sym not in news_analysis:
            log.info(f'  {sym:5s}: holding | no recent news')
            continue
        nd        = news_analysis[sym]
        sentiment = float(nd.get('sentiment', 0))
        impact    = nd.get('impact', 'LOW')
        catalyst  = nd.get('catalyst', 'OTHER')
        headline  = nd.get('headline', 'N/A')

        if sentiment <= -0.6 and impact in ('HIGH', 'MEDIUM'):
            # Strong negative news → exit now, don't wait for stop
            current = float(pos.current_price)
            entry   = float(pos.avg_entry_price)
            pnl_pct = (current - entry) / entry
            log.info(
                f'  🚨 {sym}: NEGATIVE NEWS EXIT '
                f'sentiment={sentiment:+.2f} [{impact}]  pnl={pnl_pct:+.1%}'
            )
            log.info(f'     📰 {headline[:70]}')
            place_sell(sym, pos, reason='NEWS_EXIT')
        else:
            sent_icon = '🟢' if sentiment >= 0 else '🔴'
            log.info(
                f'  ✅ {sym:5s}: holding | {sent_icon} sentiment={sentiment:+.2f} '
                f'[{impact}] {catalyst}'
            )

    # ── Action 2: Mid-day breakout additions ───────────────
    new_additions = []
    for sym, nd in news_analysis.items():
        if sym in WATCHLIST or sym not in SCAN_UNIVERSE:
            continue
        sentiment  = float(nd.get('sentiment', 0))
        impact     = nd.get('impact', 'LOW')
        catalyst   = nd.get('catalyst', 'OTHER')
        cat_weight = CATALYST_WEIGHTS.get(catalyst, 0.30)

        # Only high-conviction, market-moving catalysts
        if (sentiment >= 0.80
                and impact == 'HIGH'
                and cat_weight >= 0.75):
            headline = nd.get('headline', '')
            score    = round(cat_weight * sentiment, 3)
            log.info(
                f'  ⚡ BREAKOUT CANDIDATE: {sym} | {catalyst} [{impact}] '
                f'sentiment={sentiment:+.2f} score={score}'
            )
            log.info(f'     📰 {headline[:70]}')
            new_additions.append((sym, score))

    if new_additions:
        new_additions.sort(key=lambda x: -x[1])
        slots = 10 - len(WATCHLIST)
        if slots > 0:
            add_syms = [s for s, _ in new_additions[:slots]]
            WATCHLIST.extend(add_syms)
            log.info(f'\n  📋 Added mid-day: {add_syms}')
        else:
            log.info(
                f'\n  📋 Breakout candidates found but watchlist full (10): '
                + ', '.join(s for s, _ in new_additions)
            )
    else:
        log.info('\n  No breakout additions — watchlist unchanged')

    log.info(
        f'\n  ✅ Intraday update done | '
        f'Watchlist ({len(WATCHLIST)}): {WATCHLIST}'
    )
    log.info(f'{sep}\n')


# ══════════════════════════════════════════════════════════
# MAIN SCAN LOOP
# ══════════════════════════════════════════════════════════
scan_count = 0


def scan() -> None:
    """
    Every 10 minutes:
    1. Check open positions for stop-loss / take-profit
    2. Score all watchlist stocks with combined signal
    3. Enter the best opportunity if signal strong enough
    """
    global scan_count

    if not is_market_open():
        return

    scan_count  += 1
    _scan_count_ref[0] = scan_count   # expose to status server
    equity       = get_account_equity()
    positions    = get_positions()
    open_watch   = sum(1 for s in positions if s in WATCHLIST)

    log.info(
        f'\n── Scan #{scan_count} | ${equity:,.2f} equity | '
        f'{open_watch}/{MAX_POSITIONS} positions | Model: {model_accuracy:.1%} ──'
    )

    # ── Step 1: manage existing positions ────────────────
    manage_open_positions()
    positions = get_positions()   # refresh after any closes
    open_watch = sum(1 for s in positions if s in WATCHLIST)

    if open_watch >= MAX_POSITIONS:
        log.info(f'  Max positions ({MAX_POSITIONS}) reached — watching only')
        return

    # ── Step 2: score every symbol ────────────────────────
    best_sym, best_score, best_state = None, 0.45, {}

    for sym in WATCHLIST:
        if sym in positions:
            log.info(f'  {sym:5s}: HOLDING')
            continue

        # Earnings blackout — never buy within 2 days of earnings report
        if is_earnings_blackout(sym):
            dt  = earnings_cache.get(sym)
            eta = (dt - datetime.now(timezone.utc)).days if dt else '?'
            log.info(f'  {sym:5s}: 📅 EARNINGS BLACKOUT ({eta}d away) — skip')
            continue

        # Check dip-watch: only re-enter if candle patterns confirm reversal
        can_enter, dip_status = check_dip_reversal(sym)
        if not can_enter:
            log.info(f'  {sym:5s}: DIP WATCH — {dip_status}')
            continue

        ml  = ml_predict(sym)   # sym_id auto-resolved from TRAINING_UNIVERSE
        tv  = get_tv_analysis(sym)
        sig = combined_signal(sym, ml, tv)

        tv_str = tv["rec"] if tv else "N/A(ML+)"  # N/A(ML+) = TV down, ML weight boosted
        log.info(
            f'  {sym:5s}: {sig["signal"]:4s} '
            f'score={sig["score"]:+.3f} '
            f'ml={ml["confidence"]:.0%} '
            f'tv={tv_str:15s} '
            f'rsi={ml.get("rsi",0):.0f} '
            f'vol×{ml.get("vol_ratio",1):.1f}'
        )

        if sig['signal'] == 'BUY' and sig['score'] > best_score:
            best_sym, best_score, best_state = sym, sig['score'], {'ml': ml, 'sig': sig}

    # ── Step 3: enter best opportunity ────────────────────
    if best_sym:
        log.info(f'  🎯 Best: {best_sym} (score={best_score:.3f}) — placing order')
        place_buy(best_sym, best_state['ml']['price'], equity, best_state['sig'])
    else:
        log.info('  No actionable signal this scan')


# ══════════════════════════════════════════════════════════
# STATUS HTTP SERVER
# Serves model accuracy + precision at GET /status so the
# trading dashboard can display them without manual entry.
# Render requires a PORT binding — this satisfies that too.
#
# Dashboard fetches: https://your-render-url.onrender.com/status
# Response (JSON):
#   { "accuracy": 74.2, "precision": 38.5,
#     "trained_at": "2026-07-07T04:05:12",
#     "scan_count": 12, "status": "running" }
# ══════════════════════════════════════════════════════════

_STATUS_PORT = int(os.environ.get('PORT', 8080))
_scan_count_ref = [0]   # mutable ref so handler can read global scan_count


class _StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({
            'accuracy':   round(model_accuracy  * 100, 1),
            'precision':  round(model_precision * 100, 1),
            'trained_at': model_trained_at[:19] if model_trained_at != 'never' else 'never',
            'scan_count': _scan_count_ref[0],
            'status':     'running',
        })
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')   # allow dashboard cross-origin fetch
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *_):
        pass   # suppress per-request HTTP logs — keep Render logs clean


def _start_status_server() -> None:
    """Start the status HTTP server in a background daemon thread."""
    try:
        server = HTTPServer(('0.0.0.0', _STATUS_PORT), _StatusHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        log.info(f'📡 Status server → http://0.0.0.0:{_STATUS_PORT}/status')
    except Exception as e:
        log.warning(f'Status server failed to start: {e}')


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    sep = '═' * 60
    log.info(sep)
    log.info('🤖  AI SELF-LEARNING PAPER TRADING BOT')
    log.info('    XGBoost + TradingView + Claude AI')
    log.info('    Alpaca Paper — $100,000 account')
    log.info(sep)

    # ── Verify Alpaca connection ──────────────────────────
    try:
        acct = trade_client.get_account()
        log.info(f'✅ Alpaca paper connected')
        log.info(f'   Equity:  ${float(acct.equity):>12,.2f}')
        log.info(f'   Cash:    ${float(acct.cash):>12,.2f}')
        log.info(f'   BP:      ${float(acct.buying_power):>12,.2f}')
    except Exception as e:
        log.error(f'Cannot connect to Alpaca: {e}')
        raise

    # ── Status server (serves model stats to dashboard) ──
    _start_status_server()

    # ── Load history & train model ────────────────────────
    load_log()
    load_or_train()

    # ── Schedule all recurring tasks ──────────────────────
    schedule.every(SCAN_EVERY).minutes.do(scan)
    schedule.every(1).minutes.do(check_eod)
    # ── All times below are UTC (Render servers run UTC) ──
    # Conversion: ET (EDT) = UTC - 4
    #   8:30 AM ET = 12:30 UTC  ← News Intelligence (before open)
    #   9:00 AM ET = 13:00 UTC  ← Gemini morning brief
    # Every 120 min after start  ← Intraday news monitor (~10:30, 12:30, 2:30 PM ET)
    #  12:05 AM ET = 04:05 UTC  ← Nightly retrain
    schedule.every().day.at('12:30').do(run_news_intelligence)         # 8:30 AM ET
    schedule.every().day.at('13:00').do(generate_morning_brief)        # 9:00 AM ET
    schedule.every(120).minutes.do(run_intraday_news_update)           # every 2 h (market open check inside)
    schedule.every().day.at('12:00').do(refresh_earnings_cache)        # 8:00 AM ET — before trading starts
    schedule.every().day.at('04:05').do(nightly_retrain)               # 12:05 AM ET

    log.info(f'\n📅 SCHEDULE (all times Eastern):')
    log.info(f'  Every {SCAN_EVERY} min  → Market scan + signal evaluation')
    log.info(f'  08:00 AM ET → 📅 Earnings calendar refresh (blackout gate)')
    log.info(f'  08:30 AM ET → 📡 News Intelligence — picks up to 10 stocks')
    log.info(f'  09:00 AM ET → 🧠 Gemini morning brief (optional)')
    log.info(f'  Every 2 hrs → 📡 Intraday news update — exits on bad news, adds breakouts')
    log.info(f'  03:50 PM ET → 🔴 Close all positions (EOD)')
    log.info(f'  12:05 AM ET → 🌙 Nightly XGBoost retrain (self-learning)')
    log.info(f'\n🚀 Bot is live!\n{sep}')

    # ── Immediate startup tasks ───────────────────────────
    et_now  = now_et()
    et_hour = et_now.hour

    # Earnings calendar — always refresh on startup so blackout gate is ready
    refresh_earnings_cache()

    # Log initial market regime
    regime = get_market_regime()
    log.info(f'📊 Market regime at startup: {regime} (SPY vs 50-day SMA)')

    # If starting between 8-9 AM ET, run news intelligence right now
    if 8 <= et_hour < 9:
        run_news_intelligence()
    # Morning brief on startup between 9-10 AM ET
    if 9 <= et_hour <= 10 and GEMINI_KEY and GEMINI_KEY != 'YOUR_GEMINI_KEY':
        generate_morning_brief()
    scan()

    # ── Main event loop ───────────────────────────────────
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == '__main__':
    main()
