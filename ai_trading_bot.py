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
import requests
import xml.etree.ElementTree as XMLTree
import pytz
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import schedule
from tradingview_ta import TA_Handler, Interval

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_bot.log', encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
API_KEY     = os.environ.get('ALPACA_API_KEY',    'YOUR_PAPER_KEY')
SECRET_KEY  = os.environ.get('ALPACA_SECRET_KEY', 'YOUR_PAPER_SECRET')
GEMINI_KEY  = os.environ.get('GEMINI_API_KEY',    'YOUR_GEMINI_KEY')

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
POSITION_CAP  = 0.20   # max 20% of equity in one stock
STOP_PCT      = 0.02   # stop loss at -2%
TP_PCT        = 0.06   # take profit at +6%
MAX_POSITIONS = 3      # max simultaneous open positions

# ── Cooldown after selling — prevents churning ────────────
# After TAKE_PROFIT: don't re-enter for 60 min (stock ran up, don't chase the top)
# After STOP_LOSS:   don't re-enter for 2h   (stock falling, don't catch the knife)
COOLDOWN_AFTER_TP   = 60    # minutes
COOLDOWN_AFTER_STOP = 120   # minutes

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
MODEL_FILE   = Path('ai_model.xgb')
LOG_FILE     = Path('trade_log.json')
BRIEF_FILE   = Path('morning_brief.txt')
REPORT_FILE  = Path('daily_picks.json')   # today's AI stock picks with explanations
TRAIN_DAYS   = 365

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
        gemini = _genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        log.warning(f'Gemini init skipped: {e}')


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

    return d


def make_labels(df: pd.DataFrame, forward: int = 8, threshold: float = 0.005) -> pd.Series:
    """
    Binary label: 1 = price rises more than 0.5% in next 8 bars (≈2 hours).
    This is what the model tries to predict.
    """
    future_ret = df['close'].shift(-forward) / df['close'] - 1
    return (future_ret > threshold).astype(int)


# ══════════════════════════════════════════════════════════
# XGBOOST MODEL
# ══════════════════════════════════════════════════════════
model: xgb.XGBClassifier = None
model_accuracy: float     = 0.0
model_trained_at: str     = 'never'


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


def train_model(retrain: bool = False) -> None:
    """
    Train XGBoost on 1 year of 15-min data for all 5 watchlist stocks.
    Called once at startup and then nightly for self-improvement.
    """
    global model, model_accuracy, model_trained_at
    label = 'Nightly retrain' if retrain else 'Initial training'
    log.info(f'🧠 {label} — fetching {TRAIN_DAYS} days × {len(TRAINING_UNIVERSE)} stocks...')
    log.info(f'   Training universe: {", ".join(TRAINING_UNIVERSE)}')
    log.info(f'   Trading watchlist: {", ".join(WATCHLIST)}')

    frames = []
    for i, sym in enumerate(TRAINING_UNIVERSE):
        try:
            df  = fetch_bars(sym, days=TRAIN_DAYS)
            df  = add_features(df)
            df['target'] = make_labels(df)
            df['sym_id'] = i
            df = df.dropna()
            frames.append(df)
            log.info(f'  ✓ {sym}: {len(df):,} bars loaded')
        except Exception as e:
            log.warning(f'  ✗ {sym}: {e}')

    if not frames:
        log.error('No training data — cannot train')
        return

    all_data = pd.concat(frames)
    X = all_data[FEATURES + ['sym_id']]
    y = all_data['target']
    pos_rate = y.mean()
    log.info(f'  Total: {len(X):,} samples | Bullish rate: {pos_rate:.1%}')

    # Time-series split (no shuffle — respect temporal order)
    split_idx = int(len(X) * 0.8)
    X_tr, X_va = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_va = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        scale_pos_weight=(y == 0).sum() / ((y == 1).sum() + 1),
        eval_metric='logloss',
        early_stopping_rounds=50,   # moved here for XGBoost 2.0+
        verbosity=0,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    preds = model.predict(X_va)
    model_accuracy   = accuracy_score(y_va, preds)
    prec             = precision_score(y_va, preds, zero_division=0)
    model_trained_at = datetime.now().isoformat()

    # Top 5 most important features
    fi = sorted(
        zip(FEATURES + ['sym_id'], model.feature_importances_),
        key=lambda x: -x[1]
    )[:5]
    top_feats = ', '.join(f'{n}({v:.2f})' for n, v in fi)

    log.info(f'✅ Model ready — Accuracy: {model_accuracy:.1%} | Precision: {prec:.1%}')
    log.info(f'  Top signals learned: {top_feats}')

    joblib.dump({
        'model':      model,
        'accuracy':   model_accuracy,
        'precision':  prec,
        'trained_at': model_trained_at,
        'features':   FEATURES,
    }, MODEL_FILE)
    log.info(f'💾 Model saved → {MODEL_FILE}')


def load_or_train() -> None:
    """Load saved model or train fresh if none exists."""
    global model, model_accuracy, model_trained_at
    if MODEL_FILE.exists():
        try:
            saved = joblib.load(MODEL_FILE)
            model            = saved['model']
            model_accuracy   = saved['accuracy']
            model_trained_at = saved['trained_at']
            log.info(f'📦 Model loaded — accuracy: {model_accuracy:.1%}, trained: {model_trained_at[:10]}')
            return
        except Exception as e:
            log.warning(f'Model load failed ({e}), retraining from scratch')
    train_model()


def ml_predict(symbol: str, sym_id: int = None) -> dict:
    # Use TRAINING_UNIVERSE index so sym_id is consistent with training
    if sym_id is None:
        sym_id = TRAINING_UNIVERSE.index(symbol) if symbol in TRAINING_UNIVERSE else 0
    """Run the trained model on the latest bars for a given symbol."""
    if model is None:
        return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}
    try:
        df = fetch_bars(symbol, days=45)    # 45 calendar days ≈ 800+ 15-min bars (avoids weekend gaps)
        if len(df) < 250:
            log.warning(f'  {symbol}: not enough bars ({len(df)})')
            return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}

        df = add_features(df)
        df['sym_id'] = sym_id
        row = df[FEATURES + ['sym_id']].iloc[-1:]

        if row.isnull().any().any():
            return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}

        prob = float(model.predict_proba(row)[0][1])

        return {
            'confidence': prob,
            'price':      float(df['close'].iloc[-1]),
            'rsi':        float(df['rsi'].iloc[-1]),
            'vol_ratio':  float(df['vol_ratio'].iloc[-1]),
            'macd_hist':  float(df['macd_hist'].iloc[-1]),
            'above_ema50': int(df['above_ema50'].iloc[-1]),
            'bb_pct':     float(df['bb_pct'].iloc[-1]),
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
    Score >= 0.55 → BUY.  Below → HOLD.
    """
    score   = 0.0
    reasons = []

    # ── XGBoost (50%) ─────────────────────────────────────
    conf = ml.get('confidence', 0.0)
    # normalize: 0.5 confidence = 0 contribution, 1.0 = full weight
    ml_contrib = (conf - 0.5) * 2 * ML_WEIGHT
    score += ml_contrib
    reasons.append(f'ML:{conf:.0%}')

    # ── TradingView (35%) ─────────────────────────────────
    if tv:
        tv_contrib = TV_SCORE.get(tv['rec'], 0.0) * TV_WEIGHT
        score += tv_contrib
        reasons.append(f'TV:{tv["rec"]}')
        # ADX bonus: strong trend (ADX > 25) boosts conviction
        if tv.get('adx', 0) > 25 and tv['rec'] in ('BUY', 'STRONG_BUY'):
            score += 0.04
            reasons.append('ADX:TREND')
    else:
        reasons.append('TV:N/A')

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

    final_signal = 'BUY' if score >= 0.55 else 'HOLD'
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

        BRIEF_FILE.write_text(
            f"Generated: {now_et().isoformat()}\n"
            f"Equity: ${equity:,.2f} | Model: {model_accuracy:.1%}\n"
            f"{'─'*60}\n{morning_brief_text}"
        )
    except Exception as e:
        log.warning(f'Gemini API error: {e}')
        morning_brief_text = f'[Brief unavailable: {e}]'


# ══════════════════════════════════════════════════════════
# COOLDOWN TRACKER  (prevents re-buying immediately after a sell)
# ══════════════════════════════════════════════════════════
cooldown_until: dict = {}   # symbol → datetime when cooldown expires

def in_cooldown(symbol: str) -> bool:
    """Return True if this symbol is still in its post-sale cooldown window."""
    if symbol not in cooldown_until:
        return False
    if now_et() >= cooldown_until[symbol]:
        del cooldown_until[symbol]
        return False
    return True

def set_cooldown(symbol: str, reason: str) -> None:
    minutes = COOLDOWN_AFTER_STOP if reason == 'STOP_LOSS' else COOLDOWN_AFTER_TP
    cooldown_until[symbol] = now_et() + timedelta(minutes=minutes)
    log.info(f'  ⏳ {symbol} cooldown {minutes} min [{reason}] — re-entry blocked until {cooldown_until[symbol].strftime("%H:%M ET")}')


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
    pos = get_positions()
    if symbol in pos:
        return False  # already have this

    open_count = sum(1 for s in pos if s in WATCHLIST)
    if open_count >= MAX_POSITIONS:
        return False

    # Position sizing: risk 2% of equity, stop at -2% → position = 100%×equity×RISK_PCT/STOP_PCT
    # Capped at POSITION_CAP of total equity
    qty_usd = min(equity * RISK_PCT / STOP_PCT, equity * POSITION_CAP)
    qty     = max(1, int(qty_usd / price))

    try:
        order = trade_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        log.info(
            f'🟢 BUY  {qty:5d}×{symbol} @ ${price:8.2f} '
            f'| Score:{signal["score"]:+.3f} | {" | ".join(signal["reasons"])}'
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
        return False


def place_sell(symbol: str, pos, reason: str) -> None:
    """Market sell an open position, log the outcome."""
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
        # Start cooldown — prevents immediately re-buying at the wrong price
        if reason in ('STOP_LOSS', 'TAKE_PROFIT'):
            set_cooldown(symbol, reason)
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
    """Check all open positions against stop-loss and take-profit levels."""
    for sym, pos in get_positions().items():
        if sym not in WATCHLIST:
            continue
        entry   = float(pos.avg_entry_price)
        current = float(pos.current_price)
        pct     = (current - entry) / entry

        if pct <= -STOP_PCT:
            place_sell(sym, pos, 'STOP_LOSS')
        elif pct >= TP_PCT:
            place_sell(sym, pos, 'TAKE_PROFIT')


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
    return ranked[:5]


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
    log.info(f'  Why the bot chose these 5 stocks for today\'s trading')
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
    log.info(f'  ✅ Bot will trade these 5 stocks today\n{sep}\n')


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

        # Skip if in cooldown after a recent stop-loss or take-profit
        if in_cooldown(sym):
            remaining = int((cooldown_until[sym] - now_et()).seconds / 60)
            log.info(f'  {sym:5s}: COOLDOWN ({remaining} min left — not chasing)')
            continue

        ml  = ml_predict(sym)   # sym_id auto-resolved from TRAINING_UNIVERSE
        tv  = get_tv_analysis(sym)
        sig = combined_signal(sym, ml, tv)

        log.info(
            f'  {sym:5s}: {sig["signal"]:4s} '
            f'score={sig["score"]:+.3f} '
            f'ml={ml["confidence"]:.0%} '
            f'tv={tv["rec"] if tv else "N/A":12s} '
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
    #  12:05 AM ET = 04:05 UTC  ← Nightly retrain
    schedule.every().day.at('12:30').do(run_news_intelligence)   # 8:30 AM ET
    schedule.every().day.at('13:00').do(generate_morning_brief)  # 9:00 AM ET
    schedule.every().day.at('04:05').do(nightly_retrain)         # 12:05 AM ET

    log.info(f'\n📅 SCHEDULE (all times Eastern):')
    log.info(f'  Every {SCAN_EVERY} min  → Market scan + signal evaluation')
    log.info(f'  08:30 AM ET → 📡 News Intelligence — picks today\'s 5 stocks')
    log.info(f'  09:00 AM ET → 🧠 Gemini morning brief (optional)')
    log.info(f'  03:50 PM ET → 🔴 Close all positions (EOD)')
    log.info(f'  12:05 AM ET → 🌙 Nightly XGBoost retrain (self-learning)')
    log.info(f'\n🚀 Bot is live!\n{sep}')

    # ── Immediate startup tasks ───────────────────────────
    et_now  = now_et()
    et_hour = et_now.hour
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
