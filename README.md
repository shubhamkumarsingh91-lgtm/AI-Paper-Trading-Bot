AI Self-Learning Paper Trading Bot
XGBoost ML bot running on Alpaca paper account ($100,000 simulated). Trains on 9 legendary trader strategies, reads market news every morning via Gemini AI, picks the best 5 stocks for the day, and retrains itself every night on real trade outcomes.


What it does
Time (ET)
Action
8:30 AM
Reads news from Alpaca, Yahoo Finance, SEC EDGAR · Gemini AI picks today's 5 stocks
9:00 AM
Optional Gemini morning brief (market overview)
9:30 AM+
Scans every 10 min · buys best signal · monitors stop/target
3:50 PM
Closes all positions before market close
12:05 AM
Retrains XGBoost model on today's outcomes



Setup
1. Clone and install
git clone https://github.com/YOUR_USERNAME/AI-Paper-Trading-Bot

cd AI-Paper-Trading-Bot

pip install alpaca-py xgboost scikit-learn pandas numpy joblib tradingview-ta schedule google-generativeai requests pytz
2. Set environment variables
Never hardcode keys. Set these in Render dashboard → Environment:

Variable
Where to get it
ALPACA_API_KEY
alpaca.markets → Paper Account → API Keys
ALPACA_SECRET_KEY
Same page
GEMINI_API_KEY
aistudio.google.com → Get API Key (free)

3. Deploy on Render
Push code to GitHub
Render → New → Background Worker
Build command: pip install -r requirements.txt
Start command: python ai_trading_bot.py
Add the 3 environment variables above
Deploy


How the AI learns
Day 1:  Bot trains on 1 year of price history for 14 stocks

        → XGBoost learns patterns from 9 legendary trader strategies

        → Makes first trades based on this knowledge

Night 1: Nightly retrain runs at 12:05 AM

         → Incorporates today's actual trade outcomes

         → Model improves: "what signals led to winning trades?"

Week 2+: Model is now trained on REAL outcomes, not just history

         → Win rate improves as it learns what actually works

         → Each night it gets smarter

Accuracy starts around 80% (price history only). As real trade outcomes accumulate, the model learns which signals actually make money vs. just look good on paper.


The 49-feature model
9 legendary traders encoded as machine-learning features:

Trader
Strategy
Features
Stan Weinstein
Stage Analysis
above_sma150, sma150_rising
Mark Minervini
SEPA / VCP
ema_stack, pct_from_high, atr_contracting
William O'Neil
CAN SLIM Volume
vol_up_vs_down, vol_surge
Nicolas Darvas
Box Theory
donchian_pct, donchian_break
Paul Tudor Jones
200-SMA Trend
trend_strength_200, ma200_slope
Linda Raschke
Holy Grail
adx, adx_trending, oversold_trend
Larry Williams
%R + Temporal
williams_r, day_of_week, is_month_end
John Bollinger
Band Width/Squeeze
bb_pct, bb_squeeze, bb_width, bb_at_lower
Richard Donchian
Channel Breakout
donchian_pct, donchian_break


Plus standard momentum, RSI, MACD, stochastics, ATR, candle patterns, and time-of-day features.


News Intelligence (runs 8:30 AM ET)
Three free data sources scanned every morning:

Alpaca News API — real-time articles tagged by ticker symbol. Already included in your Alpaca account, no extra cost.

Yahoo Finance RSS — earnings beats, analyst upgrades, company news. No API key needed.

SEC EDGAR 8-K filings — the most powerful source. Companies must file within 4 business days of a merger, acquisition, major contract, or material event. This catches M&A news before most retail traders see it.

Gemini AI reads all articles and produces for each stock:

Catalyst type (M&A / Earnings / Contract / Regulatory / etc.)
Sentiment score (-1.0 bearish to +1.0 bullish)
Plain English explanation of why this news moves the stock
Bull case and main risk

Stocks are then scored: 40% news alpha + 35% ML model + 25% technicals. Top 5 become today's watchlist.


Daily picks report
Every morning you'll see this in Render logs:

════════════════════════════════════════════════════════════════════

  TODAY'S AI STOCK SELECTION — Monday July 06, 2026 · 08:30 AM ET

════════════════════════════════════════════════════════════════════

  ┌── #1 · PANW ─ M&A · HIGH IMPACT ─ BULLISH

  │

  │  NEWS CATALYST

  │     Palo Alto Networks in advanced talks to acquire Axonius for $2.3B

  │

  │  WHY THIS MATTERS TO THE STOCK PRICE

  │     Acquisition fills a gap in PANW's platform. Deal expected to add

  │     $0.40 EPS within 18 months. M&A targets typically rally 15-30%.

  │

  │  REASON TO BUY

  │     Institutions chase PANW on platform-expansion M&A — volume surge likely

  │

  │  MAIN RISK

  │     Integration risk if deal terms disappoint on closing call

  │

  │  SCORES

  │     News    : +0.72  [████████] (catalyst × sentiment)

  │     ML Model: 68%    confidence (49-feature XGBoost)

  │     COMPOSITE: 0.581 ← final ranking score

  │

  │  LEVELS

  │     Stop-loss  → $342.22  (-2%)   ← bot auto-exits here

  │     Take-profit → $370.15  (+6%)  ← bot auto-exits here

  └────────────────────────────────────────────────────────────────

Full report also saved to daily_picks.json each morning.


Risk management
Rule
Value
Source
Stop loss
-2%
Paul Tudor Jones
Take profit
+6% (3:1 R/R)
Paul Tudor Jones
Cooldown after stop loss
120 min
Prevents catching falling knife
Cooldown after take profit
60 min
Prevents buying the top
EMA exit confirmation
2 consecutive bars
Prevents selling on 1-candle noise
Max positions
3
Concentration for conviction
EOD close
3:50 PM ET
Jesse Livermore: no losers overnight



Files
File
Description
ai_trading_bot.py
Main bot — all logic in one file
ai_model.xgb
Saved XGBoost model (auto-created on first run)
trade_log.json
Every buy/sell with P&L (used for nightly retrain)
daily_picks.json
Today's AI stock picks with full reasoning
morning_brief.txt
Gemini market overview (if API key provided)
ai_bot.log
Full bot log



Live trading bot (separate repo)
trading_bot.py is a separate rules-based bot for real money. Same watchlist, same strategies, but no ML — pure Minervini + Raschke + PTJ logic with fixed rules.

Security: API keys must be set as environment variables. Never hardcode real money keys.

# Correct — reads from environment

API_KEY    = os.environ.get('ALPACA_API_KEY')

SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')


Monitoring
Check Render logs daily. Key things to look for:

Model ready — Accuracy: XX% — model health after nightly retrain
BUY entries — position opened, check entry price and stop level
SELL entries — position closed, check P&L and reason
COOLDOWN — bot correctly avoiding re-entry after a loss
WATCHLIST UPDATE — today's news picked new stocks

Win rate below 35% after 30+ trades means signals need review. Win rate above 50% means the self-learning is working.


Architecture
News Sources (8:30 AM)

  Alpaca API + Yahoo RSS + SEC EDGAR

          ↓

  Gemini Flash Analysis

  (catalyst type, sentiment, explanation)

          ↓

  Stock Ranker

  (news 40% + ML 35% + technical 25%)

          ↓

  Dynamic Watchlist (top 5 today)

          ↓

  Market Scan (every 10 min)

  TradingView TA + XGBoost ML

          ↓

  Trade Execution (Alpaca Paper)

  Stop loss / Take profit / Trailing stop

          ↓

  Nightly Retrain (12:05 AM)

  Outcomes feed back into XGBoost

