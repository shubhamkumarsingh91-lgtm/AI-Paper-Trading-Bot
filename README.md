[README.md](https://github.com/user-attachments/files/29645379/README.md)
# 🤖 AI Self-Learning Paper Trading Bot

> An autonomous trading bot powered by XGBoost ML, trained on strategies from 9 of the world's greatest traders, with daily news intelligence powered by Google Gemini.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Alpaca](https://img.shields.io/badge/Broker-Alpaca%20Paper-yellow?style=flat-square)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=flat-square)
![Gemini](https://img.shields.io/badge/AI-Gemini%20Flash-purple?style=flat-square)
![Render](https://img.shields.io/badge/Deploy-Render-green?style=flat-square)

---

## What This Bot Does

Every trading day, the bot wakes up before the market opens, reads the news, picks the best stocks, trades them, and then learns from the results overnight — fully automated, no human input needed.

```
8:30 AM  →  Reads news · Gemini AI picks today's 5 best stocks
9:30 AM  →  Market opens · Bot scans every 10 minutes for entries
 During  →  Auto stop-loss · Auto take-profit · Trailing stop
3:50 PM  →  Closes all positions before market close
12:05 AM →  Retrains the ML model on today's real trade outcomes
```

---

## How the AI Learns

The bot gets smarter every single night through a self-improvement loop:

```
Week 1                    Week 2+                   Month 2+
──────────────────────    ──────────────────────    ──────────────────────
Train on 1 year of        Real trade outcomes       Win rate improves as
price history only        feed back into model      model learns what
                          each night at midnight    actually makes money

Accuracy ~80%             Accuracy improving        Accuracy growing
```

The longer it runs, the more it learns from its own real trades. It is not just backtesting — it learns from live results.

---

## Features

### 📡 News Intelligence

Scans 3 free data sources every morning at 8:30 AM ET:

| Source | What It Finds |
|---|---|
| **Alpaca News API** | Real-time articles tagged by ticker (already in your account) |
| **Yahoo Finance RSS** | Earnings beats, analyst upgrades, company news |
| **SEC EDGAR 8-K** | Merger filings the moment they are submitted — before most traders see them |

Gemini AI reads every article and produces a plain-English briefing explaining *why* each stock was chosen, what the catalyst is, and what the risk is.

### 🧠 ML Model — 49 Features from 9 Legendary Traders

The XGBoost model is trained on strategies from the world's best traders, encoded as mathematical features:

| Trader | Strategy | Key Signals |
|---|---|---|
| Stan Weinstein | Stage Analysis | Price above 150-SMA, SMA slope |
| Mark Minervini | SEPA / VCP | EMA stack, ATR contraction, distance from high |
| William O'Neil | CAN SLIM | Volume quality: up days vs down days |
| Nicolas Darvas | Box Theory | Donchian channel position, breakout |
| Paul Tudor Jones | 200-SMA Trend | Trend strength, MA slope |
| Linda Raschke | Holy Grail | ADX strength, oversold in uptrend |
| Larry Williams | %R + Temporal | Williams %R, day of week, month-end |
| John Bollinger | Band Analysis | BB width, squeeze, lower band bounce |
| Richard Donchian | Channel Breakout | 20-bar channel position, breakout signal |

### 🛡️ Risk Management

Every trade follows strict rules so one bad trade cannot hurt the account:

| Rule | Setting | Why |
|---|---|---|
| Stop loss | −2% | Auto-exit if trade goes wrong |
| Take profit | +6% (3:1 ratio) | Lock in gains at 3× the risk |
| Trailing stop | 3% from peak | Protects profits as stock rises |
| Cooldown after loss | 120 minutes | Prevents buying a falling stock again |
| Cooldown after profit | 60 minutes | Prevents buying back at the top |
| EMA exit | 2 bars required | Avoids selling on 1-candle noise |
| Max positions | 3 stocks | Stays concentrated in best ideas |
| EOD close | 3:50 PM ET | No risky overnight holds |

---

## Daily Morning Report

Every morning at 8:30 AM, the bot prints a full briefing in Render logs explaining exactly why it chose each stock:

```
══════════════════════════════════════════════════════════════════
  🎯  TODAY'S AI STOCK SELECTION — Monday July 07, 2026 · 08:30 AM ET
══════════════════════════════════════════════════════════════════

  ┌── #1 · PANW ─ M&A · HIGH IMPACT ─ 🟢 BULLISH

  │  📰 NEWS
  │     Palo Alto Networks in talks to acquire Axonius for $2.3B

  │  💡 WHY THIS MATTERS
  │     Axonius fills a critical gap in PANW's platform. Deal adds $0.40
  │     EPS within 18 months. M&A targets typically rally 15–30% on news.

  │  ✅ REASON TO BUY
  │     Institutions chase PANW on platform M&A — expect volume surge at open

  │  ⚠️  MAIN RISK
  │     Integration risk; deal premium may weigh short-term

  │  📊 SCORES
  │     News     +0.72   [████████]
  │     ML Model   68%   confidence
  │     Composite  0.581 ← final ranking score

  │  🎯 LEVELS
  │     Stop → $342.22  |  Target → $370.15  |  Risk/Reward 1:3
  └────────────────────────────────────────────────────────────────
```

The full report is also saved to `daily_picks.json` each morning.

---

## Setup

### Step 1 — Install dependencies

```bash
pip install alpaca-py xgboost scikit-learn pandas numpy \
            joblib tradingview-ta schedule \
            google-generativeai requests pytz
```

### Step 2 — Get your API keys

**Alpaca Paper Keys** (free — no real money needed)
1. Go to [alpaca.markets](https://alpaca.markets)
2. Top-left dropdown → switch to **Paper Account**
3. API Keys → Generate New Key
4. Copy the key and secret

**Gemini API Key** (free — no credit card)
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → Create API Key
3. Free tier: 15 requests/min, enough for daily use

### Step 3 — Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → **Background Worker**
3. Connect your GitHub repo
4. Set the following:

| Field | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python ai_trading_bot.py` |

5. Go to **Environment** → Add these variables:

```
ALPACA_API_KEY      =  your_paper_api_key
ALPACA_SECRET_KEY   =  your_paper_secret_key
GEMINI_API_KEY      =  your_gemini_api_key
```

6. Click **Deploy** — the bot is now live.

> ⚠️ Never hardcode API keys in the code. Always use environment variables.

---

## Files

```
AI-Paper-Trading-Bot/
│
├── ai_trading_bot.py      ← Main bot (all logic in one file)
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
│
│  Auto-created when bot runs:
│
├── ai_model.xgb           ← Saved ML model (retrained nightly)
├── trade_log.json         ← Every trade with P&L (feeds nightly retrain)
├── daily_picks.json       ← Today's AI stock picks with full reasoning
├── morning_brief.txt      ← Gemini market overview
└── ai_bot.log             ← Full bot activity log
```

---

## Monitoring

Check Render logs each morning. Here is what to look for:

| Log Message | What It Means |
|---|---|
| `Model ready — Accuracy: 80.2%` | Model retrained successfully overnight |
| `WATCHLIST UPDATE: [...] → [...]` | News intelligence picked today's stocks |
| `BUY 56×PANW @ $349.20` | Position opened |
| `SELL 56×PANW · P&L: +$1,847` | Position closed with profit |
| `COOLDOWN 94 min remaining` | Bot correctly avoiding a re-entry |
| `EOD: closing all positions` | Bot safely closing before market close |

**Win rate guide**

- Below 35% after 30+ trades → signals need review
- 35–50% → normal early-stage performance
- Above 50% → self-learning is working well

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     8:30 AM · News Scan      │
                    │  Alpaca · Yahoo · SEC EDGAR  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Gemini AI Analysis      │
                    │  Catalyst · Sentiment · Why  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │        Stock Ranker          │
                    │  News 40% + ML 35% + TA 25% │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Dynamic Watchlist (Top 5)  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Market Scan (10 min)      │
                    │   TradingView TA + XGBoost   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Trade Execution          │
                    │  Buy · Monitor · Sell · EOD  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Nightly Retrain 12:05 AM   │
                    │  Real outcomes → XGBoost     │
                    └─────────────────────────────┘
```

---

## Live Trading Bot

`trading_bot.py` is a separate rules-based bot for real money. Same watchlist, same legendary trader strategies, but no ML — pure Minervini + Raschke + PTJ logic with deterministic rules.

| | Paper Bot | Live Bot |
|---|---|---|
| Capital | $100,000 simulated | Real money |
| Signal | XGBoost ML + TradingView | Rules-based only |
| Self-learning | Yes — nightly retrain | No |
| Risk rules | 2% stop · 6% target | Same |

> Both bots read API keys from environment variables. Never hardcode real money keys.

---

## Disclaimer

This bot is built for learning and paper trading. Past performance does not guarantee future results. Always understand the risk before trading with real money.

---

*Built with Alpaca · XGBoost · Google Gemini · TradingView TA · Render*
