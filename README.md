# Weex Trading Bot 🤖

A self-improving ML trading bot for the Weex exchange.
Supports BTC, ETH, SOL (and any Weex spot pair) with leverage.

---

## ⚡ Quick Start

### 1. Install Python dependencies
```bash
cd trading_bot
pip install -r requirements.txt
```

### 2. Create your Weex account & API keys

1. Go to [weex.com](https://www.weex.com) and register
2. Complete KYC verification
3. Go to **Account → API Management → Create API Key**
4. Give it a name, set **Read** + **Trade** permissions
5. Save your **API Key**, **Secret Key**, and **Passphrase** securely

> ⚠️ Never share your API keys with anyone.

### 3. Configure the bot

Edit `config.yaml`:

```yaml
exchange:
  api_key:    "paste_your_api_key_here"
  api_secret: "paste_your_secret_key_here"
  passphrase: "paste_your_passphrase_here"

trading:
  paper_trading: true   # ← Start here! Change to false for live.
  leverage: 5           # ← Set your desired leverage (e.g. 3, 5, 10)
```

### 4. Run in paper mode first

```bash
python bot.py
```

Watch the logs. The bot will:
- Fetch 300 candles of history
- Train its ML model
- Simulate trades (no real money) and log them to `logs/trades.csv`

### 5. Switch to live trading

Once you're satisfied with paper results, open `config.yaml` and set:
```yaml
paper_trading: false
```

Then restart the bot.

---

## 🧠 How the Bot Learns

| Component | Detail |
|-----------|--------|
| **Features** | 22 technical indicators (RSI, MACD, Bollinger Bands, EMA, ATR, volume ratio, momentum) |
| **Model** | Random Forest Classifier (200 trees) |
| **Labels** | BUY if price rises >0.5% in next 4 candles; SELL if falls >0.5%; else HOLD |
| **Retraining** | Automatically retrains every 40 completed trades using all history |
| **Pair ranking** | Tracks PnL per pair and gradually shifts capital to better performers |

---

## ⚙️ Key Settings (config.yaml)

| Setting | Default | What it does |
|---------|---------|--------------|
| `paper_trading` | `true` | Simulate trades without real money |
| `leverage` | `1` | Multiplier (e.g. 5 = 5× leverage) |
| `risk_per_trade_pct` | `0.02` | Risk 2% of equity per trade |
| `max_daily_loss_pct` | `0.05` | Auto-halt if you lose 5% in a day |
| `stop_loss_atr_mult` | `1.5` | Stop-loss = 1.5× ATR from entry |
| `take_profit_atr_mult` | `3.0` | Take-profit = 3× ATR (2:1 R/R) |
| `timeframe` | `60` | Candle interval in minutes |
| `loop_interval_s` | `3600` | How often the bot runs (seconds) |

---

## 📊 Output Files

| File | Contents |
|------|---------|
| `logs/bot.log` | Full activity log |
| `logs/trades.csv` | Every trade: entry, exit, PnL, reason |
| `models/rf_model.joblib` | Saved ML model (survives restarts) |
| `models/scaler.joblib` | Feature scaler |

---

## ⚠️ Risk Warning

Trading with leverage is high risk. This bot is **not** financial advice.
- Always start with paper trading
- Use only capital you can afford to lose
- The `max_daily_loss_pct` setting is your safety net — keep it enabled
- Monitor the bot regularly; don't leave it running unattended for days

---

## 🔧 Supported Pairs (Weex Spot format)

| Pair | Symbol |
|------|--------|
| BTC/USDT | `BTCUSDT_SPBL` |
| ETH/USDT | `ETHUSDT_SPBL` |
| SOL/USDT | `SOLUSDT_SPBL` |
| Any other | `XXXUSDT_SPBL` |

To add more pairs, just add them to the `pairs` list in `config.yaml`.
