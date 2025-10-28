# 3EMA + MACD-V + Aroon Trend Following Strategy

A trend-following trading strategy for TQQQ (Triple-leveraged QQQ ETF) using an ensemble of three indicators: Triple EMA, MACD-Volume, and Aroon.

## 🎯 Strategy Overview

This is a **trend-following** strategy that combines three technical indicators with **OR logic** (any indicator can trigger a signal):

### Strategy Components

1. **Triple EMA (3EMA)** - Fast (12), Medium (89), Slow (125) exponential moving averages
   - BUY: Any EMA crosses above another
   - SELL: Any EMA crosses below another

2. **MACD-Volume (MACD-V)** - Volume-confirmed MACD
   - BUY: MACD bullish cross + volume confirmation
   - SELL: MACD bearish cross + volume confirmation

3. **Aroon Indicator** - Trend strength indicator (length 66)
   - BUY: Aroon Up crosses above Aroon Down
   - SELL: Aroon Up crosses below Aroon Down

### Key Parameters

- **Triple EMA**: 12, 89, 125 periods
- **MACD-V**: Fast=25, Slow=30, Signal=85
- **Aroon Length**: 66 periods
- **Exit Strategy**: Exit only on opposite signal (no stop loss/take profit/holding period)

## 📊 Strategy Performance

Based on backtesting from 2018-2025:

- **Total Return**: 1193.53% (in-sample), 329.39% (out-of-sample)
- **Sharpe Ratio**: 2.005 (in-sample), 1.869 (out-of-sample)
- **Win Rate**: 70-73%
- **Profit Factor**: 5.75-6.45
- **Trades per Year**: ~4-5 trades

## 🚀 Quick Start

### Installation

```bash
# Clone this repo
git clone https://github.com/KamvysisPuzzles/im-qqqooked.git
cd im-qqqooked

# Install dependencies
pip install -r requirements.txt

# Run backtest
python main.py --mode backtest
```

### Run Backtest

```bash
python main.py --mode backtest
```

This will:
- Download TQQQ historical data
- Run the strategy backtest
- Generate performance metrics
- Create visualizations in `results/`

### Run Live Monitoring (Manual Execution)

For live trading with Telegram alerts:

```bash
python main.py --mode live
```

**Note**: This generates signals for **manual execution** via Telegram alerts. You execute trades manually in your broker.

### Setup Telegram Alerts

1. Get your Telegram bot token from [@BotFather](https://t.me/botfather)
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Update `config/strategy_config.yaml`:

```yaml
live:
  enabled: true
  telegram_enabled: true
  telegram_bot_token: "YOUR_BOT_TOKEN"
  telegram_chat_id: "YOUR_CHAT_ID"
  signal_check_interval: 3600  # Check every hour
```

## 📁 Repository Structure

```
qqqooked/
├── notebooks/
│   ├── strategies/          # Strategy implementation notebooks
│   │   └── 3EMA_MACDV_Aroon.ipynb
│   └── analysis/            # Parameter optimization notebooks
│       ├── 3EMA.ipynb
│       ├── MACD.ipynb
│       └── Aroon.ipynb
├── src/
│   ├── strategy.py          # 3EMA + MACD-V + Aroon strategy
│   ├── data_fetcher.py     # Data fetching module
│   ├── backtest.py          # Backtesting engine
│   └── live_monitor.py     # Live signal monitoring
├── config/
│   └── strategy_config.yaml # Strategy configuration
├── results/                 # Backtest results & plots
├── logs/                    # Log files
└── main.py                  # Entry point
```

## ⚙️ Configuration

Edit `config/strategy_config.yaml` to customize:

```yaml
# Strategy Parameters
strategy:
  ema1: 12
  ema2: 89
  ema3: 125
  macd_fast: 25
  macd_slow: 30
  macd_signal: 85
  aroon_length: 66

# Risk Management
risk:
  stop_loss_pct: null    # Disabled - no stop loss
  take_profit_pct: null  # Disabled - no take profit
  max_holding_days: null # Disabled - no max holding period
```

## 📈 Strategy Logic

### Signal Generation (OR Logic)

The strategy generates BUY signals when **ANY** of the three indicators triggers:

1. **3EMA Signal**: Any EMA crosses above another
2. **MACD-V Signal**: MACD crosses above signal line + volume confirmation
3. **Aroon Signal**: Aroon Up crosses above Aroon Down

### Exit Logic

Positions exit on:
- **Opposite Signal**: Any SELL signal from indicators triggers exit
- No stop loss, take profit, or maximum holding period

### Position Management

- **Direction**: LONG only (trend-following)
- **Position Size**: 10% of portfolio
- **Maximum Positions**: 1 concurrent position
- **Exits**: Only on opposite signals (pure trend following)

## ⚠️ DISCLAIMER

**THIS IS NOT FINANCIAL ADVICE**

- This is a learning/research project
- Past performance does not guarantee future results
- Trading leveraged ETFs (like TQQQ) is highly risky
- Always do your own research
- Use at your own risk

**DO NOT RISK MORE THAN YOU CAN AFFORD TO LOSE**

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Share your results
- Fork and experiment!

## 📝 License

This project is for educational purposes only. Use at your own discretion.

---

**Strategy Type**: Trend Following  
**Target Instrument**: TQQQ (3x QQQ)  
**Market**: US Stock Market  
**Timeframe**: Daily bars  
**Execution**: Manual (via Telegram alerts)

**Good luck!**
