# QQQ Mean Reversion Strategy

A hobby project where I'm trying to learn Python and trading by building a bot that trades QQQ. **I have no idea what I'm doing** - this is just for fun and learning!

## 🤔 What is this?

I built a bot that tries to buy QQQ when it's "oversold" and sell when it's "overbought". The idea is that prices bounce back to their average (mean reversion). Will it work? Probably not, but it's been fun to code!

## 🚀 Quick Start

```bash
# Clone this repo
git clone https://github.com/KamvysisPuzzles/im-qqqooked.git
cd im-qqqooked

# Install stuff
pip install -r requirements.txt

# See how it would have done (probably badly)
python main.py --mode backtest
```

## ⚙️ Settings

Mess with `config/strategy_config.yaml`:
- `lookback_period`: How many hours to look back (default: 20)
- `threshold_multiplier`: How "far" is too far from average (default: 2.0)
- `stop_loss_pct`: Get out if you lose this much (default: 2%)
- `take_profit_pct`: Take money if you make this much (default: 3%)

## ⚠️ IMPORTANT

**DON'T USE REAL MONEY!** 

This is just a learning project. I'm not a trader, I don't know what I'm doing, and this will probably lose money. Use at your own risk!

## 🤝 Want to help?

Feel free to mess around with it! I'd love bug fixes, better ideas, or ways to make it less terrible.

---

**Good luck! 📈 (but probably 📉)**