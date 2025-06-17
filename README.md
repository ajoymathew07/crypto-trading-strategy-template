# Crypto Trading Strategy Template 🚀

This repository contains a modular and extensible framework for designing, backtesting using untrade client, and evaluating crypto trading strategies. It was developed solely as a learning exercise, and core strategy components have been intentionally withheld to maintain confidentiality.

## 🧠 Overview

- Built in Python
- Uses [`untrade`](https://github.com/Untrade/untrade) backtester for simulation
- Modular architecture: signal generation, execution, and performance tracking
- Plug-and-play strategy template
- Confidential strategy logic excluded from this public release

---
## 📁 Setup
``` bash
git clone https://github.com/ajoymathew07/crypto-trading-strategy-template.git
cd crypto-trading-strategy-template/
```

## ⚙️ Untrade Backtester Integration

This project uses the [Untrade SDK](https://docs-quant.untrade.io/UntradeSDK.html) to perform high-speed backtesting of crypto trading strategies.

The backtester is designed to work with OHLCV data in `.csv` format and supports large file sizes via chunking (for files > 100MB).

### 🔑 Jupyter ID (Required)

The Untrade SDK requires a unique `jupyter_id` to run the backtester. This is used to link your results on [jupyter.untrade.io](https://jupyter.untrade.io).

## 📊 Backtest Results on BTC data
Backtest results on BTC_2019_2023_1d.csv (in data folder)with an initial capital of $1000 using the Untrade SDK.
Account transaction costs and sliipage is at a rate o 0.15 percent per transaction.
**Static Statistics:**
- 📈 **Total Trades:** 95  
- ✅ **Winning Trades:** 41  
- ❌ **Losing Trades:** 54  
- 🔁 **Win Rate:** 43.16%  
- 💰 **Gross Profit:** \$4,698.14  
- 🧾 **Net Profit:** \$4,555.64  
- 📊 **Sharpe Ratio:** 4.02  
- 📈 **Sortino Ratio:** 21.08  
- 🕒 **Average Holding Time:** 15 days 2h 46m  
- 📉 **Maximum Drawdown (%):** 11.05%  
- 💵 **Benchmark Return (%):** 325.63%

**Compound Statistics:**
- 🧧 **Initial Balance:** \$1,000  
- 💹 **Final Balance:** \$20,413.36  
- 🚀 **Total Return (%):** 1,941.34%  
- 📉 **Maximum Drawdown (%):** 30.27%  
- 🕒 **Max Time to Recovery:** 138 days  
- 🧾 **Total Fee Paid:** \$1,104.51  


