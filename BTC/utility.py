import numpy as np
import pandas as pd
from base import Strategy
import MetricUtility
'''
Utility objects help you get metrics over a specified time range for a given strategy.
This is needed to evaluate strategies at the start of every window to assign weights.

How to use
1. Initialise with a Strategy object, the risk-free-rate (6.9% currently) and transaction fees (0.15% in our case)
2. Request a metric by the getMetric method
    example:- getMetric("Drawdown", 100, 150) returns the maximum drawdown over the period [100, 150)
'''
class Utility:
    def __init__(self, strategy : Strategy, rfr=0.069, transaction_cost=0.0015, compound=False, initial_capital=1000) -> None:
        self.name = strategy.name
        self.portfolio_available = False
        self.prices = Strategy.getPrices(strategy)
        self.signals = Strategy.getSignals(strategy)
        self.risk_free_rate = rfr
        self.transaction_fee = transaction_cost
        self.initial_capital = initial_capital
        self.portfolio = MetricUtility.getPortfolio(self.signals, self.prices, initial_capital, compound, self.transaction_fee, 0, len(self.prices))
        self.realised_portfolio = MetricUtility.getPortfolio(self.signals, self.prices, initial_capital, compound, transaction_cost, 0, len(self.prices), False)
        self.datetime = pd.to_datetime(strategy.data["datetime"])
        timediff = self.datetime.diff().dt.total_seconds()
        self.time_period_in_minutes = timediff.iloc[1] / 60

    def getMetric(self, metric: str, start=0, end=-1, unrealised=True):
        if end == -1:
            end = len(self.prices)
        if unrealised:
            portfolio = self.portfolio[start : end]
        else:
            portfolio = self.realised_portfolio[start : end]
        trade_returns = MetricUtility.getTradeReturns(self.portfolio[start : end], self.signals[start : end])
        match metric:
            case "PnL":
                return portfolio[-1] - portfolio[0]
            case "Drawdown":
                return MetricUtility.getMaxDrawdownPercentageList(portfolio)[-1]
            case "PnLByDrawdown":
                return MetricUtility.getPnLByDrawdownList(portfolio)[-1]
            case "InverseDrawdown":
                dd = MetricUtility.getMaxDrawdownPercentageList(portfolio)[-1]
                return 1/(0.001 + dd)
            case "TradeReturns":
                return sum(trade_returns)/(sum([1 if i != 0 else 0 for i in trade_returns])+0.00001)
            case "Sharpe":
                return MetricUtility.getSharpeRatio(portfolio, self.risk_free_rate, self.time_period_in_minutes)
            case "NumTrades":
                return sum([1 for signal in self.signals if signal > 0])
            case "WinRate":
                return MetricUtility.getWinLossRatio(trade_returns)
            case "MaxWin":
                mx = 0
                for trade in trade_returns:
                    mx = max(mx, trade)
                return mx
            case "AvgWin":
                total_win = sum(trade for trade in trade_returns if trade > 0)
                wins = sum(1 for trade in trade_returns if trade > 0)
                return total_win / wins
            case "MaxLoss":
                mn = 0
                for trade in trade_returns:
                    mn = min(mn, trade)
                return mn
            case "AvgLoss":
                total_loss = sum (trade for trade in trade_returns if trade < 0)
                losses = sum(1 for trade in trade_returns if trade < 0)
                return total_loss / losses
            case _:
                return portfolio[-1]
        
    def getMetricList(self, metric, start, end, unrealised=True):
        if unrealised:
            portfolio = self.portfolio[start : end]
        else:
            portfolio = self.realised_portfolio[start : end]
        trade_returns = MetricUtility.getTradeReturns(portfolio, self.signals[start : end])
        match metric:
            case "PnL":
                return MetricUtility.getPnLList(portfolio)
            case "Drawdown":
                return MetricUtility.getMaxDrawdownPercentageList(portfolio)
            case "PnLByDrawdown":
                return MetricUtility.getPnLByDrawdownList(portfolio)
            case "InverseDrawdown":
                dd = MetricUtility.getMaxDrawdownPercentageList(portfolio)
                return 1/(0.001 + dd)
            case "TradeReturns":
                return trade_returns
            case "ClosePrices":
                return self.prices
            case "CAGR":
                return MetricUtility.getCAGRList(portfolio, self.time_period_in_minutes)
            case "Calmar":
                return MetricUtility.getCalmarList(portfolio, self.time_period_in_minutes)
            case "ProfitFactor":
                return MetricUtility.getProfitFactorList(trade_returns)
            case "WinRate":
                return MetricUtility.getWinLossRatioList(trade_returns)
            case _:
                return portfolio
