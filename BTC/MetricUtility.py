'''
Module Interface -

getMaxDrawdownPercentageList(portfolio) -> returns list of maximum drawdowns over the entire period

getPnLList(portfolio)                   -> returns list of PnL values over the entire period

getPnLByDrawdownList(portfolio)         -> returns list of PnL/Maximum Drawdown over the entire period

getSharpeRatioList(portfolio, risk_free_rate, time_period_in_minutes)   -> returns list of sharpe ratio over the entire period

getSortinoRatioList(portfolio, risk_free_rate, weights)                 -> returns list of sortino ratio over the entire period

getCalmarList(portfolio, time_period_in_minutes)    -> returns calmar ratio over the entire period

getProfitFactorList(trade_returns)                  -> returns profit factor over the entire period

getWinLossRatioList(trade_returns)                  -> returns win / loss ratio (win rate) over the entire period
'''

from typing import Optional
import numpy as np

def getPortfolio(signals, prices, initial_capital, compound, transaction_fee, start_at= 0, end_at: Optional[int] = None, unrealised=True):
    cash = initial_capital
    portfolio = []
    position = 0
    invest = initial_capital
    for i in range(0, len(prices)):
        if i == len(prices)-1:
            signals[i] = -position
        if compound:
            invest = cash
        if not unrealised:
            portfolio.append(cash)
        if position==0:
            if unrealised:
                portfolio.append(cash)
            if signals[i] == 1:
                entry_price = prices[i]
            elif signals[i] == -1:
                entry_price = prices[i]
        elif position == 1:
            if unrealised:
                portfolio.append(cash - invest + invest/entry_price*prices[i])
            if signals[i] == -1:
                change = invest/entry_price*(prices[i] - entry_price)
                cash += change
                cash -= invest*transaction_fee
        else:
            if unrealised:
                portfolio.append(cash + invest - invest/entry_price*prices[i])
            if signals[i]==1:
                change = invest/entry_price*(entry_price - prices[i])
                cash += change
                cash -= transaction_fee*invest
        position += signals[i]
    return portfolio
                    
def getTradeReturns(portfolio, signals):
    trade_returns = []
    entry = None  # Track the price at which we enter
    type = None
    for i in range(len(portfolio)):
        signal = signals[i]
        if signal != 0:
            if entry == None:               # entry point
                entry = portfolio[i]
                trade_returns.append(0)
                type = signal
            else:                           # exit point
                exit = portfolio[i]
                trade_returns.append((exit - entry)*(type))
                entry = None
                type = None
        else:
            trade_returns.append(0)
            
    return trade_returns

def getMaxDrawdownPercentageList(portfolio_values):
    # DD for first frame is zero
    dd_list = [0]
    peak_value = portfolio_values[0]
    for i in range(1, len(portfolio_values)):
        peak_value = max(peak_value,portfolio_values[i])
        drawdown = (peak_value-portfolio_values[i]) / peak_value * 100
        dd_list.append(max(dd_list[-1], drawdown))
    return dd_list

def getPnLList(portfolio_values):
    pnls = []
    for i in range(0, len(portfolio_values)):
        pnls.append(portfolio_values[i]-portfolio_values[0])
    return pnls

def getPnLByDrawdownList(portfolio):
    dd_list = getMaxDrawdownPercentageList(portfolio)
    mtrc = []
    for i in range(0, len(portfolio)):
        mtrc.append((portfolio[i]-portfolio[0])/(0.001 + dd_list[i]))
    return mtrc


'''
risk-free rate is 6.9% currently
time_period_in_minutes is about the dataset used, for 1 day data, it is 1440
'''
def getSharpeRatio(portfolio, risk_free_rate, time_period_in_minutes):
    portfolio = np.array(portfolio)
    # Calculate the number of periods in a year based on the time period
    periods_per_year = 252 * (1440 / time_period_in_minutes)
    # Calculate percentage change
    daily_return = np.diff(portfolio) / portfolio[:-1]
    # Calculate the Sharpe Ratio (mean excess return divided by std dev of returns)
    excess_return = daily_return.mean() - (risk_free_rate / periods_per_year)
    Sharpe_Ratio = excess_return / daily_return.std()
    # Annualize the Sharpe Ratio
    A_SharpeRatio = (periods_per_year ** 0.5) * Sharpe_Ratio
    return A_SharpeRatio

def getSharpeRatioList(portfolio, risk_free_rate, time_period_in_minutes):
    local_portfolio = []
    sharpe_list = []
    for i in range(0, len(portfolio)):
        local_portfolio.append(portfolio[i])
        sharpe_list.append(getSharpeRatio(local_portfolio, risk_free_rate, time_period_in_minutes))
    return sharpe_list

def getSortinoRatio(portfolio, risk_free_rate, weights = [1.0]):
    returns = portfolio.pct_change(1).dropna()
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns)
    exp_rets = returns.mean()
    mean = np.dot(exp_rets, weights)
    excess_return = mean - risk_free_rate
    sortino_ratio = excess_return / downside_std
    return sortino_ratio

def getSortinoRatioList(portfolio, risk_free_rate, weights = [1.0]):
    sortino_list = []
    local_portfolio = []
    for i in range(0, len(portfolio)):
        local_portfolio.append(portfolio[i])
        sortino_list.append(getSortinoRatio(local_portfolio, risk_free_rate, weights))
    return sortino_list

def getCAGRList(portfolio, time_period_in_minutes):
    cagr = [0]
    starting_value = portfolio[0]
    for i in range(1, len(portfolio)):
        ending_value = portfolio[i]
        cagr.append(((ending_value / starting_value) ** (525600/(i*time_period_in_minutes))) - 1)
    return cagr

def getCalmarList(portfolio, time_period_in_minutes):
    calmar = []
    cagr = getCAGRList(portfolio, time_period_in_minutes)
    mdd = getMaxDrawdownPercentageList(portfolio)
    for i in range(len(portfolio)):
        calmar.append(cagr[i] / mdd[i])
    return calmar

def calculateProfitFactor(trade_returns):
    total_profit = sum(trade for trade in trade_returns if trade > 0)
    total_loss = abs(sum(trade for trade in trade_returns if trade < 0))
    
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')  # Handle zero loss case
    return profit_factor

def getProfitFactorList(trade_returns):
    win_rets = 0
    loss_rets = 0
    pflist = []
    for trade in trade_returns:
        if trade > 0:
            win_rets += trade
        else:
            loss_rets -= trade
        if loss_rets == 0:
            pflist.append('inf')
        else:
            pflist.append(win_rets / loss_rets)
    return pflist

def getWinLossRatio(trade_returns):
    wins = sum(1 for trade in trade_returns if trade > 0)
    losses = sum(1 for trade in trade_returns if trade < 0)
    return wins / (wins + losses+0.00001)

def getWinLossRatioList(trade_returns):
    wins = 0
    losses = 0
    wlrlist = []
    for i in range(0, len(trade_returns)):
        if (trade_returns[i] > 0):
            wins += 1
        elif (trade_returns[i] < 0):
            losses -= 1
        if losses == 0:
            wlrlist.append('inf')
        else:
            wlrlist.append(wins/losses)
    return wlrlist
