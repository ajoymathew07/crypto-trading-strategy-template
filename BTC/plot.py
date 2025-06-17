import pandas as pd
import numpy as np
import plotly.graph_objects as go


def sharpe_ratio_calculation(returns, risk_free_rate=7):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def sortino_ratio_calculation(returns, risk_free_rate=7):
    excess_returns = returns - risk_free_rate
    downside_std = excess_returns[excess_returns < 0].std()
    return excess_returns.mean() / downside_std

def calculate_trading_metrics(df, signal_column, initial_capital=1000, transaction_fee=0.0015, periods_per_year=365, risk_free_rate=0.07):
    # Initialize variables
    trades = []  # To keep track of closed trades
    position = 0  # 0 = no position, 1 = holding long, -1 = holding short
    buy_price = 0
    contracts = 0
    total_profit = 0
    total_closed_trades = 0
    gross_profit = 0
    gross_loss = 0
    capital = initial_capital
    peak_value = capital
    drawdown = 0
    gross_profit_long = 0
    gross_loss_long = 0
    gross_profit_short = 0
    gross_loss_short = 0

    buy_and_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    capital_history = [capital]  # Store capital at each step for return calculations
    buy_and_hold_history = [initial_capital]  # Store buy and hold portfolio values for comparison

    # Iterating till second last row
    for index, row in df.iterrows():

    # If last row, check which position is open and close it
        if index == (len(df)-1):
            if position == 1:
                sell_price = row['close']
                profit = (sell_price - buy_price) * contracts  # Profit from long
                profit -= sell_price * contracts * transaction_fee  # Deduct transaction fee on selling
                trades.append(profit)
                total_profit += profit
                capital += contracts * sell_price * (1 - transaction_fee)  # Update capital after sale
                peak_value = max(peak_value, capital)  # Update peak capital
                drawdown = max(drawdown, (peak_value - capital) / peak_value * 100)  # Calculate drawdown
                total_closed_trades += 1
                gross_profit += max(0, profit)
                gross_profit_long += max(0, profit)
                gross_loss += min(0, profit)
                gross_loss_long += min(0, profit)
                df.loc[index, signal_column] = -1  # Close position in df
                break
            elif position == -1:
                sell_price = row['close']
                profit = (buy_price - sell_price) * contracts  # Profit from short
                profit -= sell_price * contracts * transaction_fee  # Deduct transaction fee on selling
                trades.append(profit)
                total_profit += profit
                capital -= contracts * sell_price * (1 - transaction_fee)  # Update capital after closing short
                peak_value = max(peak_value, capital)  # Update peak capital
                drawdown = max(drawdown, (peak_value - capital) / peak_value * 100)  # Calculate drawdown
                total_closed_trades += 1
                gross_profit += max(0, profit)
                gross_profit_short += max(0, profit)
                gross_loss += min(0, profit)
                gross_loss_short += min(0, profit)
                df.loc[index, signal_column] = 1  # Close position in df
                break
            else:
                break

        # Handle signals
        if row[signal_column] == 1:  # Buy signal
            if position == -1:  # Close short position
                sell_price = row['close']
                profit = (buy_price - sell_price) * contracts  # Profit from short
                profit -= sell_price * contracts * transaction_fee  # Deduct transaction fee on selling
                trades.append(profit)
                total_profit += profit
                capital -= contracts * sell_price * (1 - transaction_fee)  # Update capital after closing short
                peak_value = max(peak_value, capital)  # Update peak capital
                drawdown = max(drawdown, (peak_value - capital) / peak_value * 100)  # Calculate drawdown
                total_closed_trades += 1
                gross_profit += max(0, profit)
                gross_profit_short += max(0, profit)
                gross_loss += min(0, profit)
                gross_loss_short += min(0, profit)
                position = 0  # Reset position

            elif position == 0:  # Open long position
                buy_price = row['close']
                contracts = capital / buy_price  # Calculate contracts
                capital -= contracts * buy_price * (1 + transaction_fee)  # Deduct cost + transaction fee
                position = 1  # Long position

        elif row[signal_column] == -1:  # Sell signal
            if position == 1:  # Close long position
                sell_price = row['close']
                profit = (sell_price - buy_price) * contracts  # Profit from long
                profit -= sell_price * contracts * transaction_fee  # Deduct transaction fee on selling
                trades.append(profit)
                total_profit += profit
                capital += contracts * sell_price * (1 - transaction_fee)  # Update capital after sale
                peak_value = max(peak_value, capital)  # Update peak capital
                drawdown = max(drawdown, (peak_value - capital) / peak_value * 100)  # Calculate drawdown
                total_closed_trades += 1
                gross_profit += max(0, profit)
                gross_profit_long += max(0, profit)
                gross_loss += min(0, profit)
                gross_loss_long += min(0, profit)
                position = 0  # Reset position

            elif position == 0:  # Open short position
                buy_price = row['close']
                contracts = capital / buy_price  # Calculate contracts
                capital += contracts * buy_price * (1 - transaction_fee)  # Add capital from short sale
                position = -1  # Short position

        elif row[signal_column] == 2:  # Close short and open long
            if position == -1:  # Close short
                sell_price = row['close']
                profit = (buy_price - sell_price) * contracts  # Profit from short
                profit -= sell_price * contracts * transaction_fee  # Deduct transaction fee
                trades.append(profit)
                total_profit += profit
                capital -= contracts * sell_price * (1 - transaction_fee)  # Update capital
                peak_value = max(peak_value, capital)  # Update peak capital
                drawdown = max(drawdown, (peak_value - capital) / peak_value * 100)  # Calculate drawdown
                total_closed_trades += 1
                gross_profit += max(0, profit)
                gross_profit_short += max(0, profit)
                gross_loss += min(0, profit)
                gross_loss_short += min(0, profit)

            # Open long
            buy_price = row['close']
            contracts = capital / buy_price  # Calculate contracts
            capital -= contracts * buy_price * (1 + transaction_fee)  # Deduct cost + transaction fee
            position = 1  # Update position to long

        elif row[signal_column] == -2:  # Close long and open short
            if position == 1:  # Close long
                sell_price = row['close']
                profit = (sell_price - buy_price) * contracts  # Profit from long
                profit -= sell_price * contracts * transaction_fee  # Deduct transaction fee
                trades.append(profit)
                total_profit += profit
                capital += contracts * sell_price * (1 - transaction_fee)  # Update capital
                peak_value = max(peak_value, capital)  # Update peak capital
                drawdown = max(drawdown, (peak_value - capital) / peak_value * 100)  # Calculate drawdown
                total_closed_trades += 1
                gross_profit += max(0, profit)
                gross_profit_long += max(0, profit)
                gross_loss += min(0, profit)
                gross_loss_long += min(0, profit)

            # Open short
            buy_price = row['close']
            contracts = capital / buy_price  # Calculate contracts
            capital += contracts * buy_price * (1 - transaction_fee)  # Add capital from short sale
            position = -1  # Update position to short

        # Calculate portfolio value
        sell_price = row['close']
        if position == 1:  # Holding long
            portfolio_value = capital + contracts * sell_price
        elif position == -1:  # Holding short
            portfolio_value = capital - contracts * sell_price
        else:  # No position
            portfolio_value = capital

        capital_history.append(portfolio_value)

    # Buy and hold: calculate hypothetical capital if we bought and held from the start
        buy_and_hold_value = initial_capital * (row['close'] / df['close'].iloc[0])
        buy_and_hold_history.append(buy_and_hold_value)

    # Calculating metrics
    net_long_profit = gross_profit_long + gross_loss_long
    net_short_profit = gross_profit_short + gross_loss_short
    net_profit = net_long_profit + net_short_profit
    percent_returns = net_profit / initial_capital * 100
    win_trades = [trade for trade in trades if trade > 0]
    lose_trades = [trade for trade in trades if trade < 0]
    
    win_rate = len(win_trades) / total_closed_trades if total_closed_trades > 0 else 0
    max_drawdown = drawdown  # Already calculated as a percentage
    average_winning_trade = np.mean(win_trades) if win_trades else 0
    average_losing_trade = np.mean(lose_trades) if lose_trades else 0
    largest_winning_trade = max(win_trades) if win_trades else 0
    largest_losing_trade = min(lose_trades) if lose_trades else 0

    # Calculate portfolio returns based on capital changes
    capital_history = np.array(capital_history)
    portfolio_returns = np.diff(capital_history) / capital_history[:-1]  # Returns as pct change in capital

    # Risk free rate per timeframe
    risk_free_rate_per_period = risk_free_rate / periods_per_year
    excess_returns = portfolio_returns - risk_free_rate_per_period

    # Calculate Sharpe and Sortino ratios based on portfolio returns
    sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns, ddof=1) * np.sqrt(periods_per_year)
    downside_std = np.std(portfolio_returns[portfolio_returns < 0], ddof=1)
    sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    

    # Results
    metrics = {
        'Initial Capital (USDT)': initial_capital,
        'Ending Capital (USDT)': round(initial_capital + total_profit, 2),
        'Gross Profit': round(gross_profit, 2),
        'Net Profit': round(net_profit, 2),
        'Gross Loss': round(abs(gross_loss), 2),
        'Strategy Returns (%)': round(percent_returns, 2),
        'Buy and Hold Return (%)': round(buy_and_hold_return, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Total Closed Trades': round(total_closed_trades, 2),
        'Win Rate (%)': round(win_rate * 100, 2),
        'Gross Long Profit': round(gross_profit_long, 2),
        'Gross Short Profit': round(gross_profit_short, 2),
        'Net Long Profit': round(net_long_profit, 2),
        'Net Short Profit': round(net_short_profit, 2),
        'Average Winning Trade (USDT)': round(average_winning_trade, 2),
        'Average Losing Trade (USDT)': round(average_losing_trade, 2),
        'Largest Losing Trade (USDT)': round(largest_losing_trade, 2),
        'Largest Winning Trade (USDT)': round(largest_winning_trade, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Sortino Ratio': round(sortino_ratio, 2),
    }

    fig = go.Figure()

    # Plot strategy portfolio value
    fig.add_trace(go.Scatter(x=list(range(len(capital_history))), y=capital_history,
                             mode='lines', name='Strategy Portfolio Value', line=dict(color='blue')))

    # Plot buy and hold portfolio value
    fig.add_trace(go.Scatter(x=list(range(len(buy_and_hold_history))), y=buy_and_hold_history,
                             mode='lines', name='Buy and Hold Value', line=dict(color='green')))

    # Customize layout
    fig.update_layout(
        title='Strategy Portfolio Value',
        xaxis_title='Time (Number of Rows)',
        yaxis_title='Portfolio Value (USDT)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        height=600
    )

    fig.show()
    return metrics

def plotBuySellSignal(df, signal_column, indicator_on_graph=None):
    # Plotting Buy Sell Calls
    fig = go.Figure()

    # Plot close prices
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'], mode='lines', name='Close', 
        line=dict(color='lightblue')
    ))

    # Plot buy signals (+1)
    buy_signals = df[df[signal_column] == 1]
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['close'],
        mode='markers', name='Buy Signal (+1)',
        marker=dict(symbol='triangle-up', color='green', size=6)
    ))

    # Plot sell signals (-1)
    sell_signals = df[df[signal_column] == -1]
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['close'],
        mode='markers', name='Sell Signal (-1)',
        marker=dict(symbol='triangle-down', color='red', size=6)
    ))

    # Plot long close/open short signals (-2)
    close_long_signals = df[df[signal_column] == -2]
    fig.add_trace(go.Scatter(
        x=close_long_signals.index, y=close_long_signals['close'],
        mode='markers', name='Close Long/Open Short Signal (-2)',
        marker=dict(symbol='triangle-down', color='orange', size=6)
    ))

    # Plot short close/open long signals (+2)
    close_short_signals = df[df[signal_column] == 2]
    fig.add_trace(go.Scatter(
        x=close_short_signals.index, y=close_short_signals['close'],
        mode='markers', name='Close Short/Open Long Signal (+2)',
        marker=dict(symbol='triangle-up', color='purple', size=6)
    ))

    # Plot optional indicators
    if indicator_on_graph:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[indicator_on_graph], 
            mode='lines', name=indicator_on_graph, 
            line=dict(color='pink')
        ))

    # Customize layout
    fig.update_layout(
        title=f'{signal_column} Buy Sell Calls',
        xaxis_title='Time (Number of bars)',
        yaxis_title='Closing Price (USDT)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        height=600
    )

    fig.show()


import pandas as pd

# Step 1: Load your CSV data
df = pd.read_csv('output.csv', parse_dates=['datetime'])




metrics = calculate_trading_metrics(df, 'signals')
plotBuySellSignal(df, 'signals')

# The function will return the metrics and plot the strategy portfolio value
print(metrics)