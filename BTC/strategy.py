import pandas as pd
import numpy as np
from scipy import stats

# Load data
# data = pd.read_csv('../data/BTC_data/BTC_2019_2023_1d.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/BTC_data/BTC_1d_2024.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/ETH_data/ETH_2019_2023_1d.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/ETH_data/ETH_1d_2024.csv', parse_dates=['datetime'])
data = pd.read_csv('../data/SOL_data/SOL_1d_2023.csv', parse_dates=['datetime'])




data=data.copy()

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.

    Parameters:
        data (pd.DataFrame): DataFrame with 'close' column.
        period (int): Lookback period for SMA.
        std_dev (float): Standard deviation multiplier.

    Returns:
        pd.DataFrame: DataFrame with added 'Upper Band', 'Lower Band', and 'Middle Band' columns.
    """
    data['Middle Band'] = data['close'].rolling(window=period).mean()
    data['Std Dev'] = data['close'].rolling(window=period).std()

    data['Upper Band'] = data['Middle Band'] + (std_dev * data['Std Dev'])
    data['Lower Band'] = data['Middle Band'] - (std_dev * data['Std Dev'])

    return data
def calculate_supertrend(data, period=7, multiplier=3):
    data['H-L'] = data['high'] - data['low']
    data['H-C'] = np.abs(data['high'] - data['close'].shift(1))
    data['L-C'] = np.abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['H-L', 'H-C', 'L-C']].max(axis=1)
    data['ATR'] = data['TR'].rolling(period).mean()

    hl2 = (data['high'] + data['low']) / 2
    data['Upper_Band'] = hl2 + (multiplier * data['ATR'])
    data['Lower_Band'] = hl2 - (multiplier * data['ATR'])

    data['Supertrend'] = 0
    in_uptrend = True

    for i in range(1, len(data)):
        if data.loc[i, 'close'] > data.loc[i - 1, 'Upper_Band']:
            in_uptrend = True
        elif data.loc[i, 'close'] < data.loc[i - 1, 'Lower_Band']:
            in_uptrend = False

        if in_uptrend:
            data.loc[i, 'Supertrend'] = data.loc[i, 'Lower_Band']
        else:
            data.loc[i, 'Supertrend'] = data.loc[i, 'Upper_Band']

    return data



def calculate_heikin_ashi(data):
    """
    Calculate Heikin-Ashi candles from OHLC data.

    Parameters:
        data (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
        pd.DataFrame: DataFrame with the added 'HA_open', 'HA_high', 'HA_low', and 'HA_close' columns.
    """
    # Ensure the data is copied to avoid modifying the original DataFrame
    ha_data = data.copy()

    # Calculate the Heikin-Ashi close
    ha_data['HA_close'] = (ha_data['open'] + ha_data['high'] + ha_data['low'] + ha_data['close']) / 4

    # Initialize the Heikin-Ashi open
    ha_data['HA_open'] = (ha_data['open'].shift(1) + ha_data['close'].shift(1)) / 2
    ha_data.loc[0, 'HA_open'] = (ha_data['open'][0] + ha_data['close'][0]) / 2  # Special case for the first row

    # Calculate the Heikin-Ashi high and low
    ha_data['HA_high'] = ha_data[['HA_open', 'HA_close', 'high']].max(axis=1)
    ha_data['HA_low'] = ha_data[['HA_open', 'HA_close', 'low']].min(axis=1)

    return ha_data[['HA_open', 'HA_high', 'HA_low', 'HA_close']]



def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

data= calculate_bollinger_bands(data)
data= calculate_supertrend(data)
data = calculate_rsi(data)
data[['HA_open', 'HA_high', 'HA_low', 'HA_close']]= calculate_heikin_ashi(data)


# ATR (Average True Range)
data['H-L'] = data['high'] - data['low']
data['H-C'] = np.abs(data['high'] - data['close'].shift(1))
data['L-C'] = np.abs(data['low'] - data['close'].shift(1))
data['TR'] = data[['H-L', 'H-C', 'L-C']].max(axis=1)
data['ATR'] = data['TR'].rolling(7).mean()

# Initialize signals, positions, and trade_type
data['trade_type'] = 'hold'
position = [0]*10
data['signals'] = 0
stop_loss=0  # Stop loss column
# data['leverage']=2
data['position']=100

# ATR multiplier for stop loss
atr_multiplier = 1.5
holding_period=0


'''
    [CONFIDENTIAL] Strategy logic omitted for confidentiality.
    Given below is an example strategy for reference
'''
for i in range(10, len(data)):
    rsi = data.loc[i, 'RSI']
    close_price = data.loc[i, 'close']
    prev_position = position[len(position)-1]
    atr = data.loc[i, 'ATR']
    
    # Entry signals
    if prev_position == 0:  # No position
        # Long entry condition
        if rsi<30 :  # Long entry signal (momentum up)
            data.loc[i+1, 'trade_type'] = 'Long'
            data.loc[i, 'signals'] = 1
            stop_loss = (close_price - atr * atr_multiplier)
            data.loc[i,'position']=100

        
        # Short entry condition
        elif data.loc[i,'HA_open']==data.loc[i,'HA_high']:  # Short entry signal (momentum down)
            data.loc[i+1, 'trade_type'] = 'Short'
            data.loc[i, 'signals'] = -1
            stop_loss = (close_price + atr * atr_multiplier)
            data.loc[i,'position']=100

    # Exit signals 
    elif prev_position == 1:  # Long position
        # Strong reversal signal to close long and enter short 
        if data.loc[i,'HA_open']==data.loc[i,'HA_high'] :
            data.loc[i+1, 'trade_type'] = 'close Long & Short'
            data.loc[i, 'signals'] = -2
            stop_loss = 0
        # Exit long if price hits stop-loss
        if  close_price < stop_loss:  
            data.loc[i+1, 'trade_type'] = 'Exit Long'
            data.loc[i, 'signals'] = -1
            stop_loss = 0

    elif prev_position == -1:  # Short position
        
        # Strong reversal signal to close short and enter long 
        if data.loc[i,'HA_open']==data.loc[i,'HA_low']:
            data.loc[i+1, 'trade_type'] = 'close Long & Short'
            data.loc[i, 'signals'] = 2
            stop_loss = 0
        # Exit short price hits stop-loss
        if close_price > stop_loss:
            data.loc[i+1, 'trade_type'] = 'Exit Short'
            data.loc[i, 'signals'] = 1
            stop_loss = 0
    # Update position based on signals
    position.append(prev_position+data.loc[i,'signals'])
data.loc[len(data)-1,'signals']=-position[len(data)-2]
position[len(data)-1]=0  # Closing the last trade

print(len(position)," ",len(data))
print(position[-5:])
   
# Output to CSV
output_path = 'output.csv'
data[['datetime', 'open', 'high', 'low', 'close','signals', 'trade_type','position']].to_csv(output_path, index=False)

print(f"Output saved to {output_path}")