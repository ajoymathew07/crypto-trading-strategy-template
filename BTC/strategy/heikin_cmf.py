import pandas as pd
import numpy as np
from scipy import stats

# Load data
data = pd.read_csv('../../data/BTC_data/BTC_2019_2023_1d.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/BTC_data/BTC_1d_2024.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/ETH_data/ETH_2019_2023_1d.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/ETH_data/ETH_1d_2024.csv', parse_dates=['datetime'])
# data = pd.read_csv('../data/SOL_data/SOL_1d_2023.csv', parse_dates=['datetime'])




data=data.copy()
def chaikin_money_flow(data, period=14):
    """
    Calculate the Chaikin Money Flow (CMF) for a given dataset.

    Parameters:
    - data: DataFrame with columns ['high', 'low', 'close', 'volume']
    - period: Lookback period for CMF (default is 20)

    Returns:
    - CMF values as a pandas Series
    """
    # Calculate the Money Flow Multiplier (MFM)
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    
    # Calculate the Money Flow Volume (MFV)
    mfv = mfm * data['volume']
    
    # Calculate the Chaikin Money Flow (CMF) by summing MFV over the period and dividing by total volume
    cmf = mfv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    
    return cmf

data['CMF'] = chaikin_money_flow(data, period=14)

def calculate_ema(data, period=9):
    ema = data['close'].ewm(span=period, adjust=False).mean()
    return ema

def calculate_ema_crossover(data, short_period=9, long_period=21):
    short_ema = calculate_ema(data, short_period)
    long_ema = calculate_ema(data, long_period)
    return short_ema, long_ema
data['EMA9'], data['EMA21'] = calculate_ema_crossover(data)

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
def calculate_adx(data, period=14):
    """
    Calculate the Average Directional Index (ADX).

    Parameters:
        data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
        period (int): Period for ADX calculation. Default is 14.

    Returns:
        pd.DataFrame: DataFrame with the added 'ADX', '+DI', and '-DI' columns.
    """
    # Calculate True Range (TR)
    data['TR'] = np.maximum(data['high'] - data['low'], 
                            np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                       abs(data['low'] - data['close'].shift(1))))

    # Calculate +DM and -DM
    data['+DM'] = np.where((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']), 
                           np.maximum(data['high'] - data['high'].shift(1), 0), 0)
    data['-DM'] = np.where((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)), 
                           np.maximum(data['low'].shift(1) - data['low'], 0), 0)

    # Calculate smoothed values
    data['TR_smoothed'] = data['TR'].rolling(window=period).sum()
    data['+DM_smoothed'] = data['+DM'].rolling(window=period).sum()
    data['-DM_smoothed'] = data['-DM'].rolling(window=period).sum()

    # Calculate +DI and -DI
    data['+DI'] = 100 * (data['+DM_smoothed'] / data['TR_smoothed'])
    data['-DI'] = 100 * (data['-DM_smoothed'] / data['TR_smoothed'])

    # Calculate DX
    data['DX'] = 100 * (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']))

    # Calculate ADX
    data['ADX'] = data['DX'].rolling(window=period).mean()

    return data
def calculate_obv(df,period=10):
    # Parameters
    
    obv_ma_period = period  # Period for OBV moving average

    # Calculate OBV
    df['OBV'] = (np.where(df['close'] > df['close'].shift(1), df['volume'],
                            np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))).cumsum()
    
    # Calculate OBV Moving Average
    df['OBV_ma'] = df['OBV'].rolling(window=obv_ma_period).mean()
    return df

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

data[['HA_open', 'HA_high', 'HA_low', 'HA_close']]= calculate_heikin_ashi(data)





data= calculate_obv(data)
data= calculate_adx(data)
data= calculate_bollinger_bands(data)
data= calculate_supertrend(data)



# RSI
delta = data['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD
data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Hist'] = data['MACD'] - data['Signal_Line']

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

for i in range(10, len(data)):
    rsi = data.loc[i, 'RSI']
    macd_hist = data.loc[i, 'MACD_Hist']
    close_price = data.loc[i, 'close']
    prev_position = position[len(position)-1]
    atr = data.loc[i, 'ATR']
    ema_9= data.loc[i, 'EMA9']
    ema_21 = data.loc[i, 'EMA21']
    lb= data.loc[i, 'Lower_Band']
    up= data.loc[i, 'Upper_Band']
    cmf= data.loc[i, 'CMF']
    obv= data.loc[i, 'OBV']
    adx= data.loc[i, 'ADX']
    



    # Entry signals
    if prev_position == 0:  # No position
        # Long entry condition: RSI < 30 (oversold) and MACD > 0 (bullish trend)
        if data.loc[i,'HA_open']==data.loc[i,'HA_low'] or rsi<30 or (macd_hist>0 and cmf>0) :  # Long entry signal (momentum up)
            # data.loc[i+1, 'trade_type'] = 'Long'
            data.loc[i, 'signals'] = 1
            stop_loss = (close_price - atr * atr_multiplier)
            data.loc[i,'position']=100

        
        # Short entry condition: RSI > 70 (overbought) and MACD < 0 (bearish trend)
        elif data.loc[i,'HA_open']==data.loc[i,'HA_high'] or (macd_hist<0 and cmf<0):  # Short entry signal (momentum down)
            # data.loc[i+1, 'trade_type'] = 'Short'
            data.loc[i, 'signals'] = -1
            stop_loss = (close_price + atr * atr_multiplier)
            data.loc[i,'position']=100

    # Exit signals 
    elif prev_position == 1:  # Long position
        # Strong reversal signal to close long and enter short if MACD turns negative
        # if data.loc[i,'HA_open']==data.loc[i,'HA_high'] :
        #     # data.loc[i+1, 'trade_type'] = 'close Long & Short'
        #     data.loc[i, 'signals'] = -2
        #     stop_loss = 0
        # Exit long if RSI > 70 (overbought) or price hits stop-loss
        if  (macd_hist<0 and cmf<0) or close_price < stop_loss:  
            # data.loc[i+1, 'trade_type'] = 'Exit Long'
            data.loc[i, 'signals'] = -1
            stop_loss = 0

    elif prev_position == -1:  # Short position
        
        # Strong reversal signal to close short and enter long if MACD turns positive
        if data.loc[i,'HA_open']==data.loc[i,'HA_low']:
            # data.loc[i+1, 'trade_type'] = 'close Long & Short'
            data.loc[i, 'signals'] = 2
            stop_loss = 0
        # Exit short if RSI < 30 (oversold) or price hits stop-loss
        if close_price > stop_loss:
            # data.loc[i+1, 'trade_type'] = 'Exit Short'
            data.loc[i, 'signals'] = 1
            stop_loss = 0
    # Update position based on signals
    #print(i, data['signals'][data['signals']!=0])
    # data.loc[i+1, 'position'] = prev_position + data.loc[i, 'signals']
    
    position.append(prev_position+data.loc[i,'signals'])
data.loc[len(data)-1,'signals']=-position[len(data)-2]
position[len(data)-1]=0
print(len(position)," ",len(data))
print(position[-5:])
   

# Output to CSV
output_path = '../output.csv'
data[['datetime', 'open', 'high', 'low', 'close','signals', 'trade_type','position']].to_csv(output_path, index=False)

print(f"Output saved to {output_path}")