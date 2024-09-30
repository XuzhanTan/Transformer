import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(file_path):
    # Load data from CSV
    df = pd.read_csv(file_path)

    # Convert 'Datetime' column to pandas datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])    
    
    df_for_training = df[['Close']].astype(float)
    train_data, val_data = _splitData (df_for_training)
    """
    # Assuming 'Close' column contains the data needed for training
    df_for_training = df[['Close']].astype(float)

    # Define the percentage split for training and validation
    train_percentage = 0.7
    val_percentage = 0.3

    # Calculate the split index based on the lengths and percentages
    split_index = int(len(df_for_training) * train_percentage)

    # Split the data based on the index
    train_data = df_for_training.iloc[:split_index]
    val_data = df_for_training.iloc[split_index:]

    # Reset the index after splitting
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)

    # Print shapes of training and validation sets
    print("Training set shape:", train_data.shape)
    print("Validation set shape:", val_data.shape)
    """
    return train_data, val_data



def _splitData (df_for_training):
    
    #df_for_training = df[['Close']].astype(float)

    # Define the percentage split for training and validation
    train_percentage = 0.7
    val_percentage = 0.3

    # Calculate the split index based on the lengths and percentages
    split_index = int(len(df_for_training) * train_percentage)

    # Split the data based on the index
    train_data = df_for_training.iloc[:split_index]
    val_data = df_for_training.iloc[split_index:]

    # Reset the index after splitting
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)

    # Print shapes of training and validation sets
    print("Training set shape:", train_data.shape)
    print("Validation set shape:", val_data.shape)
    
    return train_data, val_data



def load_data_local (file_path):
    
    #file_path = "../data/strategy_data.csv"
    df = pd.read_csv (file_path, parse_dates=True)#, index_col = 0)

    df = df.iloc[:, :5]
    df.rename (columns={'date': 'Datetime',
                        'Strategy1' : 'Close1',
                        'Strategy2' : 'Close2',
                        'Strategy3' : 'Close3',},
               inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    #df.set_index ('Datetime', inplace = True)
    
    df_for_training = df[['Close1', 'Close2', 'Close3']].astype(float)
    train_data, val_data = _splitData (df_for_training)
    return train_data, val_data
    
def load_single_with_indicators(file_path, strategy): 
    #file_path = "../data/strategy_data.csv"
    df = pd.read_csv (file_path, parse_dates=True)

    df = df.iloc[:, [0, strategy]]
    df.rename (columns={'date': 'Datetime',
                        f'Strategy{strategy}' : 'Close1'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    #df.set_index ('Datetime', inplace = True)
    
    pdData = addIndicators (df[['Datetime', 'Close1']], 'Close1')
    df_for_training = pdData.iloc[:, 1:].astype(float)
    #df_for_training = df[['Close1', 'Close2', 'Close3']].astype(float)
    train_data, val_data = _splitData (df_for_training)
    
    return train_data, val_data

def load_data_with_indicators (file_path):
    
    #file_path = "../data/strategy_data.csv"
    df = pd.read_csv (file_path, parse_dates=True)

    df = df.iloc[:, :5]
    df.rename (columns={'date': 'Datetime',
                        'Strategy1' : 'Close1',
                        'Strategy2' : 'Close2',
                        'Strategy3' : 'Close3',}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    #df.set_index ('Datetime', inplace = True)
    
    pdData = addIndicators (df[['Datetime', 'Close1']], 'Close1')
    #df_for_training = df[['Close1', 'Close2', 'Close3']].astype(float)
    train_data, val_data = _splitData (pdData)
    
    return train_data, val_data




# Calculate Simple Moving Average (SMA)
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Calculate Exponential Moving Average (EMA)
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Calculate Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    sma = calculate_sma(data, window)
    rolling_std = data.rolling(window=window).std()
    upper_band = sma + num_std_dev * rolling_std
    lower_band = sma - num_std_dev * rolling_std
    return upper_band, lower_band

# Calculate Rate of Change (ROC)
def calculate_roc(data, n=12):
    return (data / data.shift(n) - 1) * 100

# Calculate Stochastic Oscillator
def calculate_stochastic_oscillator(data, high, low, window=14, k_window=3, d_window=3):
    high_max = data['High'].rolling(window=window).max()
    low_min = data['Low'].rolling(window=window).min()
    k = (data['Close'] - low_min) / (high_max - low_min) * 100
    d = k.rolling(window=d_window).mean()
    return k, d

# Calculate Parabolic SAR
def calculate_parabolic_sar(data, af_start=0.02, af_max=0.2):
    sar = [data['High'][0]]
    af = af_start
    ep = data['Low'][0]

    for i in range(1, len(data)):
        if sar[-1] < data['High'][i]:
            sar.append(sar[-1] + af * (ep - sar[-1]))
            af = min(af + af_start, af_max)
            ep = max(ep, data['Low'][i])
        else:
            sar.append(sar[-1] - af * (sar[-1] - ep))
            af = af_start
            ep = min(ep, data['High'][i])

    return pd.Series(sar, index=data.index)



def addIndicators (pdInput, sClose = 'Close1'):
    
    pdData = pdInput.copy()
    # Setting 'date' as the index for the reshaped data
    #pdData.set_index('Datetime', inplace=True)

    # Calculating the price change (dt1) and the previous price change (dt2)
    pdData['d1'] = pdData[sClose].diff()
    pdData['d2'] = pdData['d1'].shift(1)
    #pdData['d3'] = pdData['d2'].shift(1)
    #pdData['d4'] = pdData['d3'].shift(1)

    # Add technical indicators to feature space

    #pdData['SMA'] = calculate_sma(pdData['close'], window=20)
    pdData['EMA'] = calculate_ema(pdData[sClose], window=20)
    pdData['RSI'] = calculate_rsi(pdData[sClose])
    _, signal_line = calculate_macd(pdData[sClose])
    #reshaped_data['MACD'] = macd
    pdData['MACD_Signal'] = signal_line
    #upper_band, lower_band = calculate_bollinger_bands(pdData['close'])
    #pdData['Bollinger_Upper'] = upper_band
    #pdData['Bollinger_Lower'] = lower_band
    pdData['ROC'] = calculate_roc(pdData[sClose], 1)

    # Replacing NaN values with 0s
    pdData.dropna (inplace = True)
    #pdData.fillna (0.0, inplace=True)


    # Checking the first few rows of the updated dataframe
    print (pdData.head())
    
    return pdData


    
    
if __name__ == "__main__":

    import datetime as dt
    # # Usage
    # file_path = './dataset/AAPL_data_5min_1.csv'
    # train_data, val_data = load_data(file_path)
    
    
    file_path = "../data/strategy_data.csv"
    df = pd.read_csv (file_path, parse_dates=True)
    
    #train_data, val_data = load_data_local (file_path)
    
    #train_data, val_data = load_data_with_indicators (file_path)
    train_data, val_data = load_single_with_indicators (file_path)
    

    
    
    
    
    
    