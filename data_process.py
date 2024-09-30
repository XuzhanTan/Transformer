import pandas as pd
import numpy as np
# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# import os
# path = os.getcwd()
# print(path)

def load_data(use_example = False, dir_path = None):
    """
    if use_example:
        TOP_BRL = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA",
        "BBAS3.SA", "RENT3.SA", "LREN3.SA", "PRIO3.SA",
        "WEGE3.SA", "ABEV3.SA"
        ]
        print(len(TOP_BRL))

        portfolio_raw_df = YahooDownloader (start_date = '2011-01-01',
                                            end_date = '2023-12-31',
                                            ticker_list = TOP_BRL).fetch_data()
    else:
    """
    # first read in the raw data
    # Reading the CSV file
    # sPath = 'c://data//stategy_data.csv'
    sPath = './strategy_data.csv'
    df = pd.read_csv (sPath, parse_dates=True, index_col=0)

    # Rename the column
    df.rename (columns={'Dates': 'date'}, inplace=True)

    # Reshaping the data to the desired format
    reshaped_data = df.stack().reset_index()

    #reshaped_data = df.stack()
    reshaped_data.columns = ['date', 'tic', 'close']

    # Setting 'date' as the index for the reshaped data
    reshaped_data.set_index('date', inplace=True)

    # Sorting the dataframe by 'ticker' and then by 'date'
    reshaped_data = reshaped_data.sort_values(by=['tic', 'date'])

    # Calculating the price change (dt1) and the previous price change (dt2)
    reshaped_data['d1'] = reshaped_data.groupby('tic')['close'].diff()
    reshaped_data['d2'] = reshaped_data.groupby('tic')['d1'].shift(1)
    reshaped_data['d3'] = reshaped_data.groupby('tic')['d2'].shift(1)
    reshaped_data['d4'] = reshaped_data.groupby('tic')['d3'].shift(1)

    # Replacing NaN values with 0s
    reshaped_data.fillna(0.01, inplace=True)

    # Checking the first few rows of the updated dataframe
    reshaped_data.head()

    # Now lets copy the reshaped data to the variable used in the model
    portfolio_raw_df = reshaped_data
        
    return portfolio_raw_df, df



if __name__ == "__main__":

    import datetime as dt

    pdStackedData, pdData = load_data()