import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

def create_indicators():

    df = pd.read_csv('./data/AAPL_split_daily_data.csv')

    df.set_index(pd.DatetimeIndex(df["time_stamp"]), inplace=True)

    df = df.ta.reverse

    # Make sure df is in correct order for pandas_ta to work
    print(df.ta.datetime_ordered)

    # Creates all Technical Indicators
    df.ta.strategy()

    # View new TI info
    df.head()
    df.info()

    # Save to csv without duplicating the time_stamp feature
    df.to_csv('./data/AAPL_techincal_indicators.csv', index=False)

if __name__ == '__main__':
    create_indicators()




