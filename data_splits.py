import pandas as pd
import argparse


def main(ticker):
    data_path = f'{ticker}_raw_daily_data.csv'
    splits_path = f'{ticker}_stock_splits.csv'

    data_df = pd.read_csv(data_path)
    splits_df = pd.read_csv(splits_path)

    split_data = apply_splits(data_df, splits_df)

    split_data.to_csv('{ticker}_split_daily_data.csv')
    


def apply_splits(data, splits):
    for index, row in splits.itterrows():
        data.loc[data['time_stamp'] <= row['time_stamp'], [
            'open', 'high', 'low', 'close']] /= row['multiple']
        
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply stock splits for a ticker.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    args = parser.parse_args()
    main()
