import pandas as pd
from utils.path_info import raw_data_path
from pathlib import Path
from os.path import exists

def s0_data_prepare():
    prices_file_path = Path(raw_data_path, "prices.csv")
    if exists(prices_file_path):
        prices = pd.read_csv(prices_file_path, index_col=0, parse_dates=True)
    else:
        url = 'https://drive.google.com/uc?id=1P_5ykYLd5521QUdCxC_cMytdJ3PqESTw'
        prices = pd.read_csv(url, index_col=0, parse_dates=True)
        prices.to_csv(prices_file_path)

    return prices


if __name__ == '__main__':
    s0_data_prepare()
