import pandas as pd
from utils.path_info import raw_data_path, external_data_path
from pathlib import Path
from os.path import exists

def s0_data_prepare():
    ##############
    # BTC Prices #
    ##############
    prices_file_path = Path(raw_data_path, "prices.csv")
    if exists(prices_file_path):
        prices = pd.read_csv(prices_file_path, index_col=0, parse_dates=True)
    else:
        url = 'https://drive.google.com/uc?id=1P_5ykYLd5521QUdCxC_cMytdJ3PqESTw'
        prices = pd.read_csv(url, index_col=0, parse_dates=True)
        prices.to_csv(prices_file_path)
    #################################
    # Feature Related External Data #
    #################################

    return prices


def load_external_data():
    adjusted_prices = pd.read_csv(Path(external_data_path, "adjusted_price.csv"), parse_dates=True, dayfirst=True, index_col=0)
    volume = pd.read_csv(Path(external_data_path, "volume.csv"), parse_dates=True, dayfirst=True, index_col=0)
    volume = volume.reindex(adjusted_prices.index)
    prices_ffill_cols = [
        "BTC1 Curncy", "BTC2 Curncy", "BTC3 Curncy", "BTC4 Curncy", "DXY Index",
        "GB3 Govt", "GB6 Govt", "GB12 Govt", "GT2 Govt", "GT5 Govt", "GT10 Govt",
        "XAU Curncy", "CO1 Comdty"
    ]
    volume_ffill_cols = ["BTC1 Curncy", "BTC2 Curncy", "CO1 Comdty"]
    adjusted_prices[prices_ffill_cols] = adjusted_prices[prices_ffill_cols].ffill(limit=2)
    volume[volume_ffill_cols] = volume[volume_ffill_cols].ffill(limit=2)
    return



if __name__ == '__main__':
    s0_data_prepare()
