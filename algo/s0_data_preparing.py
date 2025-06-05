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
    adjusted_price_path = Path(external_data_path, "adjusted_price.csv")
    volume_path = Path(external_data_path, "volume.csv")
    vix_path = Path(external_data_path, "VIX Index.csv")

    if not exists(adjusted_price_path):
        url = "https://drive.google.com/uc?id=1KPmv-U3c6LfZ4RN5nmtGa2w2dAUKw-9q"
        adjusted_price = pd.read_csv(url, index_col=0, parse_dates=True, dayfirst=True)
        adjusted_price.to_csv(adjusted_price_path)

    if not exists(volume_path):
        url = "https://drive.google.com/uc?id=1KZknBvrldqVscZXqx4waqt29Zhi4TB0c"
        volume = pd.read_csv(url, index_col=0, parse_dates=True, dayfirst=True)
        volume.to_csv(volume_path)

    if not exists(vix_path):
        url = "https://drive.google.com/uc?id=1JDf-EihZ0raisD750uUIF27slYNHk45l"
        vix = pd.read_csv(url, index_col=0, parse_dates=True, dayfirst=True)
        vix.to_csv(vix_path)

    return prices


def load_external_data():
    adjusted_prices = pd.read_csv(Path(external_data_path, "adjusted_price.csv"), parse_dates=True, index_col=0)
    volume = pd.read_csv(Path(external_data_path, "volume.csv"), parse_dates=True, index_col=0)
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
