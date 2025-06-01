import pandas as pd
from utils.path_info import data_path
from pathlib import Path

def s0_data_prepare():
    prices = pd.read_csv(Path(data_path, "prices.csv"), index_col=0, parse_dates=True)
    return prices
