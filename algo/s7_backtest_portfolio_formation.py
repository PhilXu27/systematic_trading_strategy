from pathlib import Path

import pandas as pd

from utils.path_info import portfolio_value_results_path


def s7_backtest_portfolio_formation(
        prices, backtest_model_predictions, forward_looking_labels, backtest_save_prefix
):
    portfolio_value_save_path = Path(portfolio_value_results_path, backtest_save_prefix)
    portfolio_value_save_path.mkdir(parents=True, exist_ok=True)

    try:
        forward_looking_signals = forward_looking_labels[["bin"]].map(lambda x: 1.0 if x == 1.0 else 0.0)
    except AttributeError:
        forward_looking_signals = forward_looking_labels[["bin"]].applymap(lambda x: 1.0 if x == 1.0 else 0.0)
    portfolio_info = {}
    portfolio_value = pd.DataFrame()
    for model, model_predictions in backtest_model_predictions.items():
        print(f"Running Portfolio Formation for {model}")
        curr_port_info = generate_portfolio_value(prices, model_predictions["pred"].astype(int))
        curr_pv = curr_port_info[["portfolio_value"]]
        curr_pv.columns = [model]
        portfolio_value = pd.concat([portfolio_value, curr_pv], axis=1)

        if "buy_and_hold" not in portfolio_info:
            bah_portfolio = generate_portfolio_value(
                prices, pd.Series(1.0, index=model_predictions["true"].index)
            )
            portfolio_info["buy_and_hold"] = bah_portfolio
            pv = bah_portfolio[["portfolio_value"]]
            pv.columns = ["buy_and_hold"]
            portfolio_value = pd.concat([portfolio_value, pv], axis=1)
        portfolio_info[model] = curr_port_info

    for port_name, port_value in portfolio_info.items():
        port_value.to_csv(Path(portfolio_value_save_path, f"{port_name}_info.csv"))
    portfolio_value.to_csv(Path(portfolio_value_save_path, 'portfolio_value.csv'))
    return portfolio_info, portfolio_value


def generate_portfolio_value(prices, signals):
    valid_trading_period = signals.index.tolist()
    prices = prices.loc[valid_trading_period]

    portfolio = pd.DataFrame(index=prices.index, columns=['portfolio_value', 'position'])
    portfolio['portfolio_value'] = 1.0
    portfolio['position'] = 0

    for t in range(len(signals) - 1):
        current_time = signals.index[t]
        next_time = signals.index[t + 1]

        current_signal = signals.loc[current_time]
        current_position = portfolio.loc[current_time, 'position']
        current_portfolio_value = portfolio.loc[current_time, 'portfolio_value']

        # Case 1: Signal = 1 (Hold Bitcoin for next period)
        if current_signal == 1:
            # If no position (0), buy at open of t+1
            if current_position == 0:
                portfolio.loc[next_time, 'position'] = 1
                portfolio.loc[next_time, 'portfolio_value'] = current_portfolio_value
            # If already holding (1), calculate return from t close to t+1 close
            elif current_position == 1:
                portfolio.loc[next_time, 'position'] = 1
                return_t = prices.loc[next_time, 'close'] / prices.loc[current_time, 'close']
                portfolio.loc[next_time, 'portfolio_value'] = current_portfolio_value * return_t
        elif current_signal == 0:
            # If holding Bitcoin (1), sell at open of t+1
            if current_position == 1:
                portfolio.loc[next_time, 'position'] = 0
                return_t = prices.loc[next_time, 'open'] / prices.loc[current_time, 'close']
                portfolio.loc[next_time, 'portfolio_value'] = current_portfolio_value * return_t
            # If no position (0), do nothing
            elif current_position == 0:
                portfolio.loc[next_time, 'position'] = 0
                portfolio.loc[next_time, 'portfolio_value'] = current_portfolio_value
        else:
            raise ValueError
    # portfolio['portfolio_value'] = portfolio['portfolio_value'].ffill()
    # portfolio['position'] = portfolio['position'].ffill()
    return portfolio
