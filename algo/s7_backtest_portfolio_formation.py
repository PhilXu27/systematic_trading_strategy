import pandas as pd
from pathlib import Path
from utils.path_info import portfolio_value_results_path


def s7_backtest_portfolio_formation(
    prices, backtest_model_predictions, forward_looking_labels
):
    try:
        forward_looking_signals = forward_looking_labels[["bin"]].map(lambda x: 1.0 if x == 1.0 else 0.0)
    except AttributeError:
        forward_looking_signals = forward_looking_labels[["bin"]].applymap(lambda x: 1.0 if x == 1.0 else 0.0)
    portfolio_values = {}

    for model, model_predictions in backtest_model_predictions.items():
        print(f"Running Portfolio Formation for {model}")
        curr_port_values = generate_portfolio_value(prices, model_predictions["pred"].astype(int))
        if "backward_looking_benchmark" not in portfolio_values:
            portfolio_values["backward_looking_benchmark"] = generate_portfolio_value(
                prices, model_predictions["true"].astype(int)
            )
        if "buy_and_hold" not in portfolio_values:
            portfolio_values["buy_and_hold"] = generate_portfolio_value(
                prices, pd.Series(1.0, index=model_predictions["true"].index)
            )
        portfolio_values[model] = curr_port_values
    portfolio_values["forward_looking_benchmark"] = generate_portfolio_value(
        prices, forward_looking_signals["bin"].astype(int)
    )

    for port_name, port_value in portfolio_values.items():
        port_value.to_csv(Path(portfolio_value_results_path, f"{port_name}.csv"))
    return portfolio_values


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
