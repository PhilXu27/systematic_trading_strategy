from algo.s8_helper import calculate_performance_metrics, calculate_average_holding_period
from algo.s2_creating_labels import s2_load_labels
from algo.s0_data_preparing import s0_data_prepare
from pathlib import Path
from utils.path_info import final_backtest_results_path, results_path
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


sns.set(style="whitegrid", rc={"figure.figsize": (12, 6)})  # Set figure size to 12x6 inches
plt.rcParams.update({
    "font.size": 12,        # General font size
    "axes.titlesize": 14,   # Title font size
    "axes.labelsize": 12,   # Axis label font size
    "xtick.labelsize": 10,  # X-axis tick label size
    "ytick.labelsize": 10,  # Y-axis tick label size
    "legend.fontsize": 10,  # Legend font size
    "legend.title_fontsize": 12  # Legend title font size
})


def load_results(sub_result):
    clean_result_path = Path(final_backtest_results_path, "clean")
    clean_result_path.mkdir(parents=True, exist_ok=True)
    models = ["xgb_boost", "random_forest", "gradient_boost", "lightgbm"]
    info_dir = Path(final_backtest_results_path, "portfolio_value", sub_result)
    average_holding_period = {}

    for model in models:
        curr_info = pd.read_csv(Path(info_dir, f"{model}_info.csv"), index_col=0, parse_dates=True)
        average_holding_period[model] = calculate_average_holding_period(curr_info)

    portfolio_value = pd.read_csv(Path(final_backtest_results_path, "portfolio_value", sub_result, "portfolio_value.csv"), index_col=0, parse_dates=True)
    _, pmetrics = calculate_performance_metrics(
        portfolio_value, annualized_factor=24 * 365, is_reformat=True, is_mdd_detail=True
    )
    pd.Series(average_holding_period).to_csv(Path(clean_result_path, f"average_holding_period_{sub_result}.csv"))
    pmetrics.to_csv(Path(clean_result_path, f"pmetrics_{sub_result}.csv"))
    return portfolio_value

def performance_analysis():
    return




def main():
    result_info = [
        # "rebalance_120h_retrain_720h_mode_parallel_rolling_window",
        # "rebalance_120h_retrain_720h_mode_parallel_expanding_window",
        # "rebalance_120h_retrain_240h_mode_parallel_rolling_window",
        # "rebalance_120h_retrain_240h_mode_parallel_expanding_window",
        # "rebalance_60h_retrain_720h_mode_parallel_rolling_window",
        # "rebalance_60h_retrain_720h_mode_parallel_expanding_window",
        "rebalance_120h_retrain_720h_mode_parallel_rolling_window_hyperparams_search"
        # "rebalance_120h_retrain_720h_mode_parallel_expanding_window_middle_hyper_space"
        # "rebalance_120h_retrain_720h_mode_parallel_expanding_window_huge_hyper_space"
    ]
    reb_120_ret_720_info = {
        "rebalance_120h_retrain_720h_mode_parallel_rolling_window": "Rolling Window",
        "rebalance_120h_retrain_720h_mode_parallel_expanding_window": "Expanding Window"
    }
    reb_120_ret_240_info = {
        "rebalance_120h_retrain_240h_mode_parallel_rolling_window": "Rolling Window",
        "rebalance_120h_retrain_240h_mode_parallel_expanding_window": "Expanding Window"
    }
    reb_60_ret_720_plot_info = {
        "rebalance_60h_retrain_720h_mode_parallel_rolling_window": "Rolling Window",
        "rebalance_60h_retrain_720h_mode_parallel_expanding_window": "Expanding Window"
    }

    for sub_result in result_info:
        pv = load_results(sub_result)
        if sub_result in reb_120_ret_720_info:
            plot_portfolio_value(pv, reb_120_ret_720_info[sub_result], "reb_120_ret_720")
        if sub_result in reb_120_ret_240_info:
            plot_portfolio_value(pv, reb_120_ret_240_info[sub_result], "reb_120_ret_240")
        if sub_result in reb_60_ret_720_plot_info:
            plot_portfolio_value(pv, reb_60_ret_720_plot_info[sub_result], "reb_60_ret_720")
    return


def plot_portfolio_value(df, title_keyword, which_file):
    df_melted = df.melt(ignore_index=False, var_name='strategy', value_name='portfolio_value')

    # Reset index to make 'date' a column for plotting
    df_melted = df_melted.reset_index()

    # Create the line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melted, x='date', y='portfolio_value', hue='strategy', linewidth=2)

    # Customize the plot
    plt.title(f'{title_keyword} Portfolio Value Over Time by Strategy', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Display the plot
    clean_result_path = Path(final_backtest_results_path, "clean", which_file)
    clean_result_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(clean_result_path, f"{title_keyword}_portfolio_value.png"))
    return


def plot_labels():
    _, labels = s2_load_labels("labels_6_12_18_24_48")
    prices = s0_data_prepare()
    general_start = "2018-01-01"
    general_end = "2021-12-31"
    close = prices["close"]
    close = close.loc[general_start:general_end]
    close_events = close.loc[labels.index]
    fig = plt.figure(figsize=(10, 5))  # Increased width
    scatter = plt.scatter(x=close_events.index, y=close_events.values, c=labels["tVal"], s=20)

    # Adding a colorbar
    plt.colorbar(scatter, label='t_value scale')

    # Use a more sparse date locator and formatter
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    # Rotate date labels for better visibility
    plt.xticks(rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Forward-Looking Trend Scanning Labels with Window Size [6, 12, 18, 24, 48]')
    plt.savefig(Path(results_path, "labels.png"))
    return


if __name__ == '__main__':
    # main()
    plot_labels()
