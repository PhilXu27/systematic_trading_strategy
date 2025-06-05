import pandas as pd
import numpy as np
PRINT_PREFIX = None

def calculate_performance_metrics(
        input_data: pd.DataFrame,
        start: str = "1900-01-01", end: str = "2100-01-01", frequency: str = "D",
        annualized_factor: str or int = "default", exchange: str = None,
        is_reformat: bool = True, is_mdd_detail=True,
        **kwargs
) -> pd.DataFrame or (pd.DataFrame, pd.DataFrame):
    if isinstance(input_data, pd.Series):
        input_data = pd.DataFrame(input_data)
    input_data = input_data.loc[start: end]

    if isinstance(annualized_factor, int):
        pmetrics = performance_metrics_helper(input_data, annualized_factor, is_mdd_detail, **kwargs)
    else:
        raise ValueError(f"{PRINT_PREFIX}: Unsupported annualized_factor: {annualized_factor}")

    if is_reformat:
        reformatted_pmetrics = performance_metrics_reformat_helper(pmetrics, **kwargs)
        return pmetrics, reformatted_pmetrics
    else:
        return pmetrics


def performance_metrics_helper(input_data, base_per_year, is_mdd_detail, **kwargs):
    rf = kwargs.get("rf", 0.0)
    return_df = input_data.pct_change()
    p_metrics = pd.DataFrame()
    p_metrics["Total Return"] = input_data.iloc[-1] / input_data.iloc[0] - 1
    p_metrics["Annualized Return"] = np.power(p_metrics["Total Return"] + 1, (base_per_year / input_data.shape[0])) - 1
    p_metrics["Annualized Volatility"] = return_df.std(axis=0) * np.power(base_per_year, 0.5)
    p_metrics["Downside Volatility"] = return_df[return_df < -10e-12].std(axis=0) * np.power(base_per_year, 0.5)
    p_metrics["Sharpe Ratio"] = (p_metrics["Annualized Return"] - rf) / p_metrics["Annualized Volatility"]
    p_metrics["Sortino Ratio"] = (p_metrics["Annualized Return"] - rf) / p_metrics["Downside Volatility"]
    p_metrics["Max Drawdown"] = (input_data.div(input_data.cummax()) - 1.).min()
    p_metrics["Calmar Ratio"] = - (p_metrics["Annualized Return"] - rf) / p_metrics["Max Drawdown"]
    if is_mdd_detail:
        p_metrics["Max Drawdown Day"], p_metrics["Max Drawdown Start"], p_metrics["Max Drawdown Recovery"] = \
            max_drawdown_cal(input_data)[0], max_drawdown_cal(input_data)[1], max_drawdown_cal(input_data)[2]
    p_metrics["Skewness"] = return_df.skew()
    p_metrics["Kurtosis"] = return_df.kurt()

    return p_metrics


def performance_metrics_reformat_helper(p_metrics, **kwargs):
    n_digit = kwargs.get("n_digit", 3)
    n_percentage_digit = kwargs.get("n_percentage_digit", None)
    n_value_digit = kwargs.get("n_value_digit", None)
    if n_percentage_digit is None:
        n_percentage_digit = n_digit
    if n_value_digit is None:
        n_value_digit = n_digit

    date_format = kwargs.get("date_format", "%Y-%m-%d")

    reformat_metrics = p_metrics.copy()
    reformat_metrics["Total Return"] = p_metrics["Total Return"].apply(
        lambda x: percentage_reformat_n_dps(x, n_percentage_digit)
    )
    reformat_metrics["Annualized Return"] = p_metrics["Annualized Return"].apply(
        lambda x: percentage_reformat_n_dps(x, n_percentage_digit)
    )
    reformat_metrics["Annualized Volatility"] = p_metrics["Annualized Volatility"].apply(
        lambda x: percentage_reformat_n_dps(x, n_percentage_digit)
    )
    reformat_metrics["Downside Volatility"] = p_metrics["Downside Volatility"].apply(
        lambda x: percentage_reformat_n_dps(x, n_percentage_digit)
    )
    reformat_metrics["Sharpe Ratio"] = p_metrics["Sharpe Ratio"].apply(
        lambda x: absolute_val_reformat_n_dps(x, n_value_digit)
    )
    reformat_metrics["Sortino Ratio"] = p_metrics["Sortino Ratio"].apply(
        lambda x: absolute_val_reformat_n_dps(x, n_value_digit)
    )
    reformat_metrics["Max Drawdown"] = p_metrics["Max Drawdown"].apply(
        lambda x: percentage_reformat_n_dps(x, n_percentage_digit)
    )
    reformat_metrics["Calmar Ratio"] = p_metrics["Calmar Ratio"].apply(
        lambda x: absolute_val_reformat_n_dps(x, n_value_digit)
    )
    reformat_metrics["Max Drawdown Day"] = p_metrics["Max Drawdown Day"].apply(
        lambda x: datetime_reformat(x, date_format)
    )
    reformat_metrics["Max Drawdown Start"] = p_metrics["Max Drawdown Start"].apply(
        lambda x: datetime_reformat(x, date_format)
    )
    reformat_metrics["Max Drawdown Recovery"] = p_metrics["Max Drawdown Recovery"].apply(
        lambda x: datetime_reformat(x, date_format)
    )
    return reformat_metrics

def max_drawdown_cal(input_origin_df):
    """
    Maximum drawdown calculator

    Args:
        input_origin_df: pv series.

    Returns:
        day: Exact day that maximum drawdown actually happened.
        start: The highest point before maximum drawdown (when max-drawdown starts)
        end: When the pv recovers.
    """
    md_cal = (input_origin_df.div(input_origin_df.cummax()) - 1.)
    md_cal_copy = md_cal.copy()
    md_cal_copy = md_cal_copy.iloc[::-1]

    day = []
    start = []
    recover = []
    for col in md_cal:
        md_day = md_cal[col].idxmin(axis=0)
        md_start = md_cal_copy[col][md_day:].idxmax(axis=0)
        md_recover = md_cal[col][md_day:].idxmax(axis=0)
        day.append(md_day)
        start.append(md_start)
        recover.append(md_recover)
    return day, start, recover


def percentage_reformat_2dps(x):
    return format(x * 100, '.2f') + "%"


def percentage_reformat_n_dps(x, n):
    return format(x * 100, f'.{n}f') + "%"


def absolute_val_reformat_n_dps(x, n):
    return format(x, f'.{n}f')


def absolute_val_reformat_2dps(x):
    return format(x, '.2f')


def datetime_reformat_dashed(x):
    return x.strftime("%Y-%m-%d")


def datetime_reformat(x, format_str):
    return x.strftime(format_str)
