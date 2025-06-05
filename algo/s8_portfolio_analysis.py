from algo.s8_helper import calculate_performance_metrics
from pathlib import Path
from utils.path_info import portfolio_analytics_results_path


def s8_portfolio_analysis(portfolio_value, backtest_save_prefix):
    portfolio_analytics_save_path = Path(portfolio_analytics_results_path, backtest_save_prefix)
    portfolio_analytics_save_path.mkdir(parents=True, exist_ok=True)

    pmetrics, reformatted_pmetrics = calculate_performance_metrics(
        portfolio_value, annualized_factor=24*365, is_reformat=True, is_mdd_detail=True
    )
    pmetrics.to_csv(Path(portfolio_analytics_save_path, 'performance_metrics.csv'))
    reformatted_pmetrics.to_csv(Path(portfolio_analytics_save_path, 'reformatted_performance_metrics.csv'))

    return
