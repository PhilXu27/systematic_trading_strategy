from algo.s8_helper import calculate_performance_metrics
from pathlib import Path
from utils.path_info import portfolio_analytics_results_path


def s8_portfolio_analysis(portfolio_value):
    pmetrics, reformatted_pmetrics = calculate_performance_metrics(
        portfolio_value, annualized_factor=24*365, is_reformat=True, is_mdd_detail=True
    )
    pmetrics.to_csv(Path(portfolio_analytics_results_path, 'performance_metrics.csv'))
    reformatted_pmetrics.to_csv(Path(portfolio_analytics_results_path, 'reformatted_performance_metrics.csv'))

    return
