from os.path import dirname, abspath, join
from pathlib import Path

root_path = dirname(dirname(abspath(__file__)))
tool_path = join(root_path, 'tools')
configs_path = join(root_path, 'configs')
results_path = join(root_path, 'results')
data_path = join(root_path, 'data')

################
# Data Related #
################
raw_data_path = Path(data_path, 'raw')
label_data_path = Path(data_path, 'label')
external_data_path = Path(data_path, 'external')
features_data_path = Path(data_path, 'features')
part_1_data_path = Path(data_path, 'part1')

###################
# Results Related #
###################

signals_results_path = Path(results_path, 'signals')
feature_importance_results_path = Path(results_path, 'feature_importance_results')
model_evaluation_results_path = Path(results_path, 'model_evaluation_results')
portfolio_value_results_path = Path(results_path, 'portfolio_value')
portfolio_analytics_results_path = Path(results_path, 'portfolio_analytics')
part_1_results_path = Path(results_path, 'part_1')
final_backtest_results_path = Path(results_path, 'final_backtest_results')
############
# Make Dir #
############

raw_data_path.mkdir(parents=True, exist_ok=True)
label_data_path.mkdir(parents=True, exist_ok=True)
external_data_path.mkdir(parents=True, exist_ok=True)
signals_results_path.mkdir(parents=True, exist_ok=True)
features_data_path.mkdir(parents=True, exist_ok=True)
portfolio_value_results_path.mkdir(parents=True, exist_ok=True)
feature_importance_results_path.mkdir(parents=True, exist_ok=True)
model_evaluation_results_path.mkdir(parents=True, exist_ok=True)
portfolio_analytics_results_path.mkdir(parents=True, exist_ok=True)
final_backtest_results_path.mkdir(parents=True, exist_ok=True)

part_1_data_path.mkdir(parents=True, exist_ok=True)
part_1_results_path.mkdir(parents=True, exist_ok=True)
