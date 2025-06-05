from os.path import dirname, abspath, join
from pathlib import Path

root_path = dirname(dirname(abspath(__file__)))
tool_path = join(root_path, 'tools')
configs_path = join(root_path, 'configs')
results_path = join(root_path, 'results')
data_path = join(root_path, 'data')

raw_data_path = Path(data_path, 'raw')
label_data_path = Path(data_path, 'label')
external_data_path = Path(data_path, 'external')
signals_data_path = Path(data_path, 'signals')
features_data_path = Path(data_path, 'features')

portfolio_value_results_path = Path(results_path, 'portfolio_value')
############
# Make Dir #
############
raw_data_path.mkdir(parents=True, exist_ok=True)
label_data_path.mkdir(parents=True, exist_ok=True)
external_data_path.mkdir(parents=True, exist_ok=True)
signals_data_path.mkdir(parents=True, exist_ok=True)
features_data_path.mkdir(parents=True, exist_ok=True)
portfolio_value_results_path.mkdir(parents=True, exist_ok=True)
