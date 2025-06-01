from os.path import dirname, abspath, join
from pathlib import Path

root_path = dirname(dirname(abspath(__file__)))
tool_path = join(root_path, 'tools')
configs_path = join(root_path, 'configs')
results_path = join(root_path, 'results')
data_path = join(root_path, 'data')

raw_data_path = Path(join(data_path, 'raw'))
asset_pool_data_path = join(data_path, 'asset_pool')

raw_data_path.mkdir(parents=True, exist_ok=True)


