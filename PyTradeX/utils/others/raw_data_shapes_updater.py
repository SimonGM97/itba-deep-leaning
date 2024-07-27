from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.data_processing.data_extractor import DataExtractor
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
from PyTradeX.utils.general.logging_helper import get_logger
from tqdm import tqdm
from pprint import pformat
from typing import Dict


# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.log_params.get('level'),
    txt_fmt=Params.log_params.get('txt_fmt'),
    json_fmt=Params.log_params.get('json_fmt'),
    filter_lvls=Params.log_params.get('filter_lvls'),
    log_file=Params.log_params.get('log_file'),
    backup_count=Params.log_params.get('backup_count')
)


# source .pytradex_venv/bin/activate
# .pytradex_venv/bin/python PyTradeX/utils/others/raw_data_shapes_updater.py

# Instanciate BinanceClient
client = BinanceClient()

# Define raw_data_shapes_path
raw_data_shapes_path = f"{Params.bucket}/utils/raw_data_shapes/raw_data_shapes.json"

# Load raw_data_shapes
try:
    raw_data_shapes: Dict[str, tuple] = load_from_s3(path=raw_data_shapes_path)
except Exception as e:
    LOGGER.warning(
        'Unable to load raw_data_shapes.\n'
        'Exception: %s.\n', e
    )
    raw_data_shapes: Dict[str, tuple] = {
        '30min': {},
        '60min': {},
        '1d': {}
    }

# Extract intervals
intervals = Params.general_params.get('intervals')

# Populate raw_data_shapes
LOGGER.debug('Updating %s raw_data_shapes:', intervals)
for coin_name in tqdm(Params.fixed_params.get('full_coin_list')):
    # Instanciate DataExtractor
    DE = DataExtractor(
        coin_name=coin_name,
        intervals=intervals,
        client=client,
        overwrite=True,
        **Params.data_params.copy()
    )
    
    # Populate raw_data_shapes
    raw_data_shapes[intervals][coin_name] = DE.raw_data.shape

LOGGER.info('raw_data_shapes:\n%s\n', pformat(raw_data_shapes))

# Save raw_data_shapes
write_to_s3(
    asset=raw_data_shapes,
    path=raw_data_shapes_path
)