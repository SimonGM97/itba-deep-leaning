from PyTradeX.utils.others.s3_helper import load_from_s3
# from config.setup_keys import setup_keys

import pandas as pd
import yaml
from pathlib import Path
from git.repo.base import Repo
# from git.exc import InvalidGitRepositoryError
import multiprocessing
import subprocess
import os


pd.options.display.max_rows = 500
pd.set_option('display.max_columns', None)


def find_base_repo_root() -> Path:
    base_path = os.path.dirname(os.path.abspath(__file__))
    if 'PyTradeX' in base_path:
        base_path = base_path[:base_path.find('PyTradeX')]
    
    try:
        base_path = Path(Repo(base_path, search_parent_directories=True).working_tree_dir or base_path)
    except Exception as e:
        # print(f'Unable to load base_path.\n'
        #       f'Exception: {e}\n\n')
        # except InvalidGitRepositoryError:
        base_path = Path(base_path)

    return base_path / "PyTradeX"


class Params:
    initialized: bool = False

    # Environment Parameters
    bucket: str
    compute_env: str
    storage_env: str
    cwd: Path

    # General Parameters
    general_params: dict

    # Logging Parameters
    log_params: dict

    # Data Parameters
    data_params: dict
    other_coins_json: dict
    other_stocks_json: dict
    lc_ids: dict

    # ML Parameters
    ml_params: dict

    # Trading Parameters
    trading_params: dict

    # Fixed Parameters
    fixed_params: dict

    # Test Parameters
    test_params: dict

    # Step Functions
    model_building: dict
    trading_round: dict
    model_updating: dict
    repair: dict

    # Default Script Values
    default: dict

    # Paralelizing Parameters
    cpus: int
    gpus: int

    @classmethod
    def load(
        cls, 
        force_re_initialize: bool = False,
        forced_intervals: str = None
    ) -> None:
        if cls.initialized and not force_re_initialize:
            return
        
        # Load config file
        with open(os.path.join("config", "config.yaml")) as file:
            config: dict = yaml.load(file, Loader=yaml.FullLoader)

        """
        Environment Parameters
        """
        # Use so save aws/databricks parameters
        prod_env = config['env_params']['prod_env']
        if prod_env:
            cls.bucket: str = 'pytradex-prod'            
        else:
            cls.bucket: str = 'pytradex-dev'

        # Define compute_env & storage_env
        cls.compute_env: str = config['env_params']['compute_env']
        cls.storage_env: str = config['env_params']['storage_env']
        cls.cwd = find_base_repo_root()

        """
        Keys
        """
        # setup_keys()
        
        """
        General Parameters
        """
        # Define general_params
        cls.general_params: dict = config['general_params']

        if forced_intervals is not None:
            cls.general_params['intervals'] = forced_intervals

        # Extract intervals
        intervals: str = cls.general_params['intervals']

        """
        Log Parameters
        """
        cls.log_params: dict = config['log_params']

        """
        Data Parameters
        """
        # Define data_params
        cls.data_params: dict = config['data_params']

        # Re-define "periods"
        cls.data_params['periods'] = cls.data_params['periods'][intervals]

        # Re-define "save_distance"
        cls.data_params['save_distance'] = cls.data_params['save_distance'][intervals]

        # Re-define "lag_periods"
        cls.data_params['lag_periods'] = cls.data_params['lag_periods'][intervals]

        # Re-define "rolling_windows"
        cls.data_params['rolling_windows'] = cls.data_params['rolling_windows'][intervals]

        # Re-define mbp & yfinance_params
        cls.data_params['mbp'] = cls.data_params['mbp'][intervals]
        cls.data_params['yfinance_params'] = cls.data_params['yfinance_params'][intervals]

        # Define other_coins_json
        cls.other_coins_json: dict = load_from_s3(
            path=f"{cls.bucket}/utils/correlations/crypto_correlations.json"
        )

        # Define other_stocks_json
        cls.other_stocks_json: dict = load_from_s3(
            path=f"{cls.bucket}/utils/correlations/stock_correlations.json"
        )

        # Define lc_ids
        cls.lc_ids: dict = load_from_s3(
            path=f"{cls.bucket}/utils/lc_ids/lc_ids.json"
        )

        # Define raw_data_shapes
        cls.raw_data_shapes: dict = load_from_s3(
            path=f"{cls.bucket}/utils/raw_data_shapes/raw_data_shapes.json"
        )

        """
        ML Parameters
        """
        # Define ml_params
        cls.ml_params: dict = config['ml_params']

        """
        Trading Parameters
        """
        # Define trading_params
        cls.trading_params: dict = config['trading_params']

        # Define trading_fees_dict
        """
        Binance Trading fees:
            - USDT:
                - Maker: 0.0200%
                - Taker: 0.0500%
            - USDT (BNB 10% off):
                - Maker: 0.0180%
                - Taker: 0.0450%
            - USDC:
                - Maker: 0.0180%
                - Taker: 0.0450%
            - USDC (BNB 10% off):
                - Maker: 0.0162%
                - Taker: 0.0405%
        """
        complete_trading_fees: dict = {
            'binance': {
                'USDT': {
                    'maker': -0.0180/100,
                    'taker': -0.0450/100
                },
                'USDC': {
                    'maker': -0.0162/100,
                    'taker': -0.0405/100
                }
            }
            
        }
        exchange = cls.general_params['exchange']
        
        cls.trading_params['trading_fees_dict'] = complete_trading_fees[exchange] # [stable_coin][order_type]

        """
        Fixed Parameters
        """
        # Define fixed_params
        cls.fixed_params: dict = config['fixed_params']

        """
        Test Parameters
        """
        # Define test_params
        cls.test_params: dict = config['test_params']
        
        """
        Step Functions
        """
        # Define model_building
        cls.model_building: dict = config['model_building']

        # Re-define "reduce_comb_datasets"
        new_reduce_comb_datasets = cls.model_building['data_processing']['selected_features_update_params']['reduce_comb_datasets'][intervals]
        cls.model_building['data_processing']['selected_features_update_params']['reduce_comb_datasets'] = new_reduce_comb_datasets

        # Re-define "reduced_tuning_periods"
        new_reduced_tuning_periods = cls.model_building['modeling']['tuning_params']['reduced_tuning_periods'][intervals]
        cls.model_building['modeling']['tuning_params']['reduced_tuning_periods'] = new_reduced_tuning_periods

        # Define trading_round
        cls.trading_round: dict = config['trading_round']

        # Define model_updating
        cls.model_updating: dict = config['model_updating']

        """
        Default Script Variables
        """
        # Define default
        cls.default: dict = config['default']

        # Re-define "reduce_comb_datasets"
        new_reduce_comb_datasets = cls.default['data_processing']['selected_features_update_params']['reduce_comb_datasets'][intervals]
        cls.default['data_processing']['selected_features_update_params']['reduce_comb_datasets'] = new_reduce_comb_datasets

        # Re-define "reduced_tuning_periods"
        new_reduced_tuning_periods = cls.default['modeling']['tuning_params']['reduced_tuning_periods'][intervals]
        cls.default['modeling']['tuning_params']['reduced_tuning_periods'] = new_reduced_tuning_periods
        
        """
        Paralelizing
        """
        def get_gpu_count():
            cmd = "system_profiler SPDisplaysDataType | grep Chipset"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            return len(output.split("\n"))

        cls.gpus = 1 # get_gpu_count()
        cls.cpus = multiprocessing.cpu_count()
        
        # if cls.cpus > 8:
        #     cls.cpus -= 1

        # print(f'Params.cpus: {cls.cpus}\n\n')

        cls.initialized = True


if not Params.initialized:
    # print('Initializing Params.\n')
    Params.load()