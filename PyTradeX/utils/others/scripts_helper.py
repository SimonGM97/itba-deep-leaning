#!~/pytradex_venv/bin/python
from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.utils.general.logging_helper import logging
import inspect
from typing import Any
from pprint import pformat


def force_intervals(
    logger: logging.Logger,
    forced_intervals: str = None
) -> None:
    # Force intervals
    if forced_intervals is not None:
        logger.warning('Re-initializing Params attributes with %s forced intervals.', forced_intervals)

        # Modify parameters
        Params.load(
            force_re_initialize=True,
            forced_intervals=forced_intervals
        )


def reset_intervals(
    logger: logging.Logger,
    forced_intervals: str = None
) -> None:
    # Re-set intervals
    if forced_intervals is not None:
        if forced_intervals == '30min':
            reset_intervals = '60min'
        elif forced_intervals == '60min':
            reset_intervals = '30min'
        else:
            raise Exception(f'Invalid "forced_intervals" parameter: {forced_intervals}.\n')
        
        logger.warning('Re-initializing Params attributes with %s forced intervals.', reset_intervals)

        Params.load(
            force_re_initialize=True,
            forced_intervals=reset_intervals
        )


def retrieve_name(var: Any) -> str:
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def retrieve_params(workflow: str) -> dict:
    # Find file that called the function
    frame = inspect.currentframe()
    file_name = frame.f_back.f_code.co_filename.split('/')[-1]
    
    if file_name == 'trading.py':
        # Retrieve script variables
        workflow_vars: dict = getattr(Params, workflow)
        trading_vars: dict = workflow_vars.get('trading')

        # Extract Params attrs
        general_params: dict = getattr(Params, 'general_params')
        data_params: dict = getattr(Params, 'data_params')
        ml_params: dict = getattr(Params, 'ml_params')
        trading_params: dict = getattr(Params, 'trading_params')

        # Extract parameters
        intervals: str = general_params.get('intervals')
        skip_wait: bool = trading_vars.get('skip_wait')
        debug: bool = trading_vars.get('debug')
        
        return {
            'intervals': intervals,
            'skip_wait': skip_wait,
            'debug': debug,

            'general_params': general_params,
            'data_params': data_params,
            'ml_params': ml_params,
            'trading_params': trading_params
        }

    elif file_name == 'data_processing.py':
        # Instanciate BinanceClient
        client = BinanceClient()

        # Retrieve script variables
        workflow_vars: dict = getattr(Params, workflow)
        data_processing_vars: dict = workflow_vars.get('data_processing')

        # Extract Params attrs
        general_params: dict = getattr(Params, 'general_params')
        data_params: dict = getattr(Params, 'data_params')
        fixed_params: dict = getattr(Params, 'fixed_params')

        # Extract parameters
        intervals: str = general_params.get('intervals')
        if (
            'repair_coins' in data_processing_vars.keys() 
            and data_processing_vars['repair_coins'] is not None
        ):
            full_coin_list: list = data_processing_vars.get('repair_coins')
        else:
            full_coin_list: list = fixed_params.get('full_coin_list')

        update_client: bool = data_processing_vars.get('update_client')
        update_correlations: bool = data_processing_vars.get('update_correlations')
        update_lc_ids: bool = data_processing_vars.get('update_lc_ids')
        overwrite: bool = data_processing_vars.get('overwrite')
        
        collective_data_update_params: dict = data_processing_vars.get('collective_data_update_params')
        data_extractor_update_params: dict = data_processing_vars.get('data_extractor_update_params')
        data_cleaner_update_params: dict = data_processing_vars.get('data_cleaner_update_params')
        data_shifter_update_params: dict = data_processing_vars.get('data_shifter_update_params')
        data_refiner_update_params: dict = data_processing_vars.get('data_refiner_update_params')
        selected_features_update_params: dict = data_processing_vars.get('selected_features_update_params')
        data_transformer_update_params: dict = data_processing_vars.get('data_transformer_update_params')

        debug: bool = data_processing_vars.get('debug')

        return {
            'client': client,

            'intervals': intervals,
            'full_coin_list': full_coin_list,
            'data_params': data_params,

            'update_client': update_client,
            'update_correlations': update_correlations,
            'update_lc_ids': update_lc_ids,
            'overwrite': overwrite,

            'collective_data_update_params': collective_data_update_params,
            'data_extractor_update_params': data_extractor_update_params,
            'data_cleaner_update_params': data_cleaner_update_params,
            'data_shifter_update_params': data_shifter_update_params,
            'data_refiner_update_params': data_refiner_update_params,
            'selected_features_update_params': selected_features_update_params,
            'data_transformer_update_params': data_transformer_update_params,

            'debug': debug
        }
    
    elif file_name == 'modeling.py':
        # Retrieve script variables
        workflow_vars: dict = getattr(Params, workflow)
        modeling_vars: dict = workflow_vars.get('modeling')

        # Extract Params attrs
        general_params: dict = getattr(Params, 'general_params')
        data_params: dict = getattr(Params, 'data_params')
        ml_params: dict = getattr(Params, 'ml_params')
        trading_params: dict = getattr(Params, 'trading_params')
        fixed_params: dict = getattr(Params, 'fixed_params')

        # Extract parameters
        intervals: str = general_params.get('intervals')
        full_coin_list: list = fixed_params.get('full_coin_list')
        methods: list = ml_params.get('methods')
        
        tuning_params: dict = modeling_vars.get('tuning_params')
        updating_params: dict = modeling_vars.get('updating_params')
        
        return {
            'intervals': intervals,
            'full_coin_list': full_coin_list,
            'methods': methods,

            'data_params': data_params,
            'ml_params': ml_params,
            'trading_params': trading_params,

            'tuning_params': tuning_params,
            'updating_params': updating_params
        }

    else:
        raise Exception(f'Invalid "file_name" was extracted: {file_name}.\n')


def log_params(
    logger: logging.Logger,
    log_keys: list = None,
    extra: dict = None,
    **params
) -> None:
    # Find file that called the function
    frame = inspect.currentframe()
    file_name = frame.f_back.f_code.co_filename.split('/')[-1]

    # Extract initial logger_msg
    logger_msg = {
        'trading.py': "\nTRADING PARAMS:\n",
        'data_processing.py': "\nDATA PROCESSING PARAMS:\n",
        'modeling.py': "\nMODELING PARAMS:\n",
        'data_warehousing.py': "\nDATA WAREHOUSING PARAMS:\n"
    }.get(file_name, None)

    if logger_msg is None:
        logger.critical('Invalid "file_name" was extracted: %s.', file_name)
        raise Exception(f'Invalid "file_name" was extracted: {file_name}.\n')

    legger_params = []

    for param_name, param_val in params.items():
        if log_keys is None or param_name in log_keys:
            if isinstance(param_val, dict):
                logger_msg += "%s:\n%s\n\n"
                legger_params.extend([param_name, pformat(param_val)])
            else:
                logger_msg += "%s: %s\n"
                legger_params.extend([param_name, param_val])
    
    logger.info(logger_msg, *legger_params, extra=extra)



