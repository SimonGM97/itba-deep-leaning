from PyTradeX.config.params import Params
from PyTradeX.modeling.model import Model
from PyTradeX.modeling.model_registry import ModelRegistry
from PyTradeX.pipeline.ml_pipeline import MLPipeline
from PyTradeX.utils.trading.trading_helper import load_ltp_lsl_stp_ssl
from PyTradeX.utils.general.logging_helper import get_logger

from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK, atpe
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll.base import scope
import pandas as pd
import numpy as np
from functools import partial
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
import time
import warnings
import secrets
import string
from tqdm import tqdm
from typing import Dict, List, Tuple
from copy import deepcopy
from pprint import pprint


warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")

"""
TODO:
    - Implement Models:
        - Add NBEATS parameters        
    - Add mlflow functionality
    - agregar un early_stopping que frene basado en no progress
    - ensemble models
        - pick top performers (on train data) > assign weight to their predictions
        - make predictions with top performers > decide on up or down using majority vote 
            (requiere numero impar de modelos) > the forecast will be a weighted average of 
            the majority
    - Try new models:
        - Add prophet parameters
        - Add Orbit parameters
"""

# TODO: TRY GPU WHENEVER POSSIBLE!
# TODO: Random forest is typically created by multiple overfitting trees, where the average of the predictions
#       have both low variance and low bias. Thus, each tree should have rather high hyperparameters
#   - Note that increasing n_estimators shouldn't increase overall variance
# TODO: Boosting models are typically created by multiple simple models with low variance and high bias, that combined
#       in sequence result in predictions with low bias and low variance. This each tree should have rather low hyperparameters.
#   - Note that increasing n_estimators does increase overall variance


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


class ModelTuner:

    int_parameters = [
        # naive_lv
        'naive_lv.lv',

        # naive_ma
        'naive_ma.period',

        # expo_smooth
        'seasonal_periods.seasonal_periods',

        # sarimax
        'sarimax.p', 
        'sarimax.d', 
        'sarimax.q',    
        'sarimax.seasonal_P',
        'sarimax.seasonal_D',
        'sarimax.seasonal_Q',
        'sarimax.seasonal_S',

        # random_forest
        'random_forest.n_estimators',
        'random_forest.max_depth',
        'random_forest.min_samples_split',

        # lightgbm
        'lightgbm.n_estimators',
        'lightgbm.max_depth',
        'lightgbm.min_child_samples',
        'lightgbm.num_leaves',

        # prophet
        'prophet.n_changepoints',

        # lstm
        'lstm.layers',
        'lstm.units',
        'lstm.batch_size',

        # n_beats
        'n_beats.blocks_per_stack', 
        'n_beats.layer_width', 
        'n_beats.batch_size',
    ]
    
    choice_parameters = {
        # PCA
        'pca': [True, False],

        # Trading Parameters
        'follow_flow': [True, False],
        'certainty_threshold': [0.55, 0.575, 0.6, 0.65],
        'tp': [True, False],
        'sl': [True], # False
        # 'long_permission': [True, False],
        # 'short_permission': [True, False],

        # naive_lv
        'naive_lv.lv': [1] + [lv+1 for lv in Params.data_params.get('lag_periods')],

        # naive_ma
        'naive_ma.period': [lv+1 for lv in Params.data_params.get('lag_periods') if lv > 1],
        'naive_ma.weight_type': ['normal', 'decreasing'],

        # expo_smooth
        'expo_smooth.trend': ["additive", "multiplicative", None],
        'expo_smooth.damped_trend': [True, False],
        'expo_smooth.seasonal': ["additive", "multiplicative", None],
        'expo_smooth.seasonal_periods': [12, 48, 48*7],

        # sarimax
        'sarimax.trend': [None, 'ct'],
        'sarimax.seasonal_S': [0, 6, 12, 24],

        # random_forest
        'random_forest.max_features': [1.0, 'sqrt'],

        # lightgbm
        'lightgbm.objective': ['regression', 'regression_l1', 'huber', 'quantile', 'mape'],
        'lightgbm.boosting_type': ['gbdt', 'dart'],

        # xgboost
        'xgboost.objective': ['reg:squarederror', 'reg:absoluteerror', 'reg:quantileerror'], # , 'reg:pseudohubererror'
        'xgboost.booster': ['gbtree', 'dart'],

        # prophet
        'prophet.growth': ['linear', 'logistic', 'flat'],
        'prophet.holidays': [True, False],
        'prophet.yearly_seasonality': ['auto', True, False],
        'prophet.monthly_seasonality': [True, False],
        'prophet.weekly_seasonality': ['auto', True, False],
        'prophet.daily_seasonality': ['auto', True, False],
        'prophet.seasonality_mode': ['additive', 'multiplicative'],

        # orbit
        'orbit.orbit_model': ['DLT', 'ETS', 'LGT', 'KTR'],

        # lstm
        'lstm.topology': ['classic', 'bidirectional', 'convolutional'],
        'lstm.units': [16, 32, 64, 128, 256],
        'lstm.batch_size': [16, 32, 64, 128, 256],

        # n_beats
        'n_beats.num_stacks': [4, 8, 16, 32, 64, 128],  # blocks_per_stack - , '128' , '256'
        'n_beats.layer_widths': [4, 8, 16, 32, 64, 128], # , '256'
        'n_beats.batch_size': [4, 8, 16, 32, 64], # , '128' , '256'
    }

    # Model Type Choices
    model_type_choices = [
        # Naive Last Value Search Space
        {
            "algorithm": 'naive_lv',
            "naive_lv.lv": scope.int(hp.choice('naive_lv.lv', choice_parameters['naive_lv.lv']))
        },

        # Mooving Average Search Space
        {
            "algorithm": 'naive_ma',
            "naive_ma.period": scope.int(hp.choice('naive_ma.period', choice_parameters['naive_ma.period'])),
            "naive_ma.weight_type": hp.choice('naive_ma.weight_type', choice_parameters['naive_ma.weight_type'])
        },

        # Exponential Smoothing Search Space
        {
            "algorithm": 'expo_smooth',
            "expo_smooth.trend": hp.choice('expo_smooth.trend', choice_parameters['expo_smooth.trend']),
            "expo_smooth.damped_trend": hp.choice('expo_smooth.damped_trend', choice_parameters['expo_smooth.damped_trend']),
            "expo_smooth.seasonal": hp.choice('expo_smooth.seasonal', choice_parameters['expo_smooth.seasonal']),
            "expo_smooth.seasonal_periods": scope.int(hp.choice('expo_smooth.seasonal_periods', choice_parameters['expo_smooth.seasonal_periods']))
        },

        # Sarimax Search Space
        {
            "algorithm": 'sarimax',
            "sarimax.trend": hp.choice('sarimax.trend', choice_parameters['sarimax.trend']),
            "sarimax.p": scope.int(hp.quniform('sarimax.p', 0, 6, 1)),
            # "sarimax.d": scope.int(hp.quniform('sarimax.d', 0, 1, 1)),
            "sarimax.q": scope.int(hp.quniform('sarimax.q', 0, 3, 1)),
            "sarimax.seasonal_P": scope.int(hp.quniform('sarimax.seasonal_P', 0, 3, 1)),
            # "sarimax.seasonal_D": scope.int(hp.quniform('sarimax.seasonal_D', 0, 1, 1)),
            "sarimax.seasonal_Q": scope.int(hp.quniform('sarimax.seasonal_Q', 0, 2, 1)),
            "sarimax.seasonal_S": scope.int(hp.choice('sarimax.seasonal_S', choice_parameters['sarimax.seasonal_S']))
        },

        # Random Forest Search Space
        {
            "algorithm": 'random_forest',
            "random_forest.n_estimators": scope.int(hp.quniform('random_forest.n_estimators', 15, 400, 1)),
            "random_forest.max_depth": scope.int(hp.quniform('random_forest.max_depth', 15, 200, 1)),
            "random_forest.min_samples_split": scope.int(hp.quniform('random_forest.min_samples_split', 5, 200, 1)),
            "random_forest.max_features": hp.choice('random_forest.max_features', choice_parameters['random_forest.max_features']),
        },
        
        # Lightgbm Search Space
        {
            "algorithm": 'lightgbm',
            "lightgbm.objective": hp.choice('lightgbm.objective', choice_parameters['lightgbm.objective']),
            "lightgbm.boosting_type": hp.choice('lightgbm.boosting_type', choice_parameters['lightgbm.boosting_type']),
            "lightgbm.n_estimators": scope.int(hp.quniform('lightgbm.n_estimators', 5, 300, 1)),
            "lightgbm.max_depth": scope.int(hp.quniform('lightgbm.max_depth', 1, 150, 1)),
            "lightgbm.min_child_samples": scope.int(hp.quniform('lightgbm.min_child_samples', 5, 100, 1)),
            "lightgbm.learning_rate": hp.loguniform('lightgbm.learning_rate', np.log(0.001), np.log(0.3)),
            "lightgbm.num_leaves": scope.int(hp.quniform('lightgbm.num_leaves', 5, 150, 1)),
            "lightgbm.colsample_bytree": hp.uniform('lightgbm.colsample_bytree', 0.6, 1)
        },

        # XGBoost Search Space
        {
            "algorithm": 'xgboost',
            "xgboost.objective": hp.choice('xgboost.objective', choice_parameters['xgboost.objective']),
            "xgboost.booster": hp.choice('xgboost.booster', choice_parameters['xgboost.booster']),
            "xgboost.eta": hp.loguniform('xgboost.eta', np.log(0.005), np.log(0.4)), # learning_rate
            # "xgboost.gamma": None,
            "xgboost.n_estimators": scope.int(hp.quniform('xgboost.n_estimators', 5, 250, 1)),
            "xgboost.max_depth": scope.int(hp.quniform('xgboost.max_depth', 1, 120, 1)),
            # "xgboost.min_child_weight": scope.int(hp.quniform('xgboost.min_child_weight', 5, 90, 1)),
            "xgboost.colsample_bytree": hp.uniform('xgboost.colsample_bytree', 0.6, 1),
            "xgboost.lambda": hp.loguniform('xgboost.lambda', np.log(0.001), np.log(5)),
            "xgboost.alpha": hp.loguniform('xgboost.alpha', np.log(0.001), np.log(5)),
            "xgboost.max_leaves": scope.int(hp.quniform('xgboost.max_leaves', 5, 120, 1)),
        },

        # Prophet Search Space
        {
            "algorithm": 'prophet',

            "prophet.growth": hp.choice('prophet.growth', choice_parameters['prophet.growth']),
            "prophet.changepoint_range": hp.uniform('prophet.changepoint_range', 0.8, 0.95),
            "prophet.changepoint_prior_scale": hp.loguniform('prophet.changepoint_prior_scale', np.log(0.001), np.log(0.5)),
            "prophet.seasonality_prior_scale": hp.loguniform('prophet.seasonality_prior_scale', np.log(0.01), np.log(10.0)),
            # "prophet.n_changepoints": scope.int(hp.quniform('prophet.n_changepoints', 0, 20, 1)),

            "prophet.holidays": hp.choice('prophet.holidays', choice_parameters['prophet.holidays']),
            # "prophet.yearly_seasonality": hp.choice('prophet.yearly_seasonality', choice_parameters['prophet.yearly_seasonality']),
            "prophet.monthly_seasonality": hp.choice('prophet.monthly_seasonality', choice_parameters['prophet.monthly_seasonality']),
            "prophet.weekly_seasonality": hp.choice('prophet.weekly_seasonality', choice_parameters['prophet.weekly_seasonality']),
            "prophet.daily_seasonality": hp.choice('prophet.daily_seasonality', choice_parameters['prophet.daily_seasonality']),
            "prophet.seasonality_mode": hp.choice('prophet.seasonality_mode', choice_parameters['prophet.seasonality_mode']),
        },

        # Orbit Search Space
        {
            "algorithm": 'orbit',
            "orbit.orbit_model": hp.choice('orbit.orbit_model', choice_parameters['orbit.orbit_model'])
        },

        # LSTM Search Space
        {
            "algorithm": 'lstm',
            # "lstm.topology": hp.choice('lstm.topology', choice_parameters['lstm.topology']),
            "lstm.layers": scope.int(hp.quniform('lstm.layers', 1, 10, 1)), 
            "lstm.units": scope.int(hp.choice('lstm.units', choice_parameters['lstm.units'])),
            "lstm.dropout": hp.uniform('lstm.dropout', 0.0, 0.5),
            "lstm.recurrent_dropout": hp.uniform('lstm.recurrent_dropout', 0.0, 0.5),
            "lstm.learning_rate": hp.loguniform('lstm.learning_rate', np.log(0.001), np.log(0.1)),
            # "lstm.batch_size": scope.int(hp.choice('lstm.batch_size', choice_parameters['lstm.batch_size'])),
        },

        # N-Beats Search Space
        {
            "algorithm": 'n_beats',
            # "n_beats.input_chunk_length": scope.int(hp.quniform('n_beats.input_chunk_length', 10, 50, 1)),
            # "n_beats.output_chunk_length": scope.int(hp.quniform('n_beats.output_chunk_length', 1, 10, 1)),
            "n_beats.num_stacks": scope.int(hp.choice('n_beats.num_stacks', choice_parameters['n_beats.num_stacks'])),
            "n_beats.num_blocks": scope.int(hp.quniform('n_beats.num_blocks', 1, 10, 1)),
            "n_beats.num_layers": scope.int(hp.quniform('n_beats.num_layers', 2, 6, 1)),
            "n_beats.layer_widths": scope.int(hp.choice('n_beats.layer_widths', choice_parameters['n_beats.layer_widths'])),
            # "n_beats.layer_widths": hp.choice('n_beats.layer_widths', [
            #     [scope.int(hp.quniform('n_beats.layer_width_1', 32, 512, 1))]*4,
            #     [scope.int(hp.quniform('n_beats.layer_width_2', 32, 512, 1))]*8,
            #     [scope.int(hp.quniform('n_beats.layer_width_3', 32, 512, 1))]*16,
            # ]),
            "n_beats.dropout": hp.uniform('n_beats.dropout', 0, 0.5),
            "n_beats.learning_rate": hp.loguniform('n_beats.learning_rate', np.log(0.001), np.log(0.3)), # -6, -1
            "n_beats.batch_size": scope.int(hp.choice('n_beats.batch_size', choice_parameters['n_beats.batch_size']))
        },

        # TFT Search Space
    ]
    
    def __init__(
        self, 
        intervals: str = Params.general_params.get("intervals"),
        coin_names: list = Params.fixed_params.get("full_coin_list"),
        methods: list = Params.ml_params.get("methods"),

        data_params: dict = deepcopy(Params.data_params),
        ml_params: dict = deepcopy(Params.ml_params),
        trading_params: dict = deepcopy(Params.trading_params)
    ) -> None:
        # General params
        self.intervals = intervals
        self.coin_names = coin_names
        self.methods = methods

        # Update Choice Parameters
        self.choice_parameters['coin_name'] = coin_names
        self.choice_parameters['method'] = methods

        # Data params
        self.data_params = data_params

        # ML params
        self.algorithms = ml_params.get('algorithms')
        self.n_candidates = ml_params.get('n_candidates')

        self.ml_params = ml_params

        # Trading params
        self.trading_params = trading_params

        self.model_type_choices = [
            choice for choice in self.model_type_choices if choice['algorithm'] in self.algorithms
        ]
        
        # Complete Search Space
        self.search_space = {
            "coin_name": hp.choice('coin_name', self.choice_parameters['coin_name']),
            "method": hp.choice('method', self.choice_parameters['method']),
            "pca": hp.choice('pca', self.choice_parameters['pca']),
            "follow_flow": hp.choice('follow_flow', self.choice_parameters['follow_flow']),
            "certainty_threshold": hp.choice('certainty_threshold', self.choice_parameters['certainty_threshold']),
            "tp": hp.choice('tp', self.choice_parameters['tp']),
            "sl": hp.choice('sl', self.choice_parameters['sl']),
            # "long_permission": hp.choice('long_permission', self.choice_parameters['long_permission']),
            # "short_permission": hp.choice('short_permission', self.choice_parameters['short_permission']),
            "model_type": hp.choice('model_type', self.model_type_choices)
        }

        self.is_first_round = False
        self.ltp_lsl_stp_ssl: Dict[str, List[float]] = {}

        # Define load attrs
        self.model_registry: ModelRegistry = None

        self.lfm_models: List[Model] = []
        self.gfm_models: List[Model] = []       

        self.load()

    def model_filter(
        self, 
        m: Model,
        include_test_filter: bool = True
    ) -> bool:
        if (
            isinstance(m, Model)
            and isinstance(m.val_table, pd.DataFrame) 
            and m.val_table.shape[0] > 0
            and m.val_table.trading_metric > 0
        ):
            if include_test_filter:
                if (
                    isinstance(m.test_table, pd.DataFrame)
                    and m.test_table.shape[0] > 0
                    and m.test_table.trading_metric > 0
                ):
                    return True
                else:
                    LOGGER.warning(
                        'Model %s (%s | %s) was filtered out in model_filter.\n'
                        'Model test_table:\n%s\n',
                        m.model_id, m.stage, m.model_class, m.test_table
                    )
                    return False
            return True
        LOGGER.warning(
            'Model %s (%s | %s) was filtered out in model_filter.\n'
            'Model val_table:\n%s\n',
            m.model_id, m.stage, m.model_class, m.val_table
        )
        return False

    def parameter_configuration(
        self, 
        parameters_list: List[dict],
        complete_parameters: bool = False,
        choice_parameters: str = 'index', 
        debug: bool = False
    ) -> List[dict]:
        if choice_parameters not in ['index', 'values']:
            LOGGER.critical('Invalid "choice_parameters": %s.', choice_parameters)
            raise Exception(f'Invalid "choice_parameters": {choice_parameters}.\n\n')

        # if debug:
        #     print('Input parameters_list:\n')
        #     pprint(parameters_list)
        #     print('\n\n')

        int_types = [int, np.int64, np.int32] #, float, np.float32 ,np.float64]

        for parameters in parameters_list:
            # Check "algorithm" parameter
            if 'algorithm' not in parameters.keys() and type(parameters['model_type']) in int_types:
                parameters['algorithm'] = self.algorithms[parameters['model_type']]
            elif 'algorithm' in parameters.keys() and type(parameters['algorithm']) == str and type(parameters['model_type']) in int_types:
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])

            # Check "model_type" parameter
            if parameters['model_type'] is None:
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])
            if type(parameters['model_type']) == dict:
                parameters.update(**parameters['model_type'])
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])

            # Check "method" parameter
            if 'method' in self.search_space.keys() and 'method' not in parameters.keys():
                if choice_parameters == 'index':
                    parameters['method'] = 1
                else:
                    parameters['method'] = 'return'

        # Complete Dummy Parameters
        if complete_parameters:
            dummy_list = []
            for model_type in self.model_type_choices:
                dummy_list.extend(list(model_type.keys()))

            for parameters in parameters_list:
                for dummy_parameter in dummy_list:
                    if dummy_parameter not in parameters.keys():
                        parameters[dummy_parameter] = 0
        else:
            for parameters in parameters_list:
                filtered_keys = list(self.search_space.keys())
                filtered_keys += list(self.model_type_choices[parameters['model_type']].keys())

                dummy_parameters = parameters.copy()
                for parameter in dummy_parameters.keys():
                    if parameter not in filtered_keys:
                        parameters.pop(parameter)

        # Check Choice Parameters
        if choice_parameters == 'index':                   
            for parameters in parameters_list:
                choice_keys = [k for k in self.choice_parameters.keys() 
                               if k in parameters.keys() and type(parameters[k]) not in int_types]  
                for choice_key in choice_keys:
                    try:
                        parameters[choice_key] = self.choice_parameters[choice_key].index(parameters[choice_key])
                    except Exception as e:
                        LOGGER.warning(
                            'Exception in: \n'
                            'choice_key: %s\n'
                            'type(parameters[%s]): %s\n'
                            'Exception: %s\n',
                            choice_key, choice_key, type(parameters[choice_key]), e
                        )
        else:            
            for parameters in parameters_list:
                choice_keys = [k for k in self.choice_parameters.keys() 
                               if k in parameters.keys() and type(parameters[k]) in int_types]
                for choice_key in choice_keys:
                    try:
                        parameters[choice_key] = self.choice_parameters[choice_key][parameters[choice_key]]
                    except Exception as e:
                        LOGGER.warning(
                            'Exception in: \n'
                            'choice_key: %s\n'
                            'type(parameters[%s]): %s\n'
                            'Exception: %s\n',
                            choice_key, choice_key, type(parameters[choice_key]), e
                        )

        # Check int parameters
        for parameters in parameters_list:
            for parameter in parameters:
                if parameter in self.int_parameters and parameters[parameter] is not None:
                    parameters[parameter] = int(parameters[parameter])

        if debug:
            print('New parameters_list:\n')
            pprint(parameters_list)
            print('\n\n')

        return parameters_list

    def prepare_hyper_parameters(
        self,
        parameters: dict
    ) -> dict:
        hyper_param_choices = [d for d in self.model_type_choices if d['algorithm'] == parameters['algorithm']][0]

        parameters['hyper_parameters'] = {
            hyper_param: parameters.pop(hyper_param) 
            for hyper_param in hyper_param_choices.keys()
            if hyper_param != 'algorithm'
        }

        parameters['hyper_parameters'] = {
            k.replace(f'{parameters["algorithm"]}.', ''): v
            for k, v in parameters['hyper_parameters'].items()
        }

        return parameters
    
    def prepare_trading_parameters(
        self,
        parameters: dict
    ) -> dict:
        parameters['trading_parameters'] = {
            'follow_flow': parameters.pop('follow_flow'),
            'certainty_threshold': parameters.pop('certainty_threshold'),
            'long_permission': True, # parameters.pop('long_permission'),
            'short_permission': True, # parameters.pop('short_permission'),
            'max_leverage': self.trading_params.get('max_leverage')
        }
        # Add take_proffits & stop_loss
        parameters['trading_parameters']['tp'] = (None, None)
        parameters['trading_parameters']['sl'] = (None, None)

        if parameters['tp'] or parameters['sl']:
            ltp, lsl, stp, ssl = self.ltp_lsl_stp_ssl[parameters['coin_name']]
            if parameters['tp']:
                parameters['trading_parameters']['tp'] = (ltp, stp)
            if parameters['sl']:
                parameters['trading_parameters']['sl'] = (lsl, ssl)

        parameters.pop('tp')
        parameters.pop('sl')

        if 'optimized_trading_parameters' not in parameters.keys():
            parameters['optimized_trading_parameters'] = None

        return parameters

    def prepare_parameters(
        self,
        parameters: dict,
        selected_features: Dict[str, List[str]],
        warm_start_params: dict = None,
        reverse_forecasts: bool = None,
        debug: bool = False
    ) -> dict:
        if debug:
            t1 = time.time()
            print("input parameters:\n"
                  "{")
            for key in parameters:
                if key != 'selected_features':
                    print(f"    '{key}': {parameters[key]}")
            print('}\n\n')

        parameters = self.parameter_configuration(
            parameters_list=[parameters],
            complete_parameters=False,
            choice_parameters='values',
            debug=False
        )[0]

        # Add intervals, refit_model & refit_freq to parameters
        if 'version' not in parameters.keys():
            parameters['version'] = '0.0'
        if 'stage' not in parameters.keys():
            parameters['stage'] = 'development'

        if 'intervals' not in parameters.keys():
            parameters['intervals'] = self.intervals
        if 'lag' not in parameters.keys():
            parameters['lag'] = self.data_params.get('lag')
        if 'model_class' not in parameters.keys():
            parameters['model_class'] = 'LFM'
        
        if 'refit_model' not in parameters.keys():
            parameters['refit_model'] = self.ml_params.get('refit_model')
        if 'refit_freq' not in parameters.keys():
            parameters['refit_freq'] = int(self.ml_params.get('refit_freq') * self.data_params.get('periods'))

        if 'reverse_forecasts' not in parameters.keys():
            parameters['reverse_forecasts'] = reverse_forecasts
        if 'model_type' in parameters.keys():
            parameters.pop('model_type')

        # Prepare Hyper Parameters
        parameters = self.prepare_hyper_parameters(
            parameters=parameters
        )

        # Prepara Trading Parameters
        parameters = self.prepare_trading_parameters(
            parameters=parameters
        )
        if 'optimized_trading_parameters' not in parameters.keys():
            parameters['optimized_trading_parameters'] = parameters['trading_parameters'].copy()

        # Add Selected Features
        if 'selected_features' not in parameters.keys():
            parameters['selected_features'] = deepcopy(selected_features[parameters['method']])

        # Update parameters with warm_start params (if it's the first run)
        if self.is_first_round and warm_start_params is not None:
            parameters['model_id'] = warm_start_params['model_id']
            # parameters['version'] = warm_start_params['version']
            parameters['stage'] = warm_start_params['stage']
            # parameters['model_class'] = warm_start_params['model_class']

            # parameters['refit_model'] = warm_start_params['refit_model']
            # parameters['refit_freq'] = warm_start_params['refit_freq']
            parameters['reverse_forecasts'] = warm_start_params['reverse_forecasts']

            parameters['lag'] = warm_start_params['lag']

            # Reset self.is_first_round
            self.is_first_round = False

        if debug:
            print("new parameters:\n"
                  "{")
            for key in parameters:
                if key == 'selected_features' and parameters[key] is not None:
                    print(f"    '{key}' (len): {len(parameters[key])}")
                else:
                    print(f"    '{key}': {parameters[key]}")
            print('}\n')
            print(f'Time taken to prepare parameters: {round(time.time() - t1, 1)} sec.\n\n')
        
        return parameters
    
    def objective(
        self, 
        parameters: dict,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        selected_features: Dict[str, List[str]],
        reduced_tuning_periods: int = None,
        warm_start_params: str = None,
        debug: bool = False
    ) -> dict:
        try:
            # Prepare parameters
            parameters = self.prepare_parameters(
                parameters=parameters,
                selected_features=selected_features,
                warm_start_params=warm_start_params,
                reverse_forecasts=False,
                debug=debug # debug
            )

            # Instanciate MLPipeline
            ml_pipeline = MLPipeline(
                pipeline_params=parameters,
                ml_params=self.ml_params,
                trading_params=self.trading_params
            )

            # Run Model Build Pipeline
            model: Model = ml_pipeline.build_pipeline(
                ml_datasets=ml_datasets,
                reduced_tuning_periods=reduced_tuning_periods,
                model=None,
                ignore_update=False,
                find_val_table=True,
                re_fit_train_val=False,
                find_test_table=False,
                find_opt_table=False,
                tune_opt_table=False,
                find_feature_importance=False,
                debug=debug,
                dec_timeout=4.15*60
            )

            # Delete ml_pipeline from memory
            del ml_pipeline
            
            # Update self.lfm_models: Only update dev_models for new models
            self.update_lfm_models(
                new_candidate=model,
                debug=debug # debug
            )

            # Extract tuning_metric from val_table
            tuning_metric = model.val_table.tuning_metric

            if debug:
                i = 1
                attr_names = [
                    'model_id', 'coin_name', 'algorithm', 'method', 'pca'
                ]
                for model in self.lfm_models[:5]:
                    print(f"Model n: {i}\n"
                        "{")
                    for attr_name in attr_names:
                        print(f"    '{attr_name}': {getattr(model, attr_name)}")
                    print(f'    val tuning_metric: {tuning_metric}')
                    print('}\n\n')
                    i += 1
            
            # Return Loss
            return {'loss': -tuning_metric, 'status': STATUS_OK}
        except Exception as e:
            LOGGER.warning(
                'Skipping iteration.\n'
                'Exception: %s\n', e
            )
                # f'Parameters:\n{parameters}\n\n')
            return {'loss': np.inf, 'status': STATUS_OK}

    def reset_dev_lfm_models(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        debug: bool = False
    ) -> None:
        for idx in tqdm(range(len(self.lfm_models))):
            # Extract model from self.lfm_models
            model = self.lfm_models[idx]

            if model.stage == 'development':
                # Reset model.version
                model.version = '0.0'

                # Instanciate MLPipeline
                ml_pipeline = MLPipeline(
                    pipeline_params=model.pipeline_params,
                    ml_params=self.ml_params,
                    trading_params=self.trading_params
                )
                
                # Run build_pipeline
                self.lfm_models[idx] = ml_pipeline.build_pipeline(
                    ml_datasets=ml_datasets,
                    model=None, # model.model will be re-created & re-fitted
                    ignore_update=False,
                    find_val_table=True,
                    re_fit_train_val=False,
                    find_test_table=False,
                    find_opt_table=False,
                    tune_opt_table=False,
                    find_feature_importance=False,
                    debug=debug
                )

                # Delete ml_pipeline from memory
                del ml_pipeline

                print(f'Re-setted LFM Model val_table:')
                self.lfm_models[idx].val_table.show_attrs(residuals_attrs=False)

                # Save updated model
                self.lfm_models[idx].save()

        # Filter out unperformant Models
        filter_function = partial(
            self.model_filter,
            include_test_filter=False
        )
        self.lfm_models = list(filter(filter_function, self.lfm_models))

        # Sort Models based on Val Performance
        self.lfm_models = self.model_registry.sort_models(
            models=self.lfm_models,
            trading_metric=False,
            by_table='val'
        )

    def update_lfm_models(
        self,
        new_candidate: Model,
        debug: bool = False
    ) -> None:
        if new_candidate.val_table.tuning_metric > 0:
            if new_candidate.model_id not in [m.model_id for m in self.lfm_models]:
                if (
                    len(self.lfm_models) < self.n_candidates
                    or new_candidate.val_table.tuning_metric > self.lfm_models[-1].val_table.tuning_metric
                ):
                    # Add new_candidate to self.lfm_models
                    self.lfm_models.append(new_candidate)
                
                    # Drop duplicate Models (keeping most performant)
                    self.lfm_models = self.model_registry.drop_duplicate_models(
                        models=self.lfm_models,
                        from_=None,
                        trading_metric=False,
                        by_table='val',
                        debug=debug
                    )
                
                    # Sort Models
                    self.lfm_models = self.model_registry.sort_models(
                        models=self.lfm_models,
                        trading_metric=False,
                        by_table='val'
                    )

                    if len(self.lfm_models) > self.n_candidates:
                        self.lfm_models = self.lfm_models[:self.n_candidates]
                    
                    if new_candidate.model_id in [m.model_id for m in self.lfm_models]:
                        print(f'Model {new_candidate.model_id} ({new_candidate.stage} | {new_candidate.model_class}) was added to self.lfm_models.\n')
            else:
                LOGGER.warning(
                    '%s is already in lfm_models.\n'
                    'Note: This should only happen for evaluation of warm models.\n',
                    new_candidate.model_id
                )

    def update_gfm_models(
        self,
        new_candidate: Model,
        debug: bool = False
    ) -> None:
        if new_candidate.model_id not in [m.model_id for m in self.gfm_models]:
            if (
                len(self.gfm_models) < self.n_candidates
                or new_candidate.val_table.tuning_metric > self.gfm_models[-1].val_table.tuning_metric
            ):
                # Add new_candidate to self.lfm_models
                self.gfm_models.append(new_candidate)
            
                # Drop duplicate Models (keeping most performant)
                self.gfm_models = self.model_registry.drop_duplicate_models(
                    models=self.gfm_models,
                    from_=None,
                    trading_metric=False,
                    by_table='val',
                    debug=debug
                )
            
                # Sort Models
                self.gfm_models = self.model_registry.sort_models(
                    models=self.gfm_models,
                    trading_metric=False,
                    by_table='val'
                )

                if len(self.gfm_models) > self.n_candidates:
                    self.gfm_models = self.gfm_models[:self.n_candidates]
                
                if new_candidate.model_id in [m.model_id for m in self.gfm_models]:
                    print(f'Model {new_candidate.model_id} ({new_candidate.stage} | {new_candidate.model_class}) was added to self.gfm_models.\n')
        else:
            LOGGER.warning(
                '%s is already in gfm_models.\n'
                'Note: This should only happen for evaluation of warm models.\n',
                new_candidate.model_id
            )

    def introduce_lfms_into_gfm(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        debug: bool = False
    ) -> None:
        for model in tqdm(self.lfm_models):
            # Find repeated Models
            repeated_models = self.model_registry.find_repeated_models(
                new_model=model, 
                models=self.gfm_models,
                from_=None
            )

            # Define if Model is already found in self.gfm_models
            if len(repeated_models) > 0:
                print(f'Model {model.model_id} ({model.stage} | {model.model_class} - {model.intervals}) is already in GFM list.')
            else:
                # Define gfm_params
                gfm_pipeline_params: dict = deepcopy(model.pipeline_params)

                # Update gfm_params
                gfm_pipeline_params.update(**{
                    # 'coin_name': None,
                    'model_id': None,
                    'model_class': 'GFM',
                    'version': '0.0',
                    'stage': 'development',
                    'refit_model': False,
                    'refit_freq': np.inf,
                    'pca': False,
                    'optimized_trading_parameters': None,
                    'reverse_forecasts': False,
                    'train_coins': None # self.coin_names
                })

                # Instanciate MLPipeline
                ml_pipeline = MLPipeline(
                    pipeline_params=gfm_pipeline_params,
                    ml_params=self.ml_params,
                    trading_params=self.trading_params
                )

                # Run build_pipeline
                gfm: Model = ml_pipeline.build_pipeline(
                    ml_datasets=ml_datasets,
                    model=None,
                    ignore_update=False,
                    find_val_table=True,
                    re_fit_train_val=False,
                    find_test_table=False,
                    find_opt_table=False,
                    find_feature_importance=False,
                    debug=debug
                )

                # Delete ml_pipeline from memory
                del ml_pipeline

                print(f'New GFM Model val_table:')
                gfm.val_table.show_attrs(residuals_attrs=False)

                # Save Model
                gfm.save()

                # Update self.gfm_models
                self.update_gfm_models(new_candidate=gfm)

        # Sort Models based on Val Performance
        self.gfm_models = self.model_registry.sort_models(
            models=self.gfm_models,
            trading_metric=True,
            by_table='val'
        )

    def find_trading_gfm(
        self,
        selected_features: Dict[str, List[str]],
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        debug: bool = False
    ):
        # Find full features
        full_features: List[str] = []
        for key, features in selected_features.items():
            full_features.extend([f for f in features if f not in full_features])

        def ignore_feature(f: str, full_filter: bool = True):
            if full_filter:
                if (
                    f.endswith('_min')
                    or f.endswith('_max')
                    or f.endswith('range')
                    or f.endswith('_std')
                    # or '_lag' in f
                    or '_return' in f
                    or '_acceleration' in f
                    or '_jerk' in f
                ):
                    return True
                return False
            else:
                if (
                    f.endswith('_min')
                    or f.endswith('_max')
                    or f.endswith('range')
                    or f.endswith('_std')
                ):
                    return True
                return False

        # Find coin features
        coin_features = [f for f in full_features if f.startswith('coin') and not(ignore_feature(f, full_filter=False))]

        # Find trading features
        trading_features = [f for f in full_features if f.startswith('trading') and not(ignore_feature(f))]

        # Find manual features
        manual_features = [f for f in full_features if f.startswith('manual') and not(ignore_feature(f))]

        # Find keep_features
        keep_features = coin_features + trading_features + manual_features

        if debug:
            print(f'len(keep_features): {len(keep_features)}\n')
            print(f'coin_features ({len(coin_features)}):')
            pprint(coin_features)
            print('\n\n')
            print(f'trading_features ({len(trading_features)}):')
            pprint(trading_features)
            print('\n\n')
            print(f'manual_features ({len(manual_features)}):')
            pprint(manual_features)
            print('\n\n')

        # Sort self.gfm_models
        self.gfm_models = self.model_registry.sort_models(
            models=self.gfm_models,
            trading_metric=True,
            by_table='val'
        )

        # Define best Model gfm_params
        gfm_pipeline_params: dict = deepcopy(self.gfm_models[0].pipeline_params)

        # Update gfm_params
        gfm_pipeline_params.update(**{
            # 'coin_name': None,
            'model_id': ''.join(secrets.choice(string.ascii_letters) for i in range(10)) + 'tradingGFM',
            'model_class': 'GFM',
            'version': '0.0',
            'stage': 'development',
            'refit_model': False,
            'refit_freq': np.inf,
            'pca': False,
            'optimized_trading_parameters': None,
            'reverse_forecasts': False,
            'train_coins': None, # self.coin_names
            'selected_features': keep_features
        })

        # Instanciate MLPipeline
        ml_pipeline = MLPipeline(
            pipeline_params=gfm_pipeline_params,
            ml_params=self.ml_params,
            trading_params=self.trading_params
        )

        # Build trading model
        trading_gfm: Model = ml_pipeline.build_pipeline(
            ml_datasets=ml_datasets,
            model=None,
            ignore_update=False,
            find_val_table=True,
            re_fit_train_val=False,
            find_test_table=False,
            find_opt_table=False,
            find_feature_importance=False,
            debug=debug
        )

        # Delete ml_pipeline from memory
        del ml_pipeline

        print(f'New Trading GFM Model val_table:')
        trading_gfm.val_table.show_attrs(residuals_attrs=False)

        # Save Model
        trading_gfm.save()

        # Update self.gfm_models
        self.update_gfm_models(new_candidate=trading_gfm)

        # Sort Models based on Val Performance
        self.gfm_models = self.model_registry.sort_models(
            models=self.gfm_models,
            trading_metric=True,
            by_table='val'
        )

    def evaluate_models(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        debug: bool = False
    ) -> None:
        # Evaluate LFMs
        for idx in tqdm(range(len(self.lfm_models))):
            # Extract model
            model: Model = self.lfm_models[idx]

            if model.optimized_table is None or model.optimized_table.shape[0] == 0:
                if model.val_table.trading_metric > 0:
                    # Instanciate MLPipeline
                    ml_pipeline = MLPipeline(
                        pipeline_params=model.pipeline_params,
                        ml_params=self.ml_params,
                        trading_params=self.trading_params
                    )
                    
                    # Run build_pipeline
                    self.lfm_models[idx] = ml_pipeline.build_pipeline(
                        ml_datasets=ml_datasets,
                        model=model,
                        ignore_update=False,
                        find_val_table=False,
                        re_fit_train_val=True,
                        find_test_table=True,
                        find_opt_table=True,
                        tune_opt_table=True,
                        find_feature_importance=True,
                        debug=debug
                    )

                    # Delete ml_pipeline from memory
                    del ml_pipeline

                    print(f'Evaluated LFM Model optimized_table:')
                    self.lfm_models[idx].optimized_table.show_attrs(residuals_attrs=False)

                    # Save Model
                    self.lfm_models[idx].save()
            else:
                LOGGER.warning(
                    'Model %s (%s | %s - %s) will not be evaluated, as it has already been evaluated.\n'
                    'Optimized table parameters:',
                    model.model_id, model.stage, model.model_class, model.intervals
                )
                model.optimized_table.show_attrs(general_attrs=False, residuals_attrs=False)

        # Evaluate GFMs
        for idx in tqdm(range(len(self.gfm_models))):
            # Extract model
            model: Model = self.gfm_models[idx]

            if model.optimized_table is None:
                # Instanciate MLPipeline
                ml_pipeline = MLPipeline(
                    pipeline_params=model.pipeline_params,
                    ml_params=self.ml_params,
                    trading_params=self.trading_params
                )
                
                # Run build_pipeline
                self.gfm_models[idx] = ml_pipeline.build_pipeline(
                    ml_datasets=ml_datasets,
                    model=model,
                    ignore_update=False,
                    find_val_table=False,
                    re_fit_train_val=True,
                    find_test_table=True,
                    find_opt_table=True,
                    tune_opt_table=True,
                    find_feature_importance=True,
                    debug=debug
                )

                # Delete ml_pipeline from memory
                del ml_pipeline

                LOGGER.info('Evaluated GFM Model optimized_table:')
                self.gfm_models[idx].optimized_table.show_attrs(residuals_attrs=False)

                # Save Model
                self.gfm_models[idx].save()
            else:
                LOGGER.warning(
                    'Model %s (%s | %s - %s) will not be evaluated, as it has already been evaluated.\n'
                    'Optimized table parameters:',
                    model.model_id, model.stage, model.model_class, model.intervals
                )
                model.optimized_table.show_attrs(general_attrs=False, residuals_attrs=False)

    def tune_models(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        selected_features: Dict[str, List[str]],
        reset_dev_lfm_models: bool = False,
        tune_lfm_models: bool = False,
        reduced_tuning_periods: int = None,
        tune_gfm_models: bool = False,
        evaluate_models: bool = False,
        debug: bool = False, 
        deep_debug: bool = False
    ) -> None:
        if deep_debug:
            debug = True

        """
        Prepare LFM Models
        """
        # Filter broken models
        # filter_function = partial(
        #     self.model_filter,
        #     include_test_filter=False
        # )
        # self.lfm_models = list(filter(filter_function, self.lfm_models))

        # Sort lfm_models, based on val_table
        self.lfm_models = self.model_registry.sort_models(
            models=self.lfm_models,
            trading_metric=False,
            by_table='val'
        )

        if debug and len(self.lfm_models) > 0:
            print('LFMs Val Tuning Metrics:\n')
            for model in self.lfm_models:
                print(f'{model.model_id}: {model.val_table.tuning_metric} '
                      f'({model.model_class}, {model.algorithm}, {model.coin_name}, {model.method}).')
            print('\n\n')
        
        # Find ltp, stp, lsl, ssl
        self.ltp_lsl_stp_ssl: Dict[str, Tuple] = load_ltp_lsl_stp_ssl(
            coin_name=None,
            intervals=self.intervals
        )

        if debug:
            print('self.ltp_lsl_stp_ssl:')
            pprint(self.ltp_lsl_stp_ssl)
            print('\n\n')
            
        """
        Re-set def LFMs
        """
        if reset_dev_lfm_models and len(self.lfm_models) > 0:
            print(f'Re-setting LFMs:')
            self.reset_dev_lfm_models(
                ml_datasets=ml_datasets,
                debug=deep_debug
            )

            if debug:
                print(f'New LFMs Val Tuning Metric:\n')
                for model in self.lfm_models:
                    print(f'{model.model_id}: {model.val_table.tuning_metric} '
                          f'({model.model_class}, {model.algorithm}, {model.coin_name}, {model.method}).')
                print('\n\n')

        """
        Tune LFMs
        """
        if tune_lfm_models:
            # Define Max Performance Threshold
            loss_threshold = -25

            if debug and len(self.lfm_models) > 0:
                print(f'Expected starting point: {self.lfm_models[0].val_table.tuning_metric}.\n' 
                      f'loss_threshold: {loss_threshold}\n\n')
            
            # Define warm_start_params & trials
            warm_models = [model for model in self.lfm_models + self.gfm_models if model.algorithm in self.algorithms]
            if len(warm_models) > 0 and warm_models[0].warm_start_params is not None:
                warm_start_params = deepcopy(warm_models[0].warm_start_params)

                if debug:
                    print('warm_start_params:')
                    pprint(warm_start_params)
                    print('\n\n')

                best_parameters_to_evaluate = self.parameter_configuration(
                    parameters_list=[warm_start_params],
                    complete_parameters=True,
                    choice_parameters='index',
                    debug=debug
                )
                trials = generate_trials_to_calculate(best_parameters_to_evaluate)
            else:
                warm_start_params = None
                trials = None

            # Delete warm models
            del warm_models

            # Define fixed params for self.objective
            fmin_objective = partial(
                self.objective,
                ml_datasets=ml_datasets,
                selected_features=selected_features,
                reduced_tuning_periods=reduced_tuning_periods,
                warm_start_params=warm_start_params,
                debug=deep_debug
            )
            
            print(f'\n\nTuning LFMs:\n')

            # Re-set self.is_first_round
            self.is_first_round = True

            # Run fmin to optimize objective function
            result = fmin(
                fn=fmin_objective,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.ml_params.get('max_evals'),
                timeout=self.ml_params.get('timeout_mins') * 60,
                loss_threshold=loss_threshold,
                trials=trials,
                verbose=True,
                show_progressbar=True,
                early_stop_fn=None  # early_stop.no_progress_loss(percent_increase=0.2), None
            )

            # Save LFMs
            for model in self.lfm_models:
                model.save()

            if reduced_tuning_periods is not None:
                print(f'Re-setting LFMs:')
                self.reset_dev_lfm_models(
                    ml_datasets=ml_datasets,
                    debug=deep_debug
                )

            # Save LFMs
            for model in self.lfm_models:
                model.save()

        """
        Tune GFMs
        """
        if tune_gfm_models:
            print("Tuning GFMs:\n")
            self.introduce_lfms_into_gfm(
                ml_datasets=ml_datasets,
                debug=deep_debug
            )

            print("Adding trading-only GFM:\n")
            self.find_trading_gfm(
                selected_features=selected_features,
                ml_datasets=ml_datasets,
                debug=debug
            )

            if debug:
                print('GFMs Val Trading Metric:\n')
                for model in self.gfm_models:
                    print(f'{model.model_id}: {model.val_table.trading_metric} '
                        f'({model.model_class}, {model.algorithm}, {model.coin_name}, {model.method}).')
                print('\n\n')

        """
        Evaluate Models
        """
        if evaluate_models:
            print("Evaluating Models:\n")
            self.evaluate_models(
                ml_datasets=ml_datasets,
                debug=deep_debug
            )

        """
        Save all Models
        """
        # Save LFMs
        for model in self.lfm_models:
            model.save()
            model.save_backup()

        # Save GFMs
        for model in self.gfm_models:
            model.save()
            model.save_backup()

        """
        Update Model Registry
        """
        # Add Dev Models to Registry
        self.model_registry.registry['development'] = [
            (m.model_id, m.model_class) for m in self.lfm_models + self.gfm_models
            if m.stage == 'development'
        ]

        if debug:
            print(f"Dev Models found:")
            pprint(self.model_registry.registry['development'])
            print('\n\n')

        # Find all models
        champion: Model = self.model_registry.load_prod_model(light=False)
        staging_models: List[Model] = self.model_registry.load_staging_models(light=False)
        dev_models: List[Model] = self.model_registry.load_dev_models(light=False)

        models = [champion] + staging_models + dev_models
        
        # Drop duplicate models
        models = self.model_registry.drop_duplicate_models(
            models=models
        )

        # Reassign model_registry.registry
        self.model_registry.registry = {
            "production": [(m.model_id, m.model_class) for m in models if m.stage == 'production'], 
            "staging": [(m.model_id, m.model_class) for m in models if m.stage == 'staging'], 
            "development": [(m.model_id, m.model_class) for m in models if m.stage == 'development']
        }

        # Save default self.model_registry.registry
        self.model_registry.save()

        # Update Registry
        self.model_registry.update_model_stages(
            update_champion=True,
            debug=debug
        )

        # Show ModelRegistry
        print(self.model_registry)

    def load(
        self,
        debug: bool = False
    ) -> None:
        # Load Model Registry
        self.model_registry = ModelRegistry(
            n_candidates=self.n_candidates,
            intervals=self.intervals
        )

        # Load Registry Models
        reg_models: List[Model] = (
            self.model_registry.load_dev_models() + 
            self.model_registry.load_staging_models() + 
            [self.model_registry.load_prod_model()]
        )

        # Populate self.lfm_models & self.gfm_models
        if len(reg_models) > 0:
            self.lfm_models: List[Model] = [m for m in reg_models if m is not None and m.model_class == 'LFM']
            self.gfm_models: List[Model] = [m for m in reg_models if m is not None and m.model_class == 'GFM']
        else:
            self.lfm_models: List[Model] = []
            self.gfm_models: List[Model] = []
        
        # Delete reg_models from memory
        del reg_models

        if debug:
            print(self)

    def __repr__(self) -> str:
        i = 1
        for model in self.lfm_models[:5]:
            print(f'Dev Model {i}:')
            pprint({
                'Model ID': model.model_id,
                'Coin name': model.coin_name,
                'Intervals': model.intervals,
                'Algorithm': model.algorithm,
                'Method': model.method,
                'PCA': model.pca,
                'Trading Parameters': model.trading_parameters,
                'Optimized Trading Parameters': model.optimized_trading_parameters,
                'Hyper-parameters': model.hyper_parameters
            })
            print(f'\nPerformances:')
            performances = {}
            if model.val_table is not None:
                performances['Val Table Trading Metric'] = model.val_table.trading_metric
            else:
                performances['Val Table Trading Metric'] = None

            if model.test_table is not None:
                performances['Test Table Trading Metric'] = model.test_table.trading_metric
            else:
                performances['Test Table Trading Metric'] = None

            if model.optimized_table is not None:
                performances['Optimized Table Trading Metric'] = model.optimized_table.trading_metric
            else:
                performances['Optimized Table Trading Metric'] = None
            
            pprint(performances)
            print('\n\n')
            i += 1
        return ''

