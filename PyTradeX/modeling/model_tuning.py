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
        # lstm
        'lstm.layers',
        'lstm.units',
        'lstm.batch_size',

        # # n_beats
        # 'n_beats.blocks_per_stack', 
        # 'n_beats.layer_width', 
        # 'n_beats.batch_size',
    ]
    
    choice_parameters = {
        # PCA
        'pca': [True, False],

        # Trading Parameters
        'follow_flow': [True, False],
        'certainty_threshold': [0.55], # , 0.575, 0.6, 0.65],
        'tp': [True, False],
        'sl': [True], # False
        # 'long_permission': [True, False],
        # 'short_permission': [True, False],

        # lstm
        # 'lstm.topology': ['classic', 'bidirectional', 'convolutional'],
        'lstm.units': [16, 32, 64, 128, 256],
        'lstm.batch_size': [16, 32, 64, 128, 256],

        # # n_beats
        # 'n_beats.num_stacks': [4, 8, 16, 32, 64, 128],  # blocks_per_stack - , '128' , '256'
        # 'n_beats.layer_widths': [4, 8, 16, 32, 64, 128], # , '256'
        # 'n_beats.batch_size': [4, 8, 16, 32, 64], # , '128' , '256'
    }

    # Model Type Choices
    model_type_choices = [
        # LSTM Search Space
        {
            "algorithm": 'lstm',
            # "lstm.topology": hp.choice('lstm.topology', choice_parameters['lstm.topology']),
            "lstm.layers": scope.int(hp.quniform('lstm.layers', 1, 4, 1)), 
            "lstm.units": scope.int(hp.choice('lstm.units', choice_parameters['lstm.units'])),
            "lstm.dropout": hp.uniform('lstm.dropout', 0.0, 0.5),
            "lstm.recurrent_dropout": hp.uniform('lstm.recurrent_dropout', 0.0, 0.5),
            "lstm.learning_rate": hp.loguniform('lstm.learning_rate', np.log(0.001), np.log(0.01)),
            # "lstm.batch_size": scope.int(hp.choice('lstm.batch_size', choice_parameters['lstm.batch_size'])),
        },

        # # N-Beats Search Space
        # {
        #     "algorithm": 'n_beats',
        #     # "n_beats.input_chunk_length": scope.int(hp.quniform('n_beats.input_chunk_length', 10, 50, 1)),
        #     # "n_beats.output_chunk_length": scope.int(hp.quniform('n_beats.output_chunk_length', 1, 10, 1)),
        #     "n_beats.num_stacks": scope.int(hp.choice('n_beats.num_stacks', choice_parameters['n_beats.num_stacks'])),
        #     "n_beats.num_blocks": scope.int(hp.quniform('n_beats.num_blocks', 1, 10, 1)),
        #     "n_beats.num_layers": scope.int(hp.quniform('n_beats.num_layers', 2, 6, 1)),
        #     "n_beats.layer_widths": scope.int(hp.choice('n_beats.layer_widths', choice_parameters['n_beats.layer_widths'])),
        #     # "n_beats.layer_widths": hp.choice('n_beats.layer_widths', [
        #     #     [scope.int(hp.quniform('n_beats.layer_width_1', 32, 512, 1))]*4,
        #     #     [scope.int(hp.quniform('n_beats.layer_width_2', 32, 512, 1))]*8,
        #     #     [scope.int(hp.quniform('n_beats.layer_width_3', 32, 512, 1))]*16,
        #     # ]),
        #     "n_beats.dropout": hp.uniform('n_beats.dropout', 0, 0.5),
        #     "n_beats.learning_rate": hp.loguniform('n_beats.learning_rate', np.log(0.001), np.log(0.3)), # -6, -1
        #     "n_beats.batch_size": scope.int(hp.choice('n_beats.batch_size', choice_parameters['n_beats.batch_size']))
        # },

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
        self.models: List[Model] = []       

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
            ltp, lsl, stp, ssl = self.ltp_lsl_stp_ssl['ETH'] # [parameters['coin_name']]
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

        # Add coin_name, intervals, refit_model & refit_freq to parameters
        if 'coin_name' not in parameters.keys():
            parameters['coin_name'] = 'ETH'
        if 'version' not in parameters.keys():
            parameters['version'] = '0.0'
        if 'stage' not in parameters.keys():
            parameters['stage'] = 'development'

        if 'intervals' not in parameters.keys():
            parameters['intervals'] = self.intervals
        if 'lag' not in parameters.keys():
            parameters['lag'] = self.data_params.get('lag')
        if 'model_class' not in parameters.keys():
            parameters['model_class'] = 'GFM' # 'LFM'
        
        if 'refit_model' not in parameters.keys():
            parameters['refit_model'] = False # self.ml_params.get('refit_model')
        if 'refit_freq' not in parameters.keys():
            parameters['refit_freq'] = np.inf # int(self.ml_params.get('refit_freq') * self.data_params.get('periods'))

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
            parameters['selected_features'] = deepcopy(selected_features[parameters['method']][:50])

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
                dec_timeout=10*60
            )

            # Delete ml_pipeline from memory
            del ml_pipeline
            
            # Update self.lfm_models: Only update dev_models for new models
            self.update_models(
                new_candidate=model,
                debug=debug # debug
            )

            # Extract tuning_metric from val_table
            if model.val_table.is_dummy:
                tuning_metric = -np.inf
            else:
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

    def update_models(
        self,
        new_candidate: Model,
        debug: bool = False
    ) -> None:
        if new_candidate.val_table.tuning_metric > 0:
            if new_candidate.model_id not in [m.model_id for m in self.models]:
                if (
                    len(self.models) < self.n_candidates
                    or new_candidate.val_table.tuning_metric > self.models[-1].val_table.tuning_metric
                ):
                    # Add new_candidate to self.models
                    self.models.append(new_candidate)
                
                    # Drop duplicate Models (keeping most performant)
                    self.models = self.model_registry.drop_duplicate_models(
                        models=self.models,
                        from_=None,
                        trading_metric=False,
                        by_table='val',
                        debug=debug
                    )
                
                    # Sort Models
                    self.models = self.model_registry.sort_models(
                        models=self.models,
                        trading_metric=False,
                        by_table='val'
                    )

                    if len(self.models) > self.n_candidates:
                        self.models = self.models[:self.n_candidates]
                    
                    if new_candidate.model_id in [m.model_id for m in self.models]:
                        print(f'Model {new_candidate.model_id} ({new_candidate.stage} | {new_candidate.model_class}) was added to self.models.\n')
            else:
                LOGGER.warning(
                    '%s is already in lfm_models.\n'
                    'Note: This should only happen for evaluation of warm models.\n',
                    new_candidate.model_id
                )

    def evaluate_models(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        debug: bool = False
    ) -> None:
        # Evaluate self.models
        for idx in tqdm(range(len(self.models))):
            # Extract model
            model: Model = self.models[idx]

            if model.optimized_table is None or model.optimized_table.shape[0] == 0:
                if model.val_table.trading_metric > 0:
                    # Instanciate MLPipeline
                    ml_pipeline = MLPipeline(
                        pipeline_params=model.pipeline_params,
                        ml_params=self.ml_params,
                        trading_params=self.trading_params
                    )
                    
                    # Run build_pipeline
                    self.models[idx] = ml_pipeline.build_pipeline(
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

                    LOGGER.info(f'Evaluated Model optimized_table:')
                    self.models[idx].optimized_table.show_attrs(residuals_attrs=False)

                    # Save Model
                    self.models[idx].save()
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
        reduced_tuning_periods: int = None,
        debug: bool = False, 
        deep_debug: bool = False
    ) -> None:
        if deep_debug:
            debug = True

        """
        Prepare Models
        """
        # Filter broken models
        # filter_function = partial(
        #     self.model_filter,
        #     include_test_filter=False
        # )
        # self.models = list(filter(filter_function, self.models))

        # Sort models, based on val_table
        self.models = self.model_registry.sort_models(
            models=self.models,
            trading_metric=False,
            by_table='val'
        )

        if debug and len(self.models) > 0:
            print('Models Val Tuning Metrics:\n')
            for model in self.models:
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
        Tune Global Forecasting Models
        """
        # Define Max Performance Threshold
        loss_threshold = -25

        if debug and len(self.models) > 0:
            print(f'Expected starting point: {self.models[0].val_table.tuning_metric}.\n' 
                  f'loss_threshold: {loss_threshold}\n\n')
        
        # Define warm_start_params & trials
        warm_models = [model for model in self.models if model.algorithm in self.algorithms]
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
        
        print(f'\n\nTuning Global Forecasting Models:\n')

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

        # Save Models
        for model in self.models:
            model.save()

        """
        Evaluate Models
        """
        LOGGER.info("Evaluating Models:")
        self.evaluate_models(
            ml_datasets=ml_datasets,
            debug=deep_debug
        )

        # Re-Save Models
        for model in self.models:
            model.save()

        """
        Update Model Registry
        """
        # Add Dev Models to Registry
        self.model_registry.registry['development'] = [
            (m.model_id, m.model_class) for m in self.models
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

        # Populate self.models
        if len(reg_models) > 0:
            self.models: List[Model] = [m for m in reg_models if m is not None]
        else:
            self.models: List[Model] = []
        
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

