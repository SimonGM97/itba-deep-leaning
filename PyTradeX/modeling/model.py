from PyTradeX.config.params import Params
from PyTradeX.trading.trading_table import TradingTable
from PyTradeX.utils.others.timing import timing
from PyTradeX.utils.others.s3_helper import (
    write_to_s3, 
    load_from_s3,
    delete_from_s3,
    delete_s3_directory,
    S3_CLIENT
)
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.modeling.model_expectations import (
    find_model_diagnosis_dict,
    needs_repair
)

# Non-deep learning ML models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# from prophet import Prophet

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# from darts.models.forecasting.nbeats import NBEATSModel
# from darts import TimeSeries, concatenate
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Bidirectional, Conv1D
# from keras.callbacks import EarlyStopping
# from keras.optimizers import legacy
# from keras.preprocessing.sequence import TimeseriesGenerator
# from keras.optimizers import Adam
# from keras.models import load_model
# from torch.optim import Adam
# from torch.nn.modules.loss import MSELoss, L1Loss
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# MLFlow
import mlflow

# Others
import warnings
from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

import plotly.express as px
import pandas as pd
import numpy as np
import shap
import joblib
import tempfile
import io
import secrets
import string
import time
import signal
import os
from typing import Tuple, List, Dict
from pprint import pprint, pformat


# Specify libtbb.dylib file path & set up environ variables
path = '/Users/simongarciamorillo/anaconda3/lib/python3.10/site-packages/prophet/stan_model/cmdstan-2.31.0/stan/lib/stan_math/lib/tbb'
os.environ['DYLD_LIBRARY_PATH'] = path
os.environ['CUDA_VISIBLE_DEVICES'] = 'all'
# export PYTORCH_ENABLE_MPS_FALLBACK=1


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


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def stan_init(m):
    """
    Retrieve parameters from a trained model.

    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res


class Model:

    load_parquet = [
        'forecasts_df',
        'feature_importance_df'
    ]
    load_pickle = [
        # General Parameters
        'coin_name',
        'intervals',
        'lag',
        'train_coins',

        'algorithm',
        'method',
        'pca',
        'model_class',

        # Register Parameters
        'model_id',
        # 'file_name',
        # 'model_name',
        'version',
        'stage',

        # Model Parameters
        'hyper_parameters',
        'trading_parameters',
        'optimized_trading_parameters',

        'refit_model',
        'refit_freq',
        'reverse_forecasts',

        'selected_features',

        # Load Parameters
        'fitted',
        'max_seen_return',
        'min_seen_return',
        'bias',
        'forecast_multiplier',
        'last_fitting_date',
        'importance_method',
        'shap_values',

        'train_idx',
        'cum_ret_model',
    ]

    def __init__(
        self,
        model_id: str = None,
        version: str = '1.0',
        stage: str = 'Staging',

        coin_name: str = None,
        intervals: str = Params.general_params.get("intervals"),
        lag: int = Params.data_params.get("lag"),
        mbp: int = Params.data_params.get("mbp"),
        train_coins: list = None,
        
        algorithm: str = None,
        method: str = None,
        pca: bool = False,
        model_class: str = 'LFM',
        
        hyper_parameters: dict = None,
        trading_parameters: dict = None,
        optimized_trading_parameters: dict = None,

        refit_model: bool = True,
        refit_freq: int = None,
        reverse_forecasts: bool = False,
        timeout_time: float = None,

        selected_features: list = None,
        
        load_model: bool = False,
        debug: bool = False
    ) -> None:
        # General Parameters
        self.coin_name: str = coin_name
        self.intervals: str = intervals
        self.lag: int = lag
        self.mbp: int = mbp
        self.train_coins: List[str] = train_coins

        self.algorithm: str = algorithm
        self.method: str = method
        self.pca: bool = pca
        self.model_class: str = model_class

        # Register Parameters
        if model_id is not None:
            self.model_id: str = model_id
        else:
            self.model_id: str = ''.join(secrets.choice(string.ascii_letters) for i in range(10))
        
        self.file_name: str = self.find_file_name()
        self.model_name: str = self.find_model_name()
        self.version: str = version
        self.stage: str = stage

        # Features
        self.selected_features: List[str] = selected_features

        # Model Parameters
        self.hyper_parameters: dict = self.correct_hyper_parameters(hyper_parameters)
        self.trading_parameters: dict = trading_parameters
        self.optimized_trading_parameters: dict = optimized_trading_parameters

        self.refit_model: bool = refit_model
        self.refit_freq: int = refit_freq
        if algorithm == 'n_beats':
            self.refit_freq: int = 10 * self.refit_freq

        self.reverse_forecasts: bool = reverse_forecasts
        
        if timeout_time is None:
            self.timeout_time: float = np.inf
        else:
            self.timeout_time: float = timeout_time
        
        # Load Parameters
        self.model = None

        self.fitted: bool = False
        self.max_seen_return: float = None
        self.min_seen_return: float = None

        self.bias: float = None
        self.forecast_multiplier: float = None
        self.last_fitting_date: pd.DatetimeIndex = None

        self.forecasts_df: pd.DataFrame = pd.DataFrame(columns=['return_forecast'])
        self.importance_method: str = None
        self.feature_importance_df: pd.DataFrame = pd.DataFrame(columns=['feature', 'importance'])
        self.shap_values: np.ndarray = None

        self.val_table: TradingTable = None
        self.test_table: TradingTable = None
        self.optimized_table: TradingTable = None

        self.train_idx: pd.DatetimeIndex = None
        self.cum_ret_model: Dict[LinearRegression, float] = None

        # Define base_path
        self.s3_base_path = f"{Params.bucket}/modeling/models/{self.intervals}/{self.model_id}"
        
        if load_model:
            self.load(debug=debug)
    
    @property
    def trading_table_input(self) -> dict:
        return {
            'model_id': self.model_id,
            'coin_name': self.coin_name,
            'intervals': self.intervals,

            'algorithm': self.algorithm,
            'method': self.method,
            'pca': self.pca,
        }
    
    @property
    def warm_start_params(self) -> dict:
        follow_flow = self.trading_parameters.get('follow_flow')
        certainty_threshold = self.trading_parameters.get('certainty_threshold')
        tp = self.trading_parameters.get('tp')
        sl = self.trading_parameters.get('sl')
        algorithms: list = Params.ml_params.get('algorithms')

        params = {
            # General Parameters
            'coin_name': self.coin_name,
            'intervals': self.intervals,
            'lag': self.lag,

            'algorithm': self.algorithm,
            'method': self.method,
            'pca': self.pca,
            'model_class': self.model_class,

            # Register Parameters            
            'model_id': self.model_id,
            'version': self.version,
            'stage': self.stage,

            # Trading Parameters
            'follow_flow': follow_flow,
            'certainty_threshold': certainty_threshold,
            'tp': False if tp == (None, None) else True,
            'sl': False if sl == (None, None) else True,

            # Others
            'refit_model': self.refit_model,
            'refit_freq': self.refit_freq,
            'reverse_forecasts': self.reverse_forecasts,
            'model_type': algorithms.index(self.algorithm)
        }

        # Hyper-Parameters
        params.update(**{
            f'{self.algorithm}.{k}': v for k, v in self.hyper_parameters.items()
        })
        if self.algorithm == 'lightgbm' and 'lightgbm.objective' not in params.keys():
            LOGGER.warning(
                'LGBMModel is missing "objective" hyper parameters.\n'
                'Thus, "regression" objective will be added.\n'
            )
            
        params.update(**{
            'optimized_trading_parameters': self.optimized_trading_parameters
        })

        return params

    @property
    def pipeline_params(self) -> dict:
        return {
            'model_id': self.model_id,
            'version': self.version,
            'stage': self.stage,

            'coin_name': self.coin_name,
            'intervals': self.intervals,
            'lag': self.lag,
            'train_coins': self.train_coins,

            'algorithm': self.algorithm,
            'method': self.method,
            'pca': self.pca,
            'model_class': self.model_class,
            
            'hyper_parameters': self.hyper_parameters,
            'trading_parameters': self.trading_parameters,
            'optimized_trading_parameters': self.optimized_trading_parameters,

            'refit_model': self.refit_model,
            'refit_freq': self.refit_freq,
            'reverse_forecasts': self.reverse_forecasts,

            'selected_features': self.selected_features
        }

    def find_file_name(self) -> str:
        if self.algorithm == 'naive_lv':
            return f"{self.model_id}_naive_lv_model"
        
        if self.algorithm == 'naive_ma':
            return f"{self.model_id}_naive_ma_model"
        
        if self.algorithm == 'expo_smooth':
            return f"{self.model_id}_exponential_smoothing_model.pickle"

        elif self.algorithm == 'sarimax':
            return f"{self.model_id}_sarimax_model.pickle"

        elif self.algorithm == 'random_forest':
            return f"{self.model_id}_random_forest_model.pickle"

        elif self.algorithm == 'lightgbm':
            return f"{self.model_id}_lightgbm_model.pickle"
        
        elif self.algorithm == 'xgboost':
            return f"{self.model_id}_xgboost_model.pickle"

        elif self.algorithm == 'prophet':
            return f"{self.model_id}_prophet_model.pickle"

        elif self.algorithm == 'orbit':
            LOGGER.warning('Unable to save "orbit" model.')
            return ''

        elif self.algorithm == 'lstm':
            return f"{self.model_id}_lstm_model.h5"

        elif self.algorithm == 'n_beats':
            return f"{self.model_id}_n_beats_model.ckpt"
        
        elif self.algorithm == 'tft':
            return f"{self.model_id}_tft_model.ckpt"
        
    def find_model_name(self) -> str:
        return f"{self.model_id}_{self.coin_name}_{self.method}_{self.algorithm}_{self.intervals}_model"
        
    @staticmethod
    def custom_objective(y_true, y_pred) -> tuple:
        def custom_loss(y_true, y_pred):
            y_true_int = (y_true > 0).astype(int)
            y_pred_int = (y_pred > 0).astype(int)

            mae = mean_absolute_error(y_true, y_pred)
            acc = accuracy_score(y_true_int, y_pred_int)

            return mae * (1 - acc)
    
        grad = (y_pred - y_true) / y_true
        hess = np.ones_like(y_true)
        loss = custom_loss(y_true, y_pred)
        return grad, hess, loss

    def correct_hyper_parameters(
        self,
        hyper_parameters: dict,
        debug: bool = False
    ) -> dict:
        if self.algorithm == 'naive_lv':
            pass

        elif self.algorithm == 'naive_ma':
            pass

        elif self.algorithm == 'expo_smooth':
            # if not hyper_parameters['seasonal']:
            #     hyper_parameters['seasonal_periods'] = None
            
            if hyper_parameters['trend'] is None:
                hyper_parameters['damped_trend'] = False

            # hyper_parameters['freq'] = 'MS'
            hyper_parameters['trend'] = hyper_parameters.pop('trend')

        elif self.algorithm == 'sarimax':
            hyper_parameters['d'] = 0
            hyper_parameters['seasonal_D'] = 0

            if hyper_parameters['seasonal_P'] == hyper_parameters['seasonal_Q'] == 0:
                hyper_parameters['seasonal_S'] = 0
            if hyper_parameters['seasonal_P'] > 0 or hyper_parameters['seasonal_Q'] > 0:
                if hyper_parameters['seasonal_S'] == 0:
                    hyper_parameters['seasonal_S'] = 6
            if hyper_parameters['q'] == hyper_parameters['seasonal_Q']:
                if hyper_parameters['q'] > 0:
                    hyper_parameters['q'] += 1
            if hyper_parameters['p'] == hyper_parameters['seasonal_P']:
                hyper_parameters['p'] += 1

            order = (
                hyper_parameters['p'], 
                hyper_parameters['d'], 
                hyper_parameters['q']
            )
            seasonal_order = (
                hyper_parameters['seasonal_P'], 
                hyper_parameters['d'],
                hyper_parameters['seasonal_Q'], 
                hyper_parameters['seasonal_S']
            )

            hyper_parameters = {
                'order': order,  # (p, d, q)
                'seasonal_order': seasonal_order,  # (P, D, Q, S)
                'trend': hyper_parameters['sarimax.trend'], 
                'measurement_error': False,
                'time_varying_regression': False, 
                'mle_regression': True,
                'simple_differencing': False,
                'enforce_stationarity': False, 
                'enforce_invertibility': False,
                'hamilton_representation': False, 
                'concentrate_scale': False,
                'trend_offset': 1, 
                'use_exact_diffuse': False, 
                'dates': None
            }

        elif self.algorithm == 'random_forest':
            hyper_parameters.update(**{
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 23111997
            })

        elif self.algorithm == 'lightgbm':
            hyper_parameters.update(**{
                "importance_type": 'gain',
                'verbose': -1,
                "random_state": 23111997,
                "n_jobs": -1
            })

        elif self.algorithm == 'xgboost':
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            hyper_parameters.update(**{
                "verbosity": 0,
                "use_rmm": True,
                "device": 'cuda', # 'cpu', 'cuda' # cuda -> GPU
                "nthread": -1,
                "n_gpus": -1,
                "max_delta_step": 0,
                "gamma": 0,
                "subsample": 1, # hp.uniform('xgboost.subsample', 0.6, 1)
                "sampling_method": 'uniform',
                "random_state": 23111997,
                "n_jobs": -1
            })

            if hyper_parameters['objective'] == 'reg:quantileerror':
                hyper_parameters['quantile_alpha'] = 0.5

        elif self.algorithm == 'prophet':
            pass

        elif self.algorithm == 'orbit':
            pass

        elif self.algorithm == 'lstm':
            pass
        
        """
        elif self.algorithm == 'n_beats':
            # Loss & Early Stopping
            loss = hyper_parameters.pop('loss')
            if loss == 'MSE':
                hyper_parameters['loss_fn'] = MSELoss()
                min_delta = hyper_parameters['learning_rate'] * 0.5
            elif loss == 'MAE':
                hyper_parameters['loss_fn'] = L1Loss()
                min_delta = 0.0001

            my_stopper = EarlyStopping(
                monitor="val_loss", # "val_loss", "train_loss"
                patience=5, # 10, 5
                min_delta=min_delta,
                mode='min'
            )
            
            hyper_parameters.update({
                'input_chunk_length': len(self.selected_features),
                'output_chunk_length': 1,
                'generic_architecture': True,
                'expansion_coefficient_dim': 5,
                'trend_polynomial_degree': 2,
                'activation': 'ReLU',
                'likelihood': None,
                'optimizer_cls': Adam,
                'lr_scheduler_cls': None,
                'lr_scheduler_kwargs': None,
                'model_name': self.model_name,
                'log_tensorboard': False,
                'nr_epochs_val_period': 1,
                'force_reset': False,
                'save_checkpoints': False,
                'random_state': 23111997,
                'pl_trainer_kwargs': {
                    "accelerator": "gpu", # "gpu", "cpu"
                    "devices": -1, # -1, Params.cpus
                    # "auto_select_gpus": True,
                    "callbacks": [my_stopper]
                }
            })

            # Learning Rate
            learning_rate = hyper_parameters.pop('learning_rate')
            hyper_parameters['optimizer_kwargs'] = {"lr": learning_rate}

            # Epochs
            if "n_epochs" not in hyper_parameters:
                hyper_parameters['n_epochs'] = 20

        elif self.algorithm == 'tft':
            # Loss & Early Stopping
            loss = hyper_parameters.pop('loss')
            if loss == 'MSE':
                hyper_parameters['loss_fn'] = MSELoss()
                min_delta = hyper_parameters['learning_rate'] * 0.5
            elif loss == 'MAE':
                hyper_parameters['loss_fn'] = L1Loss()
                min_delta = 0.0001
            
            my_stopper = EarlyStopping(
                monitor="val_loss", # "val_loss", "train_loss"
                patience=5, # 10, 5
                min_delta=min_delta,
                mode='min'
            )
            
            hyper_parameters.update({
                'input_chunk_length': len(self.selected_features),
                'output_chunk_length': 1,
                'full_attention': False,
                'feed_forward': 'GatedResidualNetwork',
                'categorical_embedding_sizes': None,
                'add_relative_index': False,
                'likelihood': None,
                'norm_type': 'LayerNorm',
                'use_static_covariates': True,

                'generic_architecture': True,
                'expansion_coefficient_dim': 5,
                'trend_polynomial_degree': 2,
                'activation': 'ReLU',
                'likelihood': None,
                'optimizer_cls': Adam,
                'lr_scheduler_cls': None,
                'lr_scheduler_kwargs': None,
                'model_name': self.model_name,
                'log_tensorboard': False,
                'nr_epochs_val_period': 1,
                'force_reset': False,
                'save_checkpoints': False,
                'random_state': 23111997,
                'pl_trainer_kwargs': {
                    "accelerator": "gpu", # "gpu", "cpu"
                    "devices": -1, # -1, Params.cpus
                    # "auto_select_gpus": True,
                    "callbacks": [my_stopper]
                }
            })

            # Learning Rate
            learning_rate = hyper_parameters.pop('learning_rate')
            hyper_parameters['optimizer_kwargs'] = {"lr": learning_rate}

            # Epochs
            if "n_epochs" not in hyper_parameters:
                hyper_parameters['n_epochs'] = 20
        """

        if debug and hyper_parameters is not None:
            print("hyper_parameters:\n"
                  "{")
            for key in hyper_parameters.keys():
                print(f"    '{key}': {hyper_parameters[key]} ({type(hyper_parameters[key])})")
            print('}\n\n')

        return hyper_parameters

    def prepare_lstm_datasets(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expected input_shape: [samples, time_steps, features]
            - samples: This dimension represents the number of sequences or samples in 
                your dataset. Each sample is essentially a sequence of data points.
            - time_steps: This dimension represents the number of time steps in each sequence. 
              It signifies how far back in time the LSTM should look to make predictions. 
            - features: This dimension represents the number of features or variables you have in each time step. 
        """
        # generator = TimeseriesGenerator(
        #     X, y[[f'target_{self.method}']], 
        #     length=self.hyper_parameters['sequence_length'], 
        #     batch_size=1 # self.hyper_parameters['batch_size']
        # )

        # Calculate the number of sequences
        sequence_length = self.hyper_parameters['sequence_length']
        num_sequences = X.shape[0] - sequence_length

        # Reshape X into sequences
        X = np.array(X)
        X_sequences = np.array([X[i:i+sequence_length] for i in range(num_sequences)])

        # Extract y_train for corresponding sequences
        if y is not None:
            y_sequences = np.array(y[f'target_{self.method}'])[sequence_length:].reshape(num_sequences, 1, 1)
        else:
            y_sequences = None

        if debug:
            if y is None:
                y_shape = None
                y_sequences_shape = None
            else:
                y_shape = y.shape
                y_sequences_shape = y_sequences.shape

            print(f'y.shape: {y_shape}\n'
                  f'X.shape: {X.shape}\n'
                  f'sequence_length: {sequence_length}\n'
                  f'num_sequences: {num_sequences}\n'
                  f'y_sequences.shape: {y_sequences_shape}\n'
                  f'X_sequences.shape: {X_sequences.shape}\n\n')

        return X_sequences, y_sequences

    def build(
        self,
        train_target: pd.DataFrame = None, 
        train_features: pd.DataFrame = None,
        debug: bool = False
    ) -> None:
        if self.algorithm == 'naive_lv':
            self.model = self.hyper_parameters['lv']

        elif self.algorithm == 'naive_ma':
            self.model = self.hyper_parameters['period'], self.hyper_parameters['weight_type']
        
        elif self.algorithm == 'expo_smooth':
            """
            Parameters:
                alpha (float): Smoothing coefficient for the level component.
                beta (float): Smoothing coefficient for the trend component.
                gamma (float): Smoothing coefficient for the seasonal component.
                phi (float): Coefficient for the damped trend.
                periods (int): Length of the seasonality cycle, if required.
            """
            target_column = f'target_{self.method}'

            self.model = ExponentialSmoothing(
                train_target[target_column],
                trend=self.hyper_parameters['expo_smooth.trend'], 
                damped_trend=self.hyper_parameters['damped_trend'], 
                seasonal=self.hyper_parameters['seasonal'], 
                seasonal_periods=self.hyper_parameters['seasonal_periods']
            )
        
        elif self.algorithm == 'sarimax':
            if self.model is not None:
                self.hyper_parameters['start_ar_lags'] = self.model.ar_lags
                self.hyper_parameters['start_ma_lags'] = self.model.ma_lags
            
            self.model = SARIMAX(
                endog=train_target.values.astype(float),
                exog=train_features.values.astype(float),
                **self.hyper_parameters
            )
        
        elif self.algorithm == 'random_forest':
            self.model = RandomForestRegressor(**self.hyper_parameters)
        
        elif self.algorithm == 'lightgbm':
            self.model = LGBMRegressor(**self.hyper_parameters)

        elif self.algorithm == 'xgboost':
            self.model = XGBRegressor(**self.hyper_parameters)
        
        else:
            LOGGER.critical('Invalid algorithm: %s!', self.algorithm)
            raise Exception(f'Invalid algorithm: {self.algorithm}!')
        
        """
        elif self.algorithm == 'prophet':
            ignore_parameters = ['holidays', 'monthly_seasonality', 'weekly_seasonality']
            filtered_params = {k: v for k, v in self.hyper_parameters.items() if k not in ignore_parameters}
            self.model = Prophet(**filtered_params)

            if self.hyper_parameters['holidays']:
                self.model.add_country_holidays(country_name='US')

            if self.hyper_parameters['monthly_seasonality']:
                self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.1)

            if self.hyper_parameters['weekly_seasonality']:
                self.model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)

            for regressor in train_features.columns.tolist():
                # print(f"adding regressor: {regressor}")
                self.model.add_regressor(regressor)  # mode='multiplicative')

        elif self.algorithm == 'orbit':
            train_df = pd.concat([train_target, train_features], axis=1).reset_index()
            test_df = pd.concat([test_target, test_features], axis=1).reset_index()
            
            # "orbit_model": hp.choice('orbit_model', ['DLT', 'ETS', 'LGT', 'KTR'])
            if self.hyper_parameters['orbit_model'] == 'DLT':
                self.model = ETS(
                    response_col='ventas',
                    date_col='index',
                    estimator='stan-map',
                    seasonality=self.hyper_parameters['seasonality'],
                    seed=23111997,
                )
            elif self.hyper_parameters['orbit_model'] == 'LGT':
                self.model = LGT(
                    response_col='ventas',
                    date_col='index',
                    estimator='stan-map',
                    # estimator='pyro-svi',
                    seasonality=self.hyper_parameters['seasonality'],
                    seed=23111997,
                )
            pass

        elif self.algorithm == 'lstm':
            time_steps = self.hyper_parameters['sequence_length']
            features = train_features.shape[1]

            # with self.strategy.scope():
            # Define a Sequential model
            self.model = Sequential()
        
            # Add first layer
            self.model.add(LSTM(
                self.hyper_parameters['units'],
                input_shape=(time_steps, features), # (time steps, cols)
                dropout=self.hyper_parameters['dropout'], 
                recurrent_dropout=self.hyper_parameters['recurrent_dropout'],
                return_sequences=True
            ))
            
            # Add subsequent LSTM layers
            if self.hyper_parameters['layers'] > 1:
                mult_list = (np.geomspace(1, 2, self.hyper_parameters['layers']+1)[::-1] - 1)[:-1]
                for mult in mult_list[1:]:
                    # print(int(self.hyper_parameters['units'] * mult))
                    self.model.add(LSTM(
                        int(self.hyper_parameters['units'] * mult), 
                        dropout=self.hyper_parameters['dropout'], 
                        recurrent_dropout=self.hyper_parameters['recurrent_dropout'],
                        return_sequences=True
                    ))
            
            # Add Dense layer and compile
            self.model.add(Dense(1))
            
            # Describe Optimizer
            optimizer = legacy.Adam(
                learning_rate=self.hyper_parameters['learning_rate']
            )
            
            self.model.compile(
                loss="mae", # "mae", "mse"
                optimizer=optimizer
            )
        
        elif self.algorithm == 'n_beats':
            self.model = NBEATSModel(**self.hyper_parameters)

        elif self.algorithm == 'tft':
            pass
        """
        
        # Set self.fitted to False
        self.fitted = False
        
        if debug:
            print(f'self.model: {self.model}\n')

    def fit(
        self,
        train_target: pd.DataFrame = None, 
        val_target: pd.DataFrame = None, 
        train_features: pd.DataFrame = None,
        val_features: pd.DataFrame = None,
        find_forecast_multiplier: bool = True,
        debug: bool = False
    ) -> None:        
        if self.model_class == 'GFM':
            LOGGER.info(
                "Fitting %s GFM model.\n"
                "train_target.shape: %s\n"
                "val_target.shape: %s\n"
                "train_features.shape: %s\n"
                "val_features.shape: %s\n",
                self.model_id, train_target.shape, val_target.shape,
                train_features.shape, val_features.shape
            )

        # else:
        #     print(f'Fitting {self.model_id} LFM model.\n')

        # Define target_column
        target_column = f'target_{self.method}'

        # Fit self.model
        if self.algorithm == 'naive_lv':
            pass

        elif self.algorithm == 'naive_ma':
            pass

        elif self.algorithm == 'expo_smooth':
            self.model.fit()

        elif self.algorithm == 'sarimax':
            self.model = self.model.fit(
                disp=False, 
                maxiter=50
            )

        elif self.algorithm == 'random_forest':
            self.model.fit(
                train_features.values.astype(float), 
                train_target[target_column].values.astype(float)
            )

        elif self.algorithm == 'lightgbm':
            self.model.fit(
                train_features.values.astype(float), 
                train_target[target_column].values.astype(float)
            )

        elif self.algorithm == 'xgboost':
            self.model.fit(
                train_features.values.astype(float), 
                train_target[target_column].values.astype(float)
            )

        else:
            LOGGER.critical('Invalid algorithm: %s!', self.algorithm)
            raise Exception(f'Invalid algorithm: {self.algorithm}!')
        
        """
        elif self.algorithm == 'prophet':
            # Build Train and Future DataFrames
            train_df = pd.concat(
                [train_target[[target_column]].rename(columns={target_column: 'y'}),
                 train_features], axis=1
            ).reset_index().rename(columns={'index': 'ds'})

            train_df['floor'] = 0
            cap = np.max(train_target[target_column].values) * 1.1
            if cap > 0:
                train_df['cap'] = cap
            else:
                self.hyper_parameters['growth'] = 'flat'

            if debug:
                print(f'prophet train_d.tail(20): \n{train_df.tail(20)}\n\n')

            with suppress_stdout_stderr():
                if self.fitted:
                    self.model.fit(train_df, init=stan_init(self.model))
                else:
                    self.model.fit(train_df)

        elif self.algorithm == 'orbit':
            # model.fit(df=train_df)
            # print(f'orbit predicted_df: {model.predict(df=test_df)}')
            pass
        
        elif self.algorithm == 'lstm':
            # Prepare datasets
            X_sequences, y_sequences = self.prepare_lstm_datasets(
                X=train_features,
                y=train_target,
                debug=True # debug
            )

            # with self.strategy.scope():
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=10
            )
            
            history = self.model.fit(
                X_sequences, 
                y_sequences,
                epochs=self.hyper_parameters['epochs'],
                batch_size=self.hyper_parameters['batch_size'],
                verbose=1, # 0,
                callbacks=[early_stopping],
                workers=-1,
                use_multiprocessing=True
            )

            loss = history.history["loss"][-1]

        elif self.algorithm == 'n_beats':
            ts_univ_y_train = TimeSeries.from_dataframe(
                train_target, 
                value_cols=target_column,
                fill_missing_dates=True,
                freq='30T'
            ).astype(np.float32)
            ts_cov_X_train = TimeSeries.from_dataframe(
                train_features,
                fill_missing_dates=True,
                freq='30T'
            ).astype(np.float32)

            input_len = self.hyper_parameters.get('input_chunk_length')
            
            ts_univ_y_val = TimeSeries.from_dataframe(
                # pd.concat([train_target.iloc[-input_len:], val_target.iloc[:-1]]), 
                val_target,
                value_cols=target_column,
                fill_missing_dates=True,
                freq='30T'
            ).astype(np.float32)
            ts_cov_X_val = TimeSeries.from_dataframe(
                # pd.concat([train_features.iloc[-input_len:], val_features.iloc[:-1]]),
                val_features,
                fill_missing_dates=True,
                freq='30T'
            ).astype(np.float32)
            
            self.model.fit(
                series=ts_univ_y_train,
                past_covariates=ts_cov_X_train,
                val_series=ts_univ_y_val,
                val_past_covariates=ts_cov_X_val,
                # num_loader_workers=-1,
                verbose=True # False
            )

        elif self.algorithm == 'tft':
            pass
        """
        
        # Set Forecat Multiplier
        if find_forecast_multiplier:
            self.find_forecast_multiplier(
                train_target=train_target,
                train_features=train_features,
                val_target=val_target,
                val_features=val_features,
                debug=debug
            )
        else:
            LOGGER.warning(
                'find_forecast_multiplier was set to False in %s (%s | %s - %s).',
                self.model_id, self.stage, self.model_class, self.intervals
            )
            
            self.forecast_multiplier = 1
        
        # Update Version
        if not self.fitted:
            self.fitted = True
            self.version = '1.0'
        else:
            v_numbers = self.version.split('.')
            self.version = f"{int(v_numbers[0]) + 1}.{v_numbers[1]}"

        if self.model_class == 'GFM':
            LOGGER.info(
                "New %s (%s | %s - %s) version: %s.",
                self.model_id, self.stage, self.model_class, self.intervals, self.version
            )
        
        # Update Train IDX
        if self.train_idx is None:
            self.train_idx = list(train_target.index)
        else:
            self.train_idx = list(set(self.train_idx + list(train_target.index)))
        self.train_idx.sort()

        # Update self.max_seen_return & self.min_seen_return
        self.max_seen_return = train_target['target_return'].max()
        self.min_seen_return = train_target['target_return'].min()

        # Update Last Fitting Date
        self.last_fitting_date = train_target.index[-1]

        if debug:
            print(f'self.model: {self.model}\n'
                  f'hyper_parameters: \n')
            print(self.hyper_parameters)
            print(f'\n\n'
                  f'train_target: \n{train_target[target_column].tail(20)}\n\n'
                  f'forecast_multiplier: {self.forecast_multiplier}\n'
                  f'self.last_fitting_date: {self.last_fitting_date}\n\n')
            
    def find_forecast_multiplier(
        self, 
        train_target: pd.DataFrame = None, 
        val_target: pd.DataFrame = None, 
        train_features: pd.DataFrame = None,
        val_features: pd.DataFrame = None,
        debug: bool = False
    ) -> None:
        # Define Target Column
        target_column = f'target_{self.method}'

        if target_column == 'target_price':
            self.forecast_multiplier = 1
            return
        
        # Find Mean Average Target
        full_target = pd.concat([train_target, val_target])
        mat = np.mean(np.abs(full_target[target_column]))

        if debug:
            print(f'mat: {mat}\n')

        if self.algorithm == 'naive_lv':
            pass

        elif self.algorithm == 'naive_ma':
            if self.hyper_parameters['weight_type'] == 'normal':
                weights = np.array([1] * self.hyper_parameters['period'])
            else:
                weights = np.array([1 - p / self.hyper_parameters['period'] for p in range(self.hyper_parameters['period'])])

            predictions = full_target[target_column].shift(1).rolling(self.hyper_parameters['period']).apply(
                lambda x: np.sum(weights * x) / np.sum(weights)
            ).bfill().values

            map = np.mean(np.abs(predictions))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'expo_smooth':
            train_predictions = self.model.fittedvalues

            map = np.mean(np.abs(train_predictions))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'sarimax':
            train_predictions = self.model.fittedvalues

            map = np.mean(np.abs(train_predictions))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'random_forest':
            train_predictions = self.model.predict(train_features)

            map = np.mean(np.abs(train_predictions))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'lightgbm':
            train_predictions = self.model.predict(train_features)

            map = np.mean(np.abs(train_predictions))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'xgboost':
            train_predictions = self.model.predict(train_features)

            map = np.mean(np.abs(train_predictions))
            self.forecast_multiplier = mat / map

        else:
            LOGGER.critical('Invalid algorithm: %s!', self.algorithm)
            raise Exception(f'Invalid algorithm: {self.algorithm}!')
        
        """
        elif self.algorithm == 'prophet':
            # Build Train and Future DataFrames
            train_df = pd.concat(
                [train_target[[target_column]].rename(columns={target_column: 'y'}),
                 train_features], axis=1
            ).reset_index().rename(columns={'index': 'ds'})

            train_df['floor'] = 0
            cap = np.max(train_target[target_column].values) * 1.1
            if cap > 0:
                train_df['cap'] = cap
            else:
                self.hyper_parameters['growth'] = 'flat'

            if debug:
                print(f'prophet train_d.tail(20): \n{train_df.tail(20)}\n\n')

            train_predictions = self.model.predict(train_df)['yhat'].values

            map = np.mean(np.abs(train_predictions))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'orbit':
            # model.fit(df=train_df)
            # print(f'orbit predicted_df: {model.predict(df=test_df)}')
            pass
        
        elif self.algorithm == 'lstm':
            pass

        elif self.algorithm == 'n_beats':
            self.forecast_multiplier = 1

            val_forecasts = self.return_forecast(
                train_target=train_target,
                forecast_target=val_target.iloc[:300],
                train_features=train_features,
                forecast_features=val_features.iloc[:300],
                forecast_dates=val_target.index[:300],
                add_bias=None,
                steps=1,
                ignore_update=True,
                max_t=np.inf,
                raw_forecasts=True,
                debug=debug
            )[f'{self.method}_forecast']

            map = np.mean(np.abs(val_forecasts))
            self.forecast_multiplier = mat / map

        elif self.algorithm == 'tft':
            pass
        """
        
        if self.model_class == 'GFM':
            LOGGER.info(
                "Model %s (%s | %s - %s) forecast_multiplier was updated.",
                self.model_id, self.stage, self.model_class, self.intervals
            )
            debug = True
        
        if debug:
            print(f'self.forecast_multiplier: {self.forecast_multiplier}\n')

    def _update(
        self,
        train_target: pd.DataFrame = None, 
        val_target: pd.DataFrame = None,
        train_features: pd.DataFrame = None, 
        val_features: pd.DataFrame = None,
        dates=None,
        debug: bool = False
    ) -> None:
        target_column = f'target_{self.method}'

        if self.refit_model and self.needs_refitting(
            refit_date=dates[-1],
        ):
            if debug:
                LOGGER.debug("Model %s will re-fitted.", self.model_id)

            self.fit(
                train_target=train_target,
                val_target=val_target,
                train_features=train_features,
                val_features=val_features,
                debug=debug
            )
        else:
            if self.algorithm == 'naive_lv':
                pass

            elif self.algorithm == 'naive_ma':
                pass

            elif self.algorithm == 'expo_smooth':
                self.model = self.model.append(train_target[target_column].loc[dates[0]:dates[-1]])

            elif self.algorithm == 'sarimax':
                self.model = self.model.append(
                    endog=train_target[target_column].loc[dates[0]:dates[-1]],
                    exog=train_features.loc[dates[0]:dates[-1]],
                )

            elif self.algorithm == 'random_forest':
                pass

            elif self.algorithm == 'lightgbm':
                pass

            elif self.algorithm == 'xgboost':
                pass            
            
            else:
                LOGGER.critical('Invalid algorithm: %s!', self.algorithm)
                raise Exception(f'Invalid algorithm: {self.algorithm}!')
            
            """
            elif self.algorithm == 'prophet':
                # Build Train and Future DataFrames
                train_df = pd.concat(
                    [train_target[[target_column]].rename(columns={target_column: 'y'}),
                     train_features], axis=1
                ).reset_index().rename(columns={'index': 'ds'})

                train_df['floor'] = 0
                cap = np.max(train_target[target_column].values) * 1.1
                if cap > 0:
                    train_df['cap'] = cap
                else:
                    self.hyper_parameters['growth'] = 'flat'

                warm_parameters = stan_init(self.model)

                ignore_parameters = ['holidays', 'monthly_seasonality', 'weekly_seasonality']
                filtered_params = {k: v for k, v in self.hyper_parameters.items() if k not in ignore_parameters}
                self.model = Prophet(**filtered_params)

                if self.hyper_parameters['holidays']:
                    self.model.add_country_holidays(country_name='US')

                if self.hyper_parameters['monthly_seasonality']:
                    self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.1)

                if self.hyper_parameters['weekly_seasonality']:
                    self.model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)

                for regressor in train_features.columns.tolist():
                    self.model.add_regressor(regressor)  # mode='multiplicative')

                with suppress_stdout_stderr():
                    self.model.fit(train_df, init=warm_parameters)

            elif self.algorithm == 'orbit':
                pass

            elif self.algorithm == 'n_beats':
                pass

            elif self.algorithm == 'lstm': 
                pass
            """

    def dummy_forecast(
        self,
        train_target: pd.DataFrame = None,
        train_features: pd.DataFrame = None,
        forecast_features: pd.DataFrame = None, 
        forecast_target: pd.DataFrame = None,
        forecast_periods: int = 1, 
        debug: bool = False
    ) -> np.ndarray:
        if forecast_target is not None:
            forecast_periods = forecast_target.shape[0]

        target_column = f'target_{self.method}'

        # Prepare full_target
        if self.algorithm in ['naive_lv', 'naive_ma']:
            if forecast_target is None:
                forecast_target = pd.DataFrame(
                    [train_target[[target_column]].iat[-1, 0]] * forecast_periods, 
                    columns=[target_column]
                )

            full_target = pd.concat([train_target, forecast_target], axis=0)

            if debug:
                print(f'forecast_target: \n{forecast_target.tail(20)}\n\n'
                      f'full_target: \n{full_target.tail(20)}\n\n')

        if self.algorithm == 'naive_lv':
            forecast = (
                full_target
                .iloc[-(forecast_periods+self.model+10):]
                .filter(items=[target_column])
                .shift(self.model)
                .bfill()
                .iloc[-forecast_periods:]
                .values
            )

        elif self.algorithm == 'naive_ma':
            period, weight_type = self.model

            if weight_type == 'normal':
                weights = np.array([1] * period)
            else:
                weights = np.array([1 - p / period for p in range(period)])

            forecast = (
                full_target
                .iloc[-(forecast_periods+self.model[0]+10):]
                .filter(items=[target_column])
                .shift(1)
                .rolling(period)
                .apply(lambda x: np.sum(weights * x) / np.sum(weights))
                .bfill()
                .iloc[-forecast_periods:]
                .values
            )

        elif self.algorithm == 'expo_smooth':
            forecast = self.model.forecast(forecast_periods).values

        elif self.algorithm == 'sarimax':
            forecast = self.model.forecast(
                steps=forecast_features.shape[0],
                exog=forecast_features.values
            )

        elif self.algorithm == 'random_forest':
            forecast =  self.model.predict(
                forecast_features.values
            )

        elif self.algorithm == 'lightgbm':
            forecast = self.model.predict(
                forecast_features.values
            )

        elif self.algorithm == 'xgboost':
            forecast = self.model.predict(
                forecast_features.values
            )

        else:
            LOGGER.critical('Invalid algorithm: %s!', self.algorithm)
            raise Exception(f'Invalid algorithm: {self.algorithm}!')
        
        """
        elif self.algorithm == 'prophet':
            test_df = forecast_features.reset_index().rename(columns={'index': 'ds'})
            test_df['floor'] = 0
            cap = np.max(train_target[target_column].values) * 1.1
            if cap > 0:
                test_df['cap'] = cap

            if debug:
                print(f'prophet test_df.head(): \n{test_df.head(20)}\n\n')

            forecast = self.model.predict(test_df)['yhat'].values
            
        elif self.algorithm == 'orbit':
            # test_df = pd.concat([test_target, test_features], axis=1).reset_index()
            #
            # forecast = model.predict(df=test_df)
            # print(f'orbit predicted_df: {forecast}\n\n')
            forecast = None

        elif self.algorithm == 'lstm':
            # Prepare datasets
            X_fcst, _ = self.prepare_lstm_datasets(
                X=forecast_features,
                y=None,
                debug=True # debug
            )

            forecast = self.model.predict(
                x=X_fcst,
                batch_size=self.hyper_parameters['batch_size'],
                verbose=1, # 0,
                workers=-1,
                use_multiprocessing=True
            ) # .reshape(X_fcst.shape[0], 1)
            print(forecast.shape)
            # print(forecast)
            raise Exception('arreloco')

        elif self.algorithm == 'n_beats':
            input_chunk_length = self.hyper_parameters.get('input_chunk_length')

            ts_univ_y_train = TimeSeries.from_dataframe(
                train_target.iloc[-input_chunk_length-1:], 
                value_cols=target_column,
                fill_missing_dates=True
            ).astype(np.float32)
            ts_cov_X_fcst = TimeSeries.from_dataframe(
                pd.concat([train_features.iloc[-input_chunk_length-1:], 
                           forecast_features]),
                fill_missing_dates=True
            ).astype(np.float32)

            forecast = self.model.predict(
                n=forecast_periods,
                series=ts_univ_y_train,
                past_covariates=ts_cov_X_fcst,
                # trainer=pl.Trainer(accelerator='cpu'),
                verbose=False, # False
                n_jobs=1, # -1
            ).values().reshape(forecast_periods, )

        elif self.algorithm == 'tft':
            forecast = None
        """

        if debug:
            print(f'len(forecast): {len(forecast)}\n'
                  f'forecast[-100:]: {forecast[-100:]}\n\n')

        return forecast

    def needs_refitting(
        self, 
        refit_date
    ) -> bool:
        if not np.isposinf(self.refit_freq):
            if refit_date >= self.last_fitting_date + pd.Timedelta(minutes=self.mbp * self.refit_freq):
                return True
        return False

    def return_forecast(
        self,
        train_target: pd.DataFrame = None,
        forecast_target: pd.DataFrame = None,
        train_features: pd.DataFrame = None,
        forecast_features: pd.DataFrame = None,
        forecast_dates=None,
        add_bias: float = None,
        steps: int = None,
        ignore_update: bool = False,
        max_t: int = None,
        raw_forecasts: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Define Forecast Parameters & Datasets
        """
        # Define start_data
        start_time = time.time()
        
        # Validate forecast_dates
        if forecast_dates is None:
            if forecast_features is None:
                LOGGER.critical('If forecast_features is "None", then forecast_dates can not be "None" as well.')
                raise Exception('If forecast_features is "None", then forecast_dates can not be "None" as well.\n\n')
            forecast_dates = forecast_features.index

        # Validate steps
        if steps is None:
            steps = min([self.refit_freq, len(forecast_dates)])
        
        if self.algorithm == 'n_beats':
            steps = 1

        # Validate forecast_multiplier
        if self.forecast_multiplier is None:
            LOGGER.warning('Setting self.forecast_multiplier to 1.')
            self.forecast_multiplier = 1

        # Find prev_obs
        prev_obs = forecast_dates - pd.Timedelta(value=self.mbp, unit='minutes')

        if debug:
            print(f'steps: {steps}\n'
                  f'Forecast Dates ({len(forecast_dates)}): {forecast_dates}\n'
                  f'last prev_obs: {prev_obs[-10:]}\n\n')

        # Find full_target & full_features
        full_target = pd.concat([train_target, forecast_target], axis=0)
        if not(train_features is None and forecast_features is None):
            full_features = pd.concat([train_features, forecast_features], axis=0)
        else:
            full_features = None

        """
        Run Target Forecasts in Batches (Date Groups) defined by Steps
        """
        # Define empty forecast_df
        forecast_df = pd.DataFrame(columns=[f'{self.method}_forecast'])

        # Define date groups on which to perform forecasting rounds
        if steps > 1:
            date_groups = [forecast_dates[x:x+steps] for x in range(0, len(forecast_dates), steps)]
        else:
            date_groups = [[forecast_date] for forecast_date in forecast_dates]

        # Run a forecasting round on each date_group
        for dates in date_groups:
            if max_t is not None and time.time() - start_time > max_t:
                return None
            
            if debug:
                print(f'dates ({len(dates)}): {dates}\n'
                      f'initial day: {dates[0]}\n'
                      f'last day: {dates[-1]}\n')
                
            if full_features is None:
                # Find new_forecast_features
                new_forecast_features = None

                # Find forecast_periods
                forecast_periods = len(dates)
            else:
                # Find new_forecast_features
                new_forecast_features = full_features.loc[full_features.index.isin(dates)].copy()

                # Validate that new_forecast_features contains all expected dates
                idx_diff = set(dates).symmetric_difference(set(new_forecast_features.index))
                if len(idx_diff) > 0:
                    # Find missing dates
                    missing_dates = [idx for idx in dates if idx not in new_forecast_features.index]
                    LOGGER.warning('There is a difference between expected forecast dates and actual forecast features.')
                    LOGGER.warning(
                        'dates: %s\n'
                        'new_forecast_features.index:\n%s\n'
                        'missing_dates:\n%s\n',
                        dates, pformat(new_forecast_features.index), pformat(missing_dates)
                    )

                    if isinstance(dates, list):
                        dates = pd.DatetimeIndex(dates)

                    # Correct new_forecast_features
                    dates = dates.intersection(new_forecast_features.index)
                    new_forecast_features = full_features.loc[full_features.index.isin(dates)].copy()
                
                # Find forecast_periods
                forecast_periods = new_forecast_features.shape[0]

                if debug:
                    print(f'new_forecast_features.shape: {new_forecast_features.shape}\n')

            if len(dates) == 0:
                continue

            # Calculate new target forecasts & multiply by self.forecast_multiplier
            new_forecasts = self.dummy_forecast(
                train_target=train_target,
                train_features=train_features,
                forecast_features=new_forecast_features,
                forecast_periods=forecast_periods,
                debug=debug
            ) * self.forecast_multiplier

            if debug:
                print(f'self.forecast_multiplier: {self.forecast_multiplier}\n'
                      f'new_forecasts ({len(new_forecasts)}): \n{new_forecasts}\n\n')

            # Define new forecast_df
            new_forecast_df = pd.DataFrame(
                new_forecasts,
                columns=[f'{self.method}_forecast'],
                index=dates
            )

            # Concatenate new forecasts
            forecast_df = pd.concat([forecast_df, new_forecast_df], axis=0)

            # Add forecasted dates to train datasets
            train_target = pd.concat([train_target, full_target.loc[full_target.index.isin(dates)]])
            if full_features is not None:
                train_features = pd.concat([train_features, new_forecast_features])
            else:
                train_features = None
            
            # Remove forecasted dates from validation target
            val_target = full_target.loc[~(full_target.index.isin(train_target.index))]
            if full_features is not None:
                val_features = full_features.loc[~(full_features.index.isin(train_features.index))]
            else:
                val_features = None

            # Update model (re-fitting)
            if dates[-1] in prev_obs:
                self._update(
                    train_target=train_target,
                    val_target=val_target,
                    train_features=train_features,
                    val_features=val_features,
                    dates=dates,
                    debug=debug
                )

            if debug:
                print(f'self.last_fitting_date: {self.last_fitting_date}\n\n')

        assert forecast_df.shape[0] == len(forecast_dates)

        # Return target forecasts, if specified
        if raw_forecasts:
            return forecast_df

        # Add target values to forecast_df
        concat_df = full_target.loc[full_target.index >= prev_obs[0]]
        forecast_df = pd.concat([forecast_df, concat_df], axis=1)

        forecast_df[concat_df.columns] = forecast_df[concat_df.columns].shift(1)
        forecast_df = forecast_df.loc[forecast_df.index.isin(forecast_dates)]

        if debug:
            print(f'mean abs {self.method}: {float(full_target[f"target_{self.method}"].abs().mean())}\n'
                  f'mean abs prediction: {float(forecast_df[f"{self.method}_forecast"].abs().mean())}\n'
                  f'raw forecast_df: \n{forecast_df.tail(100)}\n\n')

        if self.method == 'price':
            """
            full_target['target_price_lv'] = full_target['target_price'].shift(1)
            full_target = full_target.loc[full_target.index.isin(forecast_df.index)]

            forecast_df['target_price_lv'] = full_target['target_price_lv']
            forecast_df['return_forecast'] = forecast_df.apply(
                lambda row: (row.price_forecast - row.target_price_lv) / row.target_price_lv, axis=1
            )
            """
            forecast_df = pd.concat([self.forecasts_df, forecast_df], axis=0)
            if debug:
                print(f'previous_forecast: \n{self.forecasts_df}\n\n'
                      f'first price forecasts: \n{forecast_df.head()}\n\n')
            forecast_df['return_forecast'] = forecast_df['price_forecast'].pct_change()
            forecast_df = forecast_df.loc[forecast_df.index.isin(forecast_dates)]
            if debug:
                print(f'new forecast_df.head(): \n{forecast_df.head()}\n\n')

        elif self.method in ['acceleration', 'jerk']:
            # Re-create return_forecast
            if self.method == 'acceleration':
                forecast_df['return_forecast'] = (
                    forecast_df['acceleration_forecast'] 
                    + forecast_df['target_return']
                )
            elif self.method == 'jerk':
                forecast_df['return_forecast'] = (
                    forecast_df['jerk_forecast'] 
                    + forecast_df['target_return']
                    + forecast_df['target_acceleration']
                )

        # Add bias
        if add_bias is not None:
            forecast_df['return_forecast'] = forecast_df['return_forecast'] + add_bias
            if debug:
                print(f'add_bias: {add_bias}\n'
                      f'forecast_df.tail() (after add_bias): \n{forecast_df.tail(20)}\n')

        if debug:
            print(f'full_target.tail(): \n{full_target.tail(5)}\n\n'
                  f'forecast_df.tail(): \n{forecast_df.tail(5)}\n\n')

        # Reverse forecasts
        if self.reverse_forecasts is not None and self.reverse_forecasts:
            forecast_df['return_forecast'] = -1 * forecast_df['return_forecast']
            if debug:
                print(f'forecast_df.tail() (reversed_pred): \n{forecast_df.tail(20)}\n\n')

        # Delete unnecessary datasets from memory
        del train_target
        del forecast_target
        del train_features
        del forecast_features
        del full_target
        del full_features

        # Add guardrails
        forecast_df['return_forecast'] = np.where(
            forecast_df['return_forecast'] > self.max_seen_return,
            self.max_seen_return,
            forecast_df['return_forecast']
        )

        forecast_df['return_forecast'] = np.where(
            forecast_df['return_forecast'] < self.min_seen_return,
            self.min_seen_return,
            forecast_df['return_forecast']
        )

        # Return forecasts
        if ignore_update:
            return forecast_df[['return_forecast']]
        else:
            self.forecasts_df = (
                self.forecasts_df.iloc[:-24]
                .combine_first(forecast_df[['return_forecast']])
                .combine_first(self.forecasts_df)
                .sort_index(ascending=True)
            )

            return self.forecasts_df.loc[self.forecasts_df.index.isin(forecast_df.index)]
    
    def forecast_expected_cum_ret(
        self, 
        start_at: pd.Timestamp,
        n_forecast: int,
        alphas: list = None,
        refit_cum_ret_model: bool = True, 
        update_error: bool = True,
        pesimistic_correction: float = None,
        debug: bool = False
    ) -> pd.DataFrame:
        from scipy.stats import norm

        if alphas is None:
            alphas = [0.5, 0.1, 0.05]

        if self.cum_ret_model is None:
            self.cum_ret_model = {
                'model': None,
                'se': None
            }
        
        train_df = (
            self.test_table
            .loc[self.test_table.index < start_at]
            .interpolate()
            .copy()
        )
        train_df.index = np.arange(1, train_df.shape[0]+1)

        if refit_cum_ret_model or self.cum_ret_model['model'] is None:
            # print(self.test_table['total_cum_returns'])
            self.cum_ret_model['model'] = LinearRegression(n_jobs=-1)
            self.cum_ret_model['model'].fit(
                train_df.index.values.reshape(-1, 1),
                train_df['total_cum_returns']
            )
        
        if update_error or self.cum_ret_model['se'] is None or self.cum_ret_model['rse'] is None:
            train_preds = self.cum_ret_model['model'].predict(
                train_df.index.values.reshape(-1, 1)
            )

            # Standard Error
            self.cum_ret_model['se'] = np.sqrt(mean_squared_error(
                train_df['total_cum_returns'].values, 
                train_preds
            ))

        forecast_idx = np.arange(train_df.shape[0] + 1, train_df.shape[0] + 1 + n_forecast)
        forecasts = self.cum_ret_model['model'].predict(forecast_idx.reshape(-1, 1))

        forecast_cols = ['Cum Ret Forecast']
        for alpha in alphas:
            forecast_cols.extend([f'Lower CI {100 * alpha} %', f'Upper CI {100 * alpha} %'])

        forecasts_ci_df = pd.DataFrame(columns=forecast_cols)
        forecasts_ci_df['Cum Ret Forecast'] = forecasts

        for alpha in alphas:
            z_value = norm.ppf(1 - alpha/2)
            forecasts_ci_df[f'Lower CI {100 * alpha} %'] = forecasts - z_value * self.cum_ret_model['se']
            forecasts_ci_df[f'Upper CI {100 * alpha} %'] = forecasts + z_value * self.cum_ret_model['se']

        forecasts_ci_df = forecasts_ci_df - forecasts[0]

        forecasts_ci_df.index = pd.date_range(
            start=start_at, 
            freq=f'{self.mbp}min', 
            periods=n_forecast
        )

        # pesimistic_correction
        if pesimistic_correction is not None:
            ls = np.linspace(0, forecasts[-1] * pesimistic_correction, len(forecasts))
            for c in forecasts_ci_df.columns:
                forecasts_ci_df[c] = forecasts_ci_df[c] - ls

        if debug:
            print(f'forecasts_ci_df.head():\n {forecasts_ci_df.head()}\n\n')

        return forecasts_ci_df
    
    def update_tables(
        self, 
        y_ml_train: pd.DataFrame,
        y_ml_test: pd.DataFrame,
        X_ml_train: pd.DataFrame,
        X_ml_test: pd.DataFrame,
        actuals_test: pd.DataFrame,
        smooth_returns: bool = False,
        return_weight: float = None,
        accuracy_weight: bool = False,
        debug: bool = False
    ) -> None:
        # Update Forecasts
        new_forecasts = self.return_forecast(
            train_target=y_ml_train,
            forecast_target=y_ml_test,
            train_features=X_ml_train,
            forecast_features=X_ml_test,
            forecast_dates=X_ml_test.index,
            add_bias=None,
            steps=None,
            ignore_update=False,
            debug=debug
        )

        # Find Intersection
        intersection = (
            actuals_test
            .index
            .intersection(new_forecasts.index)
        )
        actuals_test = actuals_test.loc[intersection]
        new_forecasts = new_forecasts.loc[intersection]

        # Prepare New TradingTable input
        table_input = pd.concat(
            [new_forecasts[['return_forecast']], actuals_test], 
            axis=1
        )

        # Update self.test_table
        self.test_table = self.test_table.update(
            new_table_input=table_input.copy(),
            update_performance=True,
            smooth_returns=smooth_returns,
            return_weight=return_weight,
            debug=debug
        )

        # Update self.optimized_table
        if self.optimized_table is not None:
            self.optimized_table = self.optimized_table.update(
                new_table_input=table_input.copy(),
                update_performance=True,
                smooth_returns=smooth_returns,
                return_weight=return_weight,
                debug=debug
            )
        else:
            LOGGER.warning('Model %s optimized table is None.', self.model_id)

    def find_shap_importance(
        self,
        test_features: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        # Instanciate explainer
        explainer = shap.TreeExplainer(self.model)

        # Calculate shap values
        if self.model_class == 'GFM' and self.algorithm != 'lightgbm':
            approximate = True
        else:
            approximate = False
        
        self.shap_values = explainer.shap_values(
            test_features,
            approximate=approximate
        )
        
        # Find the sum of shap feature values
        shap_sum = np.abs(self.shap_values).mean(axis=0)
        
        # Find feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': test_features.columns.tolist(),
            'importance': shap_sum
        })

        # Define importance_method
        self.importance_method = 'shap'

        # Sort DataFrame by shap_value
        importance_df.sort_values(by=['importance'], ascending=False, ignore_index=True, inplace=True)

        # Find shap cumulative percentage importance
        importance_df['cum_perc'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

        # Assign result to the self.feature_importance_df attribute
        self.feature_importance_df = importance_df

        if debug:
            print(f'importance df (top 20) (importance_method: {self.importance_method}): \n{self.feature_importance_df.iloc[:20]}\n\n')

    def find_native_importance(
        self,
        test_features: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        # Define DataFrame to describe importances on (utilizing native feature importance calculation method)
        importance_df = pd.DataFrame({
            'feature': test_features.columns.tolist(),
            'importance': self.model.feature_importances_.tolist()
        })

        # Define importance_method
        self.importance_method = f'native_{self.algorithm}'

        # Sort DataFrame by shap_value
        importance_df.sort_values(by=['importance'], ascending=False, ignore_index=True, inplace=True)

        # Find shap cumulative percentage importance
        importance_df['cum_perc'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

        # Assign result to the self.feature_importance_df attribute
        self.feature_importance_df = importance_df

        if debug:
            print(f'importance df (top 20) (importance_method: {self.importance_method}): \n{self.feature_importance_df.iloc[:20]}\n\n')

    def find_feature_importance(
        self,
        test_features: pd.DataFrame,
        importance_method: str = 'shap', 
        debug: bool = False
    ) -> None:
        print(f'\nFinding feature_importance_df for Model {self.model_id} ({self.stage} | {self.model_class} - {self.intervals}):\n')
        if importance_method == 'shap':
            try:
                # Find Shap Feature Importance
                self.find_shap_importance(
                    test_features=test_features,
                    debug=debug
                )
            except Exception as e:
                LOGGER.warning(
                    'Unable to calculate shap feature importance on %s (%s).\n'
                    'Exception: %s\n'
                    'Re-trying with a native approach.\n',
                    self.model_id, self.algorithm, e
                )
                
                # Find Native Feature Importance
                self.find_native_importance(
                    test_features=test_features,
                    debug=debug
                )
        else:
            # Find Native Feature Importance
            self.find_native_importance(
                test_features=test_features,
                debug=debug
            )

    def build_performance_fig(
        self, 
        plot: bool = False, 
        return_fig: bool = False
    ):
        complete_table = pd.concat([self.val_table, self.test_table])
        complete_table['total_cum_returns'] = ((1 + complete_table['real_trading_return']).cumprod() - 1) * 100
        
        fig = px.line(
            complete_table,
            y='total_cum_returns',
            title=f'{self.coin_name} Val & Test Cumulative Returns',
            color_discrete_sequence=['#5A01B5']
        )
        fig.add_vline(x=self.test_table.index[0], line_width=3, line_dash="dash", line_color="black")
        fig.update_layout(width=900)

        if plot:
            fig.show()
        if return_fig:
            return fig

    def diagnose_model(
        self,
        debug: bool = False
    ) -> Dict[str, bool]:
        # Find Diagnostics Dict
        diagnostics_dict = find_model_diagnosis_dict(model=self)

        # TODO: Include comparison of production predictions & registered predictions in opt/test table
        """
        def check_trading_table(self):
            # File System
            # base_path = os.path.join(
            #     Params.base_cwd, Params.bucket, "trading", "bot", self.intervals
            # )

            # S3
            s3_base_path = f"{Params.bucket}/trading/bot/{self.intervals}"
            
            # Real Returns
            real_returns = (
                self.trading_returns
                .loc[~(
                    (self.trading_returns['real_return'] == 0) &
                    (self.trading_returns['return_residual'] == 0)
                )]
                .filter(items=self.champion.optimized_table.columns.tolist())
                .iloc[:-1]
                .copy()
            )
            real_returns['total_cum_returns'] = real_returns['total_cum_returns'] * 100

            print(f'real_returns:\n {real_returns}\n\n')

            # File System
            # real_returns.to_parquet(os.path.join(base_path, 'real_returns_plot_df.parquet'))

            # S3
            write_to_s3(
                asset=real_returns,
                path=f"{s3_base_path}/real_returns_plot_df.parquet"
            )

            # Top Models Returns
            all_models: List[Model] = (
                self.model_registry.staging_models +
                self.model_registry.prod_models +
                [self.model_registry.champion]
            )
            idx_dict = real_returns.groupby('model_id').groups
            top_concat_list = [
                model.optimized_table.loc[model.optimized_table.index.isin(idx_dict[model.model_id])].copy()
                for model in all_models if model.model_id in idx_dict.keys()
            ]
            top_concat_list = list(filter(lambda df: len(df) > 0, top_concat_list))

            if len(top_concat_list) > 1:
                top_model_rets = pd.concat(top_concat_list, axis=0)
            elif len(top_concat_list) == 1:
                top_model_rets = top_concat_list[0]
            else:
                top_model_rets = None

            if top_model_rets is not None:
                top_model_rets = (
                    top_model_rets
                    .loc[~(top_model_rets.index.duplicated(keep='first')) &
                        (top_model_rets.index.isin(real_returns.index))]
                    .sort_index()
                )

                # Correct first return
                first_idx = real_returns.index[0]
                top_model_rets.at[first_idx, 'real_trading_return'] = real_returns.at[first_idx, 'real_trading_return']
                top_model_rets['total_cum_returns'] = ((1 + top_model_rets['real_trading_return']).cumprod() - 1) * 100

                compare_df = (
                    real_returns
                    .iloc[1:]
                    .compare(
                        top_model_rets.iloc[1:], 
                        result_names=("real_ret", "top_model_ret")
                    )
                    .dropna(axis=1, how='all')
                )

                # Add initial value
                # if cum_rets.iat[0] != 0:
                #     add_date = cum_rets.index[0] - pd.Timedelta(minutes=self.mbp)
                #     cum_rets.loc[add_date] = 0
                #     top_model_rets.loc[add_date, 'total_cum_returns'] = 0

                #     cum_rets.sort_index(inplace=True)
                #     top_model_rets.sort_index(inplace=True)

                # File System
                # top_model_rets.to_parquet(os.path.join(base_path, 'top_model_returns_plot_df.parquet'))

                # S3
                write_to_s3(
                    asset=top_model_rets,
                    path=f"{s3_base_path}/top_model_returns_plot_df.parquet"
                )

                return compare_df
            else:
                # File System
                # pd.DataFrame(
                #     columns=self.champion.optimized_table.columns.tolist()
                # ).to_parquet(os.path.join(base_path, 'top_model_returns_plot_df.parquet'))

                # S3
                write_to_s3(
                    asset=pd.DataFrame(columns=self.champion.optimized_table.columns.tolist()),
                    path=f"{s3_base_path}/top_model_returns_plot_df.parquet"
                )
        """

        if debug:
            print(f'Model {self.model_id} ({self.stage} | {self.model_class} - {self.intervals}) diagnostics_dict:')
            pprint(diagnostics_dict)
            print('\n\nTODO: Include comparison of production predictions & registered predictions in opt/test table.\n\n')

        return diagnostics_dict

    def save_backup(
        self,
        debug: bool = False
    ) -> None:
        # Find Diagnostics Dict
        try:
            diagnostics_dict = self.diagnose_model(debug=debug)
        except Exception as e:
            LOGGER.error(
                'Unable to diagnose Model %s (%s | %s - %s).\n'
                'Exception: %s\n',
                self.model_id, self.stage, self.model_class, self.intervals, e
            )

            diagnostics_dict = {'error_calculating_diagnostics_dict': True}

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                'Unable to save Model %s (%s | %s - %s) to backup.\n'
                'diagnostics_dict:\n%s\n',
                self.model_id, self.stage, self.model_class, self.intervals,
                pformat(diagnostics_dict)
            )
        else:
            LOGGER.info(f"Saving {self.model_id} to backup.")

            # Save Model
            self.save(backup=True)

    def save(
        self,
        pickle_files: bool = True,
        parquet_files: bool = True,
        trading_tables: bool = True,
        model: bool = True,
        backup: bool = False,
        debug: bool = False
    ) -> None:
        # S3
        if backup:
            s3_base_path = f"{Params.bucket}/backup/models/{self.intervals}/{self.model_id}"
        else:
            s3_base_path = self.s3_base_path

        """
        Save .pickle files
        """
        if pickle_files:
            model_attr = {key: value for (key, value) in self.__dict__.items() if key in self.load_pickle}

            # Save pickled attrs
            write_to_s3(
                asset=model_attr,
                path=f"{s3_base_path}/{self.model_id}_model_attr.pickle"
            )

            if debug:
                print(f'Saved Attributes: {[k for k in model_attr.keys()]}\n')
                pprint(model_attr)
                print(f'\n\n')

        """
        Save .parquet files
        """
        if parquet_files:
            for attr_name in self.load_parquet:
                df: pd.DataFrame = getattr(self, attr_name)
                if df is not None:
                    # Save parquet file
                    write_to_s3(
                        asset=df,
                        path=f"{s3_base_path}/{self.model_id}_model_{attr_name}.parquet",
                        partition_cols=None
                    )
                else:
                    LOGGER.warning('Unable to save %s_model_%s.parquet (as attr is None).', self.model_id, attr_name)

        """
        Save self.val_table & self.test_table
        """
        if trading_tables:
            # Define base dir
            s3_base_table_dir = f"trading/trading_table/{self.intervals}/{self.model_id}"

            if self.val_table is not None:
                self.val_table.save()
            else:
                LOGGER.warning(
                    '%s model val_table is None.\n'
                    'Errasing val_table files.\n',
                    self.model_id
                )
                try:
                    # os.remove(os.path.join(base_table_path, f"{self.model_id}_val_trading_df.parquet"))
                    # delete_from_s3(path=f"{s3_base_table_path}/{self.model_id}_val_trading_df.parquet")
                    delete_s3_directory(
                        bucket=Params.bucket, 
                        directory=f"{s3_base_table_dir}/{self.model_id}_val_trading_df"
                    )
                except:
                    pass
                try:
                    # os.remove(os.path.join(base_table_path, f"{self.model_id}_val_trading_df_attr.pickle"))
                    delete_from_s3(path=f"{Params.bucket}/{s3_base_table_dir}/{self.model_id}_val_trading_df_attr.pickle")
                except:
                    pass     
            
            if self.test_table is not None:
                self.test_table.save()
            else:
                LOGGER.warning(
                    '%s model test_table is None.\n'
                    'Errasing test_table files.\n',
                    self.model_id
                )
                try:
                    # os.remove(os.path.join(base_table_path, f"{self.model_id}_test_trading_df.parquet"))
                    # delete_from_s3(path=f"{s3_base_table_path}/{self.model_id}_test_trading_df.parquet")
                    delete_s3_directory(
                        bucket=Params.bucket, 
                        directory=f"{s3_base_table_dir}/{self.model_id}_test_trading_df"
                    )
                except:
                    pass
                try:
                    # os.remove(os.path.join(base_table_path, f"{self.model_id}_test_trading_df_attr.pickle"))
                    delete_from_s3(path=f"{Params.bucket}/{s3_base_table_dir}/{self.model_id}_test_trading_df.pickle")
                except:
                    pass

            if self.optimized_table is not None:
                self.optimized_table.save()
            else:
                LOGGER.warning(
                    '%s model optimized_table is None.\n'
                    'Errasing optimized_table files.\n',
                    self.model_id
                )
                try:
                    # os.remove(os.path.join(base_table_path, f"{self.model_id}_opt_trading_df.parquet"))
                    # delete_from_s3(path=f"{s3_base_table_path}/{self.model_id}_opt_trading_df.parquet")
                    delete_s3_directory(
                        bucket=Params.bucket, 
                        directory=f"{s3_base_table_dir}/{self.model_id}_opt_trading_df"
                    )
                except:
                    pass
                try:
                    # os.remove(os.path.join(base_table_path, f"{self.model_id}_opt_trading_df_attr.pickle"))
                    delete_from_s3(path=f"{Params.bucket}/{s3_base_table_dir}/{self.model_id}_opt_trading_df_attr.pickle")
                except:
                    pass

        """
        Step 4) Save self.model
        """
        if model:
            if self.model is not None:
                if self.algorithm not in ['naive_lv', 'naive_ma']:
                    # Find write_format
                    write_format = self.file_name.split('.')[-1]

                    # Define save_path
                    s3_save_path = f"{s3_base_path}/{self.file_name}"
                    bucket, key = s3_save_path.split('/')[0], '/'.join(s3_save_path.split('/')[1:])

                    if write_format == 'pickle':
                        with tempfile.TemporaryFile() as fp:
                            joblib.dump(self.model, fp)
                            fp.seek(0)
                            S3_CLIENT.put_object(
                                Body=fp.read(), 
                                Bucket=bucket,
                                Key=key
                            )

                    # elif write_format in ['h5', 'ckpt']:
                    #     buffer = io.BytesIO()
                    #     torch.save(self.model, buffer)

                    #     S3_CLIENT.put_object(
                    #         Bucket=bucket, 
                    #         Key=key,
                    #         Body=buffer.getvalue()
                    #     )
                    else:
                        LOGGER.critical('Invalid "write_format" %s parameter.\n\n', write_format)
                        raise Exception(f'Invalid "write_format" {write_format} parameter.\n\n')
            else:
                LOGGER.warning('self.model is None!')

    def load(
        self,
        pickle_files: bool = True,
        parquet_files: bool = True,
        trading_tables: bool = True,
        model: bool = True,
        debug: bool = False
    ) -> None:
        """
        Step 1) Load .pickle files
        """
        if pickle_files:
            try:
                # Find pickled model attrs
                model_attr: dict = load_from_s3(path=f"{self.s3_base_path}/{self.model_id}_model_attr.pickle")

                # Set attrs
                for attr_key, attr_value in model_attr.items():
                    if attr_key in self.load_pickle:
                        setattr(self, attr_key, attr_value)

                # Define file_name & model_name
                self.file_name = self.find_file_name()
                self.model_name = self.find_model_name()

                # Load FeatureSelector selected_features
                fs_selected_features: dict = load_from_s3(
                    path=f"{Params.bucket}/data_processing/feature_selector/{self.intervals}/global/feature_selector_attr.pickle",
                    load_reduced_dataset=False
                )['selected_features']

                # Define available features
                available_features = []
                for key, features in fs_selected_features.items():
                    available_features.extend([f for f in features if f not in available_features])

                # Check selected_features
                remove_features = set(self.selected_features) - set(available_features)
                if len(remove_features) > 0:
                    LOGGER.warning(
                        'Model %s (%s | %s) has features not found in FS.selected_features.\n'
                        'remove_features:\n%s\n',
                        self.model_id, self.stage, self.model_class, pformat(remove_features)
                    )

                    self.selected_features = [f for f in self.selected_features if f in available_features]
                
                if debug:
                    print(f'Loaded Attributes: \n')
                    pprint({k: v for k, v in model_attr.items() if not isinstance(v, list)})
                    print('\n\n')
            except Exception as e:
                    LOGGER.error(
                        'Unable to load model attr (f"%s_model_attr.pickle: %s).\n'
                        'Exception: %s\n',
                        self.model_id, self.intervals, e
                    )
            
        """
        Step 2) Load .parquet files
        """
        if parquet_files:
            for attr_name in self.load_parquet:
                try:
                    # Load parquet file
                    setattr(self, attr_name, load_from_s3(
                        path=f"{self.s3_base_path}/{self.model_id}_model_{attr_name}.parquet",
                        load_reduced_dataset=False,
                        partition_cols=None
                    ))
                except Exception as e:
                    LOGGER.warning(
                        'Unable to load %s_model_%s.parquet.\n'
                        'Exception: %s.\n',
                        self.model_id, attr_name, e
                    )

        """
        Step 3) Load self.val_table, self.test_table & self.optimized_table
            Naming convention:
            - <model_id>_<val/test/opt>_trading_df
        """
        if trading_tables:
            trading_table_input = self.trading_table_input.copy()

            # Validation Table
            self.val_table = TradingTable(
                **trading_table_input.copy(),
                trading_parameters=self.trading_parameters.copy(),
                initialize=True,
                table_name=f"{self.model_id}_val_trading_df",
                load_table=True,
                debug=debug
            )

            # Test Table
            self.test_table = TradingTable(
                **trading_table_input.copy(),
                trading_parameters=self.trading_parameters,
                initialize=True,
                table_name=f"{self.model_id}_test_trading_df",
                load_table=True,
                debug=debug # debug
            )

            # Optimized Table
            self.optimized_table = TradingTable(
                **trading_table_input.copy(),
                trading_parameters=self.optimized_trading_parameters,
                initialize=True,
                table_name=f"{self.model_id}_opt_trading_df",
                load_table=True,
                debug=debug
            )

        """
        Step 4) Load self.model
        """ 
        if model:
            try:
                if self.algorithm == 'naive_lv':
                    self.model = self.hyper_parameters['lv']

                elif self.algorithm == 'naive_ma':
                    self.model = self.hyper_parameters['period'], self.hyper_parameters['weight_type']
                
                else:
                    load_format = self.file_name.split('.')[-1]

                    # File System
                    # save_path = os.path.join(self.base_path, self.file_name)

                    # if load_format == 'pickle':
                    #     self.model = joblib.load(save_path)

                    # elif self.algorithm == 'lstm':
                    #     self.model = load_model(save_path)

                    # elif self.algorithm == 'n_beats':
                    #     self.model = NBEATSModel.load(save_path)

                    # S3
                    s3_save_path = f"{self.s3_base_path}/{self.file_name}"
                    bucket, key = s3_save_path.split('/')[0], '/'.join(s3_save_path.split('/')[1:])

                    if load_format == 'pickle':
                        with tempfile.TemporaryFile() as fp:
                            S3_CLIENT.download_fileobj(
                                Fileobj=fp,
                                Bucket=bucket,
                                Key=key
                            )
                            fp.seek(0)

                            self.model = joblib.load(fp)

                    # elif load_format == 'ckpt':
                    #     obj = S3_CLIENT.get_object(
                    #         Bucket=bucket,
                    #         Key=key
                    #     )
                    #     buffer = io.BytesIO()

                    #     self.model = torch.load(io.BytesIO(obj['Body'].read()), buffer)
            except Exception as e:
                LOGGER.error(
                    'Unable to load model (%s: %s).\n'
                    'Exception: %s\n',
                    self.model_id, self.intervals, e
                )
    
    def __repr__(self) -> str:
        # Define register attributes
        reg_attrs = {
            'Model ID': self.model_id,
            'Model Name': self.model_name,
            'Model File Name': self.file_name,
            'Version': self.version,
            'Stage': self.stage
        }

        # Define general attributes
        gen_attrs = {
            'Coin name': self.coin_name,
            'Intervals': self.intervals,
            'Algorithm': self.algorithm,
            'Method': self.method,
            'PCA': self.pca,
            'Model Class': self.model_class
        }

        # Define ml attributes
        n_features = 0 if self.selected_features is None else len(self.selected_features)

        ml_attrs = {
            'Hyper Parameters': self.hyper_parameters,
            'Trading Parameters': self.trading_parameters,
            'Optimized Trading Parameters': self.optimized_trading_parameters,

            # 'Fitted': self.fitted,
            # 'Refit Model': self.refit_model,
            'Reverse Forecasts': self.reverse_forecasts,
            'Refit Frequency': self.refit_freq,

            'Selected Features (len)': n_features
        }

        # Load Parameters
        # print(f'Loaded Attributes:')
        # pprint({
        #     'Model': self.model,

        #     'Bias': self.bias,
        #     'Forecast Multiplier': self.forecast_multiplier, 
        #     'Last Fitting Datae': self.last_fitting_date,
            
        #     'Forecasts DF (tail)': self.forecasts_df.tail(),
        #     'Feature Importance DF (tail)': self.feature_importance_df.tail(),

        #     'Train IDX (tail)': self.train_idx[-5:],
        #     'Cumulative Return Model': self.cum_ret_model
        # })
        # print('\n\n')

        # Prepare output
        output = "Model:\n"
        output += f"Register Attributes:\n{pformat(reg_attrs)}\n\n"
        output += f"General Attributes:\n{pformat(gen_attrs)}\n\n"
        output += f"ML Attributes:\n{pformat(ml_attrs)}\n\n"

        if self.optimized_table is not None:
            # Add optimized table performances
            table_attrs: str = self.optimized_table.show_attrs(
                general_attrs=False,
                residuals_attrs=False,
                performance_attrs=True
            )

            output += f"Optimized Table performance:\n{table_attrs}"
        elif self.test_table is not None:
            # Add test table performances
            table_attrs: str = self.test_table.show_attrs(
                general_attrs=False,
                residuals_attrs=False,
                performance_attrs=True
            )

            output += f"Test Table performance:\n{table_attrs}"
        else:
            # Add val table performances
            table_attrs: str = self.val_table.show_attrs(
                general_attrs=False,
                residuals_attrs=False,
                performance_attrs=True
            )

            output += f"Validation Table performance:\n{table_attrs}"

        return output
    