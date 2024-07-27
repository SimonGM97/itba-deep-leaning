from PyTradeX.config.params import Params
from PyTradeX.utils.trading.trading_helper import load_ltp_lsl_stp_ssl
from PyTradeX.utils.pipeline.pipeline_helper import load_GFM_train_coins
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing
from PyTradeX.modeling.model import Model
from PyTradeX.data_processing.data_extractor import DataExtractor
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.data_processing.data_shifter import DataShifter
from PyTradeX.data_processing.data_refiner import DataRefiner
from PyTradeX.data_processing.feature_selector import FeatureSelector
from PyTradeX.data_processing.data_transformer import DataTransformer
from PyTradeX.trading.trading_table import TradingTable

import pandas as pd
import numpy as np
import time
from copy import deepcopy
from tqdm import tqdm
from wrapt_timeout_decorator import *
from typing import List, Tuple, Dict
from pprint import pprint


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


class MLPipeline:

    mbp = Params.data_params.get("mbp")

    def __init__(
        self,
        pipeline_params: dict,
        ml_params: dict,
        trading_params: dict
    ) -> None:
        # Parameters
        self.pipeline_params = pipeline_params
        self.ml_params = ml_params
        self.trading_params = trading_params

        # Train Datasets
        self.actuals_train: pd.DataFrame = None
        self.y_ml_train: pd.DataFrame = None
        self.X_ml_train: pd.DataFrame = None

        # Val Datasets
        self.actuals_val: pd.DataFrame = None
        self.y_ml_val: pd.DataFrame = None
        self.X_ml_val: pd.DataFrame = None

        # Test Datasets
        self.actuals_test: pd.DataFrame = None
        self.y_ml_test: pd.DataFrame = None
        self.X_ml_test: pd.DataFrame = None

    """
    ML Datasets
    """

    def prepare_LFM_datasets(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        reduced_tuning_periods: int = None,
        coin_name: str = None
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame
    ]:
        # Validate coin_name
        if coin_name is None:
            coin_name = self.pipeline_params.get('coin_name')

        # Extract input datasets
        y_trans: pd.DataFrame = ml_datasets[coin_name]['y_trans'].copy()
        X_trans: pd.DataFrame = ml_datasets[coin_name]['X_trans'].copy()
        X_trans_pca: pd.DataFrame = ml_datasets[coin_name]['X_trans_pca'].copy()
        cleaned_data: pd.DataFrame = ml_datasets[coin_name]['cleaned_data'].copy()

        # Define y_ml, X_ml
        y_ml = y_trans

        selected_features: List[str] = self.pipeline_params.get('selected_features')
        if self.pipeline_params.get('pca'):
            X_ml = X_trans_pca
        else:
            X_ml = X_trans[selected_features]

        # Find intersection index
        intersection = (
            y_ml.index
            .intersection(X_ml.index)
            .intersection(cleaned_data.index)
        )

        # Filter datasets
        y_ml = y_ml.loc[intersection]
        X_ml = X_ml.loc[intersection]
        cleaned_data = cleaned_data.loc[intersection]

        # Prepare actuals
        rename_cols = {
            'coin_return': 'real_return',
            'coin_open': 'open',
            'coin_high': 'high',
            'coin_low': 'low',
            'coin_price': 'price'
        }
        keep_cols = [v for v in rename_cols.values()]

        actuals = (
            cleaned_data
            .rename(columns=rename_cols)
            .filter(items=keep_cols)
        )

        # Reduce periods for tuning purposes
        if reduced_tuning_periods is not None:
            y_ml = y_ml.tail(reduced_tuning_periods)
            X_ml = X_ml.tail(reduced_tuning_periods)
            actuals = actuals.tail(reduced_tuning_periods)

        # Define Train-Val-Test Periods
        periods = y_ml.shape[0]
        train_test_split = self.ml_params.get('train_test_split')
        if self.pipeline_params.get('model_class') == 'LFM':
            train_val_split = self.ml_params.get('lfm_train_val_split')
        else:
            train_val_split = self.ml_params.get('gfm_train_val_split')

        test_periods = int((1 - train_test_split) * periods)
        val_periods = int((1 - train_val_split) * (periods - test_periods))
        train_periods = periods - test_periods - val_periods

        # Define Train Datasets
        actuals_train = actuals.iloc[:train_periods]
        y_ml_train = y_ml.iloc[:train_periods]
        X_ml_train = X_ml.iloc[:train_periods]

        # Define Val Datasets
        actuals_val = actuals.iloc[train_periods:train_periods+val_periods]
        y_ml_val = y_ml.iloc[train_periods:train_periods+val_periods]
        X_ml_val = X_ml.iloc[train_periods:train_periods+val_periods]

        # Define Test Datasets
        actuals_test = actuals.iloc[-test_periods:]
        y_ml_test = y_ml.iloc[-test_periods:]
        X_ml_test = X_ml.iloc[-test_periods:]

        # Delete unnecessary datasets from memory
        del y_trans
        del X_trans
        del X_trans_pca
        del cleaned_data

        del y_ml
        del X_ml
        del actuals
        
        return (
            # Return train datasets
            actuals_train,
            y_ml_train,
            X_ml_train,

            # Return val datasets
            actuals_val,
            y_ml_val,
            X_ml_val,

            # Return test datasets
            actuals_test,
            y_ml_test,
            X_ml_test
        )

    def prepare_GFM_datasets(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        train_coins: List[str] = None,
        reduced_tuning_periods: int = None
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame
    ]:
        # Define empty train gfm datasets
        gfm_actuals_train: pd.DataFrame = None
        gfm_y_ml_train: pd.DataFrame = None
        gfm_X_ml_train: pd.DataFrame = None

        # Define empty val gfm datasets
        gfm_actuals_val: pd.DataFrame = None
        gfm_y_ml_val: pd.DataFrame = None
        gfm_X_ml_val: pd.DataFrame = None

        # Define empty test gfm datasets
        gfm_actuals_test: pd.DataFrame = None
        gfm_y_ml_test: pd.DataFrame = None
        gfm_X_ml_test: pd.DataFrame = None

        # Populate datasets
        # print(f'Preparing {self.pipeline_params.get("model_id")} GFM datasets.\n')
        for coin_name in train_coins:
            # Extract new datasets
            (
                # Train datasets
                new_actuals_train,
                new_y_ml_train,
                new_X_ml_train,

                # Val datasets
                new_actuals_val,
                new_y_ml_val,
                new_X_ml_val,

                # Test datasets
                new_actuals_test,
                new_y_ml_test,
                new_X_ml_test
            ) = self.prepare_LFM_datasets(
                coin_name=coin_name,
                ml_datasets=ml_datasets,
                reduced_tuning_periods=reduced_tuning_periods
            )

            # Add "coin_name"
            new_actuals_train.insert(0, 'coin_name', coin_name)
            new_y_ml_train.insert(0, 'coin_name', coin_name)
            new_X_ml_train.insert(0, 'coin_name', coin_name)

            new_actuals_val.insert(0, 'coin_name', coin_name)
            new_y_ml_val.insert(0, 'coin_name', coin_name)
            new_X_ml_val.insert(0, 'coin_name', coin_name)

            new_actuals_test.insert(0, 'coin_name', coin_name)
            new_y_ml_test.insert(0, 'coin_name', coin_name)
            new_X_ml_test.insert(0, 'coin_name', coin_name)

            # Concatenate train datasets
            gfm_actuals_train = pd.concat([gfm_actuals_train, new_actuals_train], axis=0)
            gfm_y_ml_train = pd.concat([gfm_y_ml_train, new_y_ml_train], axis=0)
            gfm_X_ml_train = pd.concat([gfm_X_ml_train, new_X_ml_train], axis=0)

            # Concatenate val datasets
            gfm_actuals_val = pd.concat([gfm_actuals_val, new_actuals_val], axis=0)
            gfm_y_ml_val = pd.concat([gfm_y_ml_val, new_y_ml_val], axis=0)
            gfm_X_ml_val = pd.concat([gfm_X_ml_val, new_X_ml_val], axis=0)

            # Concatenate test datasets
            gfm_actuals_test = pd.concat([gfm_actuals_test, new_actuals_test], axis=0)
            gfm_y_ml_test = pd.concat([gfm_y_ml_test, new_y_ml_test], axis=0)
            gfm_X_ml_test = pd.concat([gfm_X_ml_test, new_X_ml_test], axis=0)


        return (
            # Return train datasets
            gfm_actuals_train,
            gfm_y_ml_train,
            gfm_X_ml_train,

            # Return val datasets
            gfm_actuals_val,
            gfm_y_ml_val,
            gfm_X_ml_val,

            # Return test datasets
            gfm_actuals_test,
            gfm_y_ml_test,
            gfm_X_ml_test
        )

    def prepare_datasets(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        train_coins: List[str] = None,
        reduced_tuning_periods: int = None,
        debug: bool = False
    ) -> None:
        # Run data-preparation method, depending on model_class
        if self.pipeline_params.get('model_class') == 'LFM':
            datasets = self.prepare_LFM_datasets(
                ml_datasets=ml_datasets,
                reduced_tuning_periods=reduced_tuning_periods
            )
        else:
            datasets = self.prepare_GFM_datasets(
                ml_datasets=ml_datasets,
                train_coins=train_coins,
                reduced_tuning_periods=reduced_tuning_periods
            )

        # Extract attributes
        (
            # Train datasets
            self.actuals_train,
            self.y_ml_train,
            self.X_ml_train,

            # Val datasets
            self.actuals_val,
            self.y_ml_val,
            self.X_ml_val,

            # Test datasets
            self.actuals_test,
            self.y_ml_test,
            self.X_ml_test
        ) = datasets

        if debug:
            LOGGER.debug(
                'actuals_train.shape: %s\n'
                'y_ml_train.shape: %s\n'
                'X_ml_train.shape: %s\n'

                'actuals_val.shape: %s\n'
                'y_ml_val.shape: %s\n'
                'X_ml_val.shape: %s\n'

                'actuals_test.shape: %s\n'
                'y_ml_test.shape: %s\n'
                'X_ml_test.shape: %s\n',
                self.actuals_train.shape, self.y_ml_train.shape,
                self.X_ml_train.shape, self.actuals_val.shape,
                self.y_ml_val.shape, self.X_ml_val.shape,
                self.actuals_test.shape, self.y_ml_test.shape,
                self.X_ml_test.shape
            )
    
    """
    TradingTables
    """

    def find_val_table(
        self,
        model: Model,
        ignore_update: bool = False,
        max_t: float = None,
        debug: bool = False
    ) -> TradingTable:
        # Define datasets
        if model.model_class == 'LFM':
            y_ml_train = self.y_ml_train.copy()
            X_ml_train = self.X_ml_train.copy()

            y_ml_val = self.y_ml_val.copy()
            X_ml_val = self.X_ml_val.copy()

            actuals_val = self.actuals_val.copy()
        else:
            y_ml_train = (
                self.y_ml_train
                .loc[self.y_ml_train['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )
            X_ml_train = (
                self.X_ml_train
                .loc[self.X_ml_train['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

            y_ml_val = (
                self.y_ml_val
                .loc[self.y_ml_val['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )
            X_ml_val = (
                self.X_ml_val
                .loc[self.X_ml_val['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

            actuals_val = (
                self.actuals_val
                .loc[self.actuals_val['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

        # Extract divisions from ml_params
        divisions = self.ml_params.get('divisions')

        # Define forecasting steps
        long_steps = y_ml_val.shape[0]
        medium_steps = int(np.ceil(y_ml_val.shape[0] / divisions))
        short_steps = 1

        steps = {
            'naive_lv': long_steps,
            'naive_ma': long_steps,

            'expo_smooth': medium_steps,
            'sarimax': medium_steps,
            'prophet': medium_steps,

            'random_forest': medium_steps,
            'lightgbm': medium_steps,
            'xgboost': medium_steps,
            
            'lstm': short_steps,
            'n_beats': short_steps
        }[model.algorithm]

        if debug:
            print(f'steps: {steps}\n')

        # Find validation forecasts
        val_forecast = model.return_forecast(
            train_target=y_ml_train,
            forecast_target=y_ml_val,
            train_features=X_ml_train,
            forecast_features=X_ml_val,
            forecast_dates=y_ml_val.index,
            add_bias=None,
            steps=steps,
            ignore_update=ignore_update,
            max_t=max_t,
            debug=debug
        )

        if debug:
            if not isinstance(val_forecast, pd.DataFrame):
                print(f'val_forecast: {val_forecast}\n')
            else:
                print(f'val_forecast.tail():\n {val_forecast.tail()}\n\n')

        # Prepare TradingTable Input
        table_input = pd.concat([val_forecast[['return_forecast']], actuals_val], axis=1)
        
        # Initialize parameters
        trading_table_input = deepcopy(model.trading_table_input)

        trading_table_input.update({
            'table_name': f"{model.model_id}_val_trading_df",
            'initialize': True,
            'load_table': False,
            'debug': False
        })

        if debug:
            print("trading_table_input:\n"
                  "{")
            for key in trading_table_input:
                print(f"    '{key}': {trading_table_input[key]}")
            print('}\n\n')

        # Extract trading_parameters
        trading_parameters = deepcopy(model.trading_parameters)

        # Instanciate val_table
        val_table = TradingTable(
            table_input.copy(),
            trading_parameters=trading_parameters,
            **trading_table_input
        )

        # Complete table
        val_table.complete_table(
            find_best_dist=False,
            dummy_proof=True
        )

        # Measure performance, if table is not dummy
        if not val_table.is_dummy:
            val_table.measure_trading_performance(
                smooth_returns=self.trading_params.get('smooth_returns'),
                return_weight=self.trading_params.get('return_weight'),
                debug=False
            )

        return val_table
    
    def find_test_table(
        self,
        model: Model,
        ignore_update: bool = False,
        debug: bool = False
    ) -> TradingTable:
        # Define datasets
        if model.model_class == 'LFM':
            train_target = pd.concat([self.y_ml_train, self.y_ml_val]).copy()
            train_features = pd.concat([self.X_ml_train, self.X_ml_val]).copy()

            forecast_target = self.y_ml_test.copy()
            forecast_features = self.X_ml_test.copy()

            actuals_test = self.actuals_test.copy()
        else:
            y_ml_train = (
                self.y_ml_train
                .loc[self.y_ml_train['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )
            X_ml_train = (
                self.X_ml_train
                .loc[self.X_ml_train['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

            y_ml_val = (
                self.y_ml_val
                .loc[self.y_ml_val['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )
            X_ml_val = (
                self.X_ml_val
                .loc[self.X_ml_val['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

            train_target = pd.concat([y_ml_train, y_ml_val])
            train_features = pd.concat([X_ml_train, X_ml_val])

            forecast_target = (
                self.y_ml_test
                .loc[self.y_ml_test['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )
            forecast_features = (
                self.X_ml_test
                .loc[self.X_ml_test['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

            actuals_test = (
                self.actuals_test
                .loc[self.X_ml_test['coin_name'] == model.coin_name]
                .drop(columns='coin_name', errors='ignore')
                .copy()
            )

        # Find test forecasts
        test_forecast = model.return_forecast(
            train_target=train_target,
            forecast_target=forecast_target,
            train_features=train_features,
            forecast_features=forecast_features,
            forecast_dates=forecast_target.index,
            add_bias=None,
            steps=None,
            ignore_update=ignore_update,
            max_t=None,
            debug=debug
        )

        # Delete train_target & train_features from memory
        del train_target
        del train_features
        del forecast_target
        del forecast_features

        if debug:
            if not isinstance(test_forecast, pd.DataFrame):
                print(f'test_forecast: {test_forecast}\n')
            else:
                print(f'test_forecast.tail():\n {test_forecast.tail()}\n\n')

        # Prepare TradingTable Input
        table_input = pd.concat([test_forecast[['return_forecast']], actuals_test], axis=1)
        
        # Initialize parameters
        trading_table_input = deepcopy(model.trading_table_input)

        trading_table_input.update({
            'table_name': f"{model.model_id}_test_trading_df",
            'initialize': True,
            'load_table': False,
            'debug': False
        })

        if debug:
            print("trading_table_input:\n"
                  "{")
            for key in trading_table_input:
                print(f"    '{key}': {trading_table_input[key]}")
            print('}\n\n')

        # Extract trading_parameters
        trading_parameters = deepcopy(model.trading_parameters)

        # Instanciate test_table
        test_table = TradingTable(
            table_input.copy(),
            trading_parameters=trading_parameters,
            **trading_table_input
        )

        # Complete test_table
        test_table.complete_table(
            find_best_dist=False, # True
            dummy_proof=False
        )

        # Measure test_table trading performance
        test_table.measure_trading_performance(
            smooth_returns=self.trading_params.get('smooth_returns'),
            return_weight=self.trading_params.get('return_weight'),
            debug=False
        )

        return test_table

    def _find_opt_table(
        self,
        model: Model,
        optimized_params: dict,
        debug: bool = False
    ) -> TradingTable:
        # Create Optimized Table
        input_table_cols = [
            'return_forecast',
            'real_return',
            'open',
            'high',
            'low',
            'price'
        ]

        optimized_table = TradingTable(
            model.test_table[input_table_cols].copy(),
            **model.trading_table_input.copy(),
            trading_parameters=optimized_params,
            initialize=True,
            table_name=f"{model.model_id}_opt_trading_df",
            load_table=False,
            debug=debug
        )

        # Update Residuals Attributes
        for attr_name, attr_value in model.test_table.residuals_attrs.items():
            setattr(optimized_table, attr_name, attr_value)
        
        # Complete table
        optimized_table.complete_table(
            find_best_dist=False,
            dummy_proof=False
        )

        # Measure new performance
        optimized_table.measure_trading_performance(
            smooth_returns=self.trading_params.get('smooth_returns'),
            return_weight=self.trading_params.get('return_weight'),
            debug=False
        )

        return optimized_table

    def tune_opt_table(
        self,
        model: Model,
        debug: bool = False
    ) -> TradingTable:
        print(f'Tuning optimized_table from Model {model.model_id} ({model.stage} | {model.model_class} - {model.intervals}):')
        def helper_fun(
            space: tuple, 
            debug_: bool = False
        ):
            if debug_:
                print('space')
                pprint(space)
                print('\n\n')
            
            # Retrieve Parameters
            follow_flow, certainty_threshold, (ltp, stp), (lsl, ssl), leverage, long_permission, short_permission = space
            if not long_permission and not short_permission:
                return

            optimized_params = {
                'follow_flow': follow_flow,
                'certainty_threshold': certainty_threshold,
                'max_leverage': leverage,
                'tp': (ltp, stp),
                'sl': (lsl, ssl),
                'long_permission': long_permission,
                'short_permission': short_permission
            }

            optimized_table = self._find_opt_table(
                model=model,
                optimized_params=optimized_params,
                debug=debug_
            )

            # if debug:
            #     print(f'optimized_table.trading_metric: {optimized_table.trading_metric}\n\n\n')
            #     fig = px.line(
            #         optimized_table,
            #         y='total_cum_returns',
            #         title=f'{optimized_table.coin_name} test trading_metric: {round(optimized_table.trading_metric, 2)} '
            #               f'(ret_pvalue_score: {round(optimized_table.ret_pvalue_score, 1)}, '
            #               f'weighted_accuracy_score: {round(optimized_table.weighted_accuracy_score, 1)}, '
            #               f'cum_ret_score: {round(optimized_table.cum_ret_score, 2)}%)'
            #     )
            #     fig.show()
            #     time.sleep(10)

            performances.append(optimized_table)

        # Define search space
        follow_flow_list = [True, False]

        certainty_threshold_list = [0.55, 0.575, 0.6, 0.65]

        ltp_, lsl_, stp_, ssl_ = load_ltp_lsl_stp_ssl(
            coin_name=model.coin_name,
            intervals=model.intervals
        )

        tp_list = [(ltp_, stp_), (None, None)]  # (ltp_, stp_) // (ltp_, stp_), (None, None)
        sl_list = [(lsl_, ssl_)]  # (None, None) // (lsl_, ssl_), (None, None)

        long_permission_list = [model.trading_parameters['long_permission']] # [True, False], [True]
        short_permission_list = [model.trading_parameters['short_permission']] # [True, False], [True]

        leverage_list = list(set([1, self.trading_params.get('max_leverage')]))

        performances = []

        search_space = [(follow_flow, thresh, tp, sl, leverage, long_perm, short_perm) 
                        for follow_flow in follow_flow_list
                        for thresh in certainty_threshold_list
                        for tp in tp_list
                        for sl in sl_list
                        for leverage in leverage_list
                        for long_perm in long_permission_list
                        for short_perm in short_permission_list]
        
        # with ThreadPoolExecutor(max_workers=Params.cpus) as executor:
        for space in tqdm(search_space):
            # tune_fun = partial(
            #     helper_fun,
            #     space=space
            # )
            # executor.submit(tune_fun)
            # time.sleep(0.01)
            helper_fun(space=space, debug_=False)

        def performances_sort(t: TradingTable):
            return t.trading_metric
        
        performances.sort(key=performances_sort, reverse=True)

        if debug:
            print(f'best_table')
            pprint(performances[0].trading_metric)
            print('\n\n')

        return performances[0]
    
    def find_opt_table(
        self,
        model: Model,
        tune_opt_table: bool = False,
        debug: bool = False
    ) -> TradingTable:
        if model.optimized_trading_parameters is None or tune_opt_table:
            return self.tune_opt_table(
                model=model,
                debug=debug
            )
        else:
            return self._find_opt_table(
                model=model,
                optimized_params=model.optimized_trading_parameters,
                debug=debug
            )
    
    """
    GFM Specific
    """
    def prepare_idx(
        self,
        df: pd.DataFrame
    ):
        """
        Define actual idx, depending on max_idx and expected df shape. This avoids duplicated idx for
        GFM datasets.
        """
        max_idx = df.index.max()
        n = df.shape[0]
        min_idx = max_idx - pd.Timedelta(minutes=self.mbp * (n-1))
        df.index = pd.date_range(min_idx, max_idx, freq='30min')
        
        return df

    def find_GFM_coin(
        self,
        model: Model,
        debug: bool = False
    ) -> Model:
        # Define top performance
        top_performance = None

        print(f'Finding {model.model_id} best coin.\n')
        for coin_name in tqdm(model.train_coins):
            # Reset required Model attributes
            model.coin_name = coin_name

            # Find forecast_multiplier
            model.find_forecast_multiplier(
                train_target=self.y_ml_train.loc[self.y_ml_train['coin_name'] == model.coin_name].drop(columns='coin_name', errors='ignore'),
                val_target=self.y_ml_val.loc[self.y_ml_val['coin_name'] == model.coin_name].drop(columns='coin_name', errors='ignore'),
                train_features=self.X_ml_train.loc[self.X_ml_train['coin_name'] == model.coin_name].drop(columns='coin_name', errors='ignore'),
                val_features=self.X_ml_val.loc[self.X_ml_val['coin_name'] == model.coin_name].drop(columns='coin_name', errors='ignore'),
                debug=debug
            )

            val_table = self.find_val_table(
                model=model,
                ignore_update=True,
                max_t=None,
                debug=debug
            )

            if val_table.trading_metric is not None:
                if (
                    top_performance is None 
                    or val_table.trading_metric > top_performance['val_table'].trading_metric
                ):
                    top_performance = {
                        'coin_name': coin_name,
                        'val_table': val_table,
                        'forecast_multiplier': model.forecast_multiplier
                    }

            if debug:
                print(f'{coin_name} performances:')
                val_table.show_attrs(
                    general_attrs=False,
                    residuals_attrs=False,
                    performance_attrs=True
                )

        # Set Up Parameters
        model.coin_name = top_performance['coin_name']
        model.val_table = top_performance['val_table']
        model.forecast_multiplier = top_performance['forecast_multiplier']

        print(f'Found performance:')
        model.val_table.show_attrs(
            general_attrs=False,
            residuals_attrs=False,
            performance_attrs=True
        )

        return model

    """
    Pipelines
    """
    @timeout(dec_timeout=None)
    def build_pipeline(
        self,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        reduced_tuning_periods: int = None,
        model: Model = None,
        ignore_update: bool = False,
        find_val_table: bool = False,
        re_fit_train_val: bool = False,
        find_test_table: bool = False,
        find_opt_table: bool = False,
        tune_opt_table: bool = False,
        find_feature_importance: bool = False,
        debug: bool = False
    ) -> Model:
        # Instanciate Model
        if model is None:
            model = Model(
                **self.pipeline_params,
                load_model=False,
                debug=debug
            )
        
        # Find GFM train_coins
        if model.model_class == 'GFM' and model.train_coins is None:
            # Load train_coins_dict
            train_coins_dict = load_GFM_train_coins(
                intervals=model.intervals
            )

            # Re-set model.train_coins
            print(f'Re-setting train_coins in Model {model.model_id} ({model.stage} | {model.model_class} - {model.intervals}).\n'
                  f'len(train_coins): {len(train_coins_dict[model.coin_name][model.method])}.\n\n')
            model.train_coins = train_coins_dict[model.coin_name][model.method]

        # Prepare Datasets
        self.prepare_datasets(
            ml_datasets=ml_datasets,
            train_coins=model.train_coins,
            reduced_tuning_periods=reduced_tuning_periods,
            debug=debug # debug
        )

        # Build Unfitted Model
        if model.model is None:
            model.build(
                train_target=self.y_ml_train.drop(columns='coin_name', errors='ignore'),
                train_features=self.X_ml_train.drop(columns='coin_name', errors='ignore'),
                debug=debug
            )
        
        # Fit Model
        if not model.fitted:
            model.fit(
                train_target=self.y_ml_train.drop(columns='coin_name', errors='ignore'),
                val_target=self.y_ml_val.drop(columns='coin_name', errors='ignore'),
                train_features=self.X_ml_train.drop(columns='coin_name', errors='ignore'),
                val_features=self.X_ml_val.drop(columns='coin_name', errors='ignore'),
                find_forecast_multiplier=True,
                debug=debug
            )
        
        # Update Forecast Multiplier
        if model.forecast_multiplier is None:
            LOGGER.warning('%s self.forecast_multiplier is None.', model.model_id)
            if model.last_fitting_date <= self.y_ml_train.index[-1]:
                model.find_forecast_multiplier(
                    train_target=self.y_ml_train.drop(columns='coin_name', errors='ignore'),
                    val_target=self.y_ml_val.drop(columns='coin_name', errors='ignore'),
                    train_features=self.X_ml_train.drop(columns='coin_name', errors='ignore'),
                    val_features=self.X_ml_val.drop(columns='coin_name', errors='ignore'),
                    debug=debug
                )
            else:
                model.find_forecast_multiplier(
                    train_target=pd.concat([self.y_ml_train, self.y_ml_val]).drop(columns='coin_name', errors='ignore'),
                    val_target=self.y_ml_test.drop(columns='coin_name', errors='ignore'),
                    train_features=pd.concat([self.X_ml_train, self.X_ml_val]).drop(columns='coin_name', errors='ignore'),
                    val_features=self.X_ml_test.drop(columns='coin_name', errors='ignore'),
                    debug=debug
                )

        # Find Validation Table
        if find_val_table or model.val_table is None:
            model.val_table = self.find_val_table(
                model=model,
                ignore_update=ignore_update,
                max_t=None,
                debug=debug
            )

        # Refit with train & validation datasets
        if re_fit_train_val:
            model.fit(
                train_target=pd.concat([self.y_ml_train, self.y_ml_val]).drop(columns='coin_name', errors='ignore'),
                val_target=self.y_ml_test.drop(columns='coin_name', errors='ignore'),
                train_features=pd.concat([self.X_ml_train, self.X_ml_val]).drop(columns='coin_name', errors='ignore'),
                val_features=self.X_ml_test.drop(columns='coin_name', errors='ignore'),
                find_forecast_multiplier=True,
                debug=debug
            )

        # Find Test Table
        if find_test_table:
            model.test_table = self.find_test_table(
                model=model,
                ignore_update=ignore_update,
                debug=debug
            )

        # Find Optimized Performance
        if find_opt_table or tune_opt_table:
            model.optimized_table = self.find_opt_table(
                model=model,
                tune_opt_table=tune_opt_table,
                debug=debug
            )
            model.optimized_trading_parameters = model.optimized_table.trading_parameters.copy()

        # Update Feature Importance
        if find_feature_importance:
            if model.model_class == 'LFM':
                test_features = self.X_ml_test.drop(columns='coin_name', errors='ignore')
            else:
                test_features = self.X_ml_test.loc[self.X_ml_test['coin_name'] == model.coin_name].drop(columns='coin_name', errors='ignore')
            
            model.find_feature_importance(
                test_features=test_features,
                importance_method=self.ml_params.get('importance_method'), 
                debug=True
            )

        return model

    def update_pipeline(
        self,
        model: Model,
        ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
        optimize_trading_parameters: bool = False,
        update_feature_importance: bool = False,
        ignore_last_update_periods: int = None,
        debug: bool = False
    ) -> Model:
        # Prepare pipeline datasets
        datasets = self.prepare_LFM_datasets(
            ml_datasets=ml_datasets,
            reduced_tuning_periods=None
        )

        (
            # Train datasets
            self.actuals_train,
            self.y_ml_train,
            self.X_ml_train,

            # Val datasets
            self.actuals_val,
            self.y_ml_val,
            self.X_ml_val,

            # Test datasets
            self.actuals_test,
            self.y_ml_test,
            self.X_ml_test
        ) = datasets

        # Concat y_ml
        y_ml = pd.concat([self.y_ml_train, self.y_ml_val, self.y_ml_test])
        
        # Concatenate X_ml
        X_ml = pd.concat([self.X_ml_train, self.X_ml_val, self.X_ml_test])

        # Concatenate actuals
        actuals = pd.concat([self.actuals_train, self.actuals_val, self.actuals_test])

        # Find last_test_idx & first_new_idx
        last_test_idx = model.test_table.index[-1]
        first_new_idx = last_test_idx + pd.Timedelta(minutes=self.mbp)
        
        # Define y_ml_train, y_ml_test, X_ml_train, X_ml_test & actuals_test
        y_ml_train, y_ml_test = y_ml.loc[:last_test_idx], y_ml.loc[first_new_idx:]
        X_ml_train, X_ml_test = X_ml.loc[:last_test_idx], X_ml.loc[first_new_idx:]
        actuals_test = actuals.loc[first_new_idx:]

        if ignore_last_update_periods is not None:
            y_ml_test = y_ml_test.iloc[:-ignore_last_update_periods]
            X_ml_test = X_ml_test.iloc[:-ignore_last_update_periods]
            actuals_test = actuals_test.iloc[:-ignore_last_update_periods]

        # Delete pipeline datasets
        del y_ml
        del X_ml
        del actuals

        if debug:
            print(f'y_ml_train.shape: {y_ml_train.shape} {y_ml_train.index[0], y_ml_train.index[-1]}\n'
                  f'X_ml_train.shape: {X_ml_train.shape} {X_ml_train.index[0], X_ml_train.index[-1]}\n'
                  f'actuals_test.shape: {actuals_test.shape} {actuals_test.index[0], actuals_test.index[-1]}\n'
                  f'y_ml_test.shape: {y_ml_test.shape} {y_ml_test.index[0], y_ml_test.index[-1]}\n'
                  f'X_ml_test.shape: {X_ml_test.shape} {X_ml_test.index[0], X_ml_test.index[-1]}\n')

        # Update model tables
        if y_ml_test.shape[0] > 0:
            model.update_tables(
                y_ml_train=y_ml_train,
                y_ml_test=y_ml_test,
                X_ml_train=X_ml_train,
                X_ml_test=X_ml_test,
                actuals_test=actuals_test.copy(),
                smooth_returns=self.trading_params.get('smooth_returns'),
                return_weight=self.trading_params.get('return_weight'),
                debug=debug
            )
        else:
            LOGGER.warning(
                'Unable to update Model %s (%s | %s - %s), as no new observations were found.',
                model.model_id, model.stage, model.model_class, model.intervals
            )

        # Update model optimized_table
        if optimize_trading_parameters or model.optimized_table is None:
            model.optimized_table = self.tune_opt_table(
                model=model,
                debug=debug
            )
            
            model.optimized_trading_parameters = model.optimized_table.trading_parameters.copy()

        # Update Feature Importance
        if update_feature_importance:
            X_ml = pd.concat([X_ml_train, X_ml_test], axis=0)
            test_features = X_ml.loc[X_ml.index.isin(model.test_table.index)]

            model.find_feature_importance(
                test_features=test_features,
                importance_method=self.ml_params.get('importance_method'), 
                debug=debug
            )
        
        return model

    @timing
    def data_pipeline(
        self,
        model: Model,
        DE: DataExtractor,
        DC: DataCleaner,
        DS: DataShifter,
        DR: DataRefiner,
        FS: FeatureSelector,
        DT: DataTransformer,
        loaded_collective_data: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame
    ]:
        # Define category features
        category_names = [
            'coin', 
            'long_short', 
            'other_coins', 
            'stock', 
            'economic',
            'sentiment_lc', 
            'sentiment_btc_fgi', 
            'on_chain'
        ]

        category_features = {
            cat: [f for f in model.selected_features if f.startswith(cat)] for cat in category_names
        }

        # Extract new raw_data
        raw_data: pd.DataFrame = DE.extractor_pipeline(
            periods=DE.new_n,
            loaded_collective_data=loaded_collective_data,
            skip_collective_data_check=False,
            validate_collective_data_=True,
            save_collective_data=False,
            validate_data=True,
            accelerated=True,
            category_features=category_features,
            debug=debug,
            **{'expected_periods': DE.new_n}
        )

        # Combine first with current data
        raw_data = (
            raw_data
            .combine_first(DE.raw_data)
            .tail(DE.new_n)
        )

        # Clean raw_data
        cleaned_data: pd.DataFrame = DC.cleaner_pipeline(
            df=raw_data.copy(),
            unused_data=DE.unused_data,
            remove_unexpected_neg_values=True,
            non_neg_cols=None, # Let DC select non_neg_cols
            remove_inconsistent_prices=True,
            handle_rows_and_columns=True,
            expected_cols=None, # Let DC select expected_cols
            new_data=None,
            remove_outliers=True,
            update_outliers_dict=False,
            z_threshold=DC.z_threshold,
            impute_nulls=True,
            update_imputers=False,
            validate_data=True,
            debug=debug,
            **{'expected_periods': DE.new_n}
        )

        # Shift cleaned_data
        cleaned_data_shift: pd.DataFrame = DS.shifter_pipeline(
            df=cleaned_data.copy(),
            record_df=False,
            reset_consistency_storage=False,
            update_shift_dict=False,
            placeholder=True, # False
            validate_data=True,
            accelerated=True,
            category_features=category_features,
            debug=debug,
            **{'expected_periods': DE.new_n}
        )

        # Refine shifted & cleaned data
        result = DR.refiner_pipeline(
            df=cleaned_data_shift.copy(),
            update_outliers_dict=False,
            reset_expectations=False,
            validate_data=True,
            accelerated=True, # True | False
            debug=debug,
            **{'expected_periods': DE.new_n}
        )

        y: pd.DataFrame = result[0]
        X: pd.DataFrame = result[1]

        # Transform y to obtain y_ml
        y_ml: pd.DataFrame = DT.transformer_pipeline(
            df=y.copy(),
            df_name='y_trans',
            selected_features=deepcopy(FS.selected_features),
            key='target',
            ohe=False,
            scale=False,
            trunc=True,
            pca=False,
            refit_transformers=False,
            validate_data=True,
            debug=debug,
            **{'expected_periods': DE.new_n}
        )

        # Transform X to obtain X_ml
        if model.pca:
            X_ml: pd.DataFrame = DT.transformer_pipeline(
                df=X.copy(),
                df_name='X_trans_pca',
                selected_features=deepcopy(FS.selected_features),
                key='features',
                ohe=True,
                scale=True,
                trunc=True,
                pca=True,
                refit_transformers=False,
                validate_data=True,
                debug=debug,
                **{'expected_periods': DE.new_n}
            )
        else:
            X_ml: pd.DataFrame = DT.transformer_pipeline(
                df=X.copy(),
                df_name='X_trans',
                selected_features=deepcopy(FS.selected_features),
                key='features',
                ohe=True,
                scale=True,
                trunc=True,
                pca=False,
                refit_transformers=False,
                validate_data=True,
                debug=debug,
                **{'expected_periods': DE.new_n}
            )

        # Filter Features
        if not model.pca:
            X_ml = X_ml[model.selected_features.copy()]

        # Define X_fcst
        X_ml, X_fcst = X_ml.iloc[:-model.lag], X_ml.iloc[-model.lag:]

        # Run asserts
        assert len(set(y_ml.index).symmetric_difference(set(X_ml.index))) == 0
        assert X_fcst.shape[0] > 0

        return (
            y_ml, X_ml, X_fcst
        )

    @timing
    def inference_pipeline(
        self,
        model: Model,
        DE: DataExtractor,
        DC: DataCleaner,
        DS: DataShifter,
        DR: DataRefiner,
        FS: FeatureSelector,
        DT: DataTransformer,
        loaded_collective_data: pd.DataFrame = None,
        debug: bool = False
    ) -> dict:
        # Run Data Pipeline
        y_ml, X_ml, X_fcst = self.data_pipeline(
            model=model,
            DE=DE,
            DC=DC,
            DS=DS,
            DR=DR,
            FS=FS,
            DT=DT,
            loaded_collective_data=loaded_collective_data,
            debug=debug
        )

        # Predict y_forecast
        y_forecast = model.return_forecast(
            train_target=y_ml,
            forecast_target=None,
            train_features=X_ml,
            forecast_features=X_fcst,
            forecast_dates=X_fcst.index,
            add_bias=None,
            steps=model.lag,
            ignore_update=True,
            max_t=None,
            raw_forecasts=False,
            debug=debug
        )

        # Prepare forecast TradingTable Input
        last_open = DE.client.last_prices[model.coin_name][0]
        if last_open is None or last_open is np.nan:
            last_open = y_ml['target_price'].iat[-1]
        
        dummy_df = pd.DataFrame(
            np.array([[0, last_open, 0, 0, 0]]),
            columns=['real_return', 'open', 'high', 'low', 'price'],
            index=y_forecast.index
        )

        table_input = pd.concat([y_forecast[['return_forecast']], dummy_df], axis=1)

        # Extract residuals attrs
        dist_attrs = model.optimized_table.residuals_attrs.copy()
        
        # Instanciate new forecast TradingTable
        forecast = TradingTable(
            table_input,
            **model.trading_table_input.copy(),
            trading_parameters=model.optimized_trading_parameters.copy(),
            initialize=True,
            table_name=f'{model.model_id}_forecast_df',
            load_table=False,
            debug=debug
        )

        for attr_name, attr_value in dist_attrs.items():
            setattr(forecast, attr_name, attr_value)

        # Complete forecast TradingTable
        forecast.complete_table(
            find_best_dist=False,
            dummy_proof=False
        )

        # Return dictionary
        drop_cols = [
            'real_return', 'high', 'low', 'price',
            'return_residual', 'ml_accuracy', 'flow',
            'tp_triggered', 'sl_triggered', 
            'trading_signal', 'passive_trading',
            'exposure_adj', # 'exposure',
            'fees', 'trading_return', 
            'real_trading_return', 'total_cum_returns', 
            'trading_accuracy', 'trade_class'
        ]

        forecast_dict = (
            pd.DataFrame(forecast)
            .reset_index(drop=False)
            .drop(columns=drop_cols)
            .rename(columns={'index': 'forecast_date'})
            .to_dict()
        )

        return {
            k: v[0] for k, v in forecast_dict.items()
        }
