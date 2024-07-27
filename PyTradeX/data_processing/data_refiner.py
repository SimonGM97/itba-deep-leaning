from PyTradeX.config.params import Params
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.data_processing.feature_selector import FeatureSelector
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair
)
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing

from statsmodels.tsa.seasonal import seasonal_decompose
from holidays import CountryHoliday
from numba import jit
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import kxy
import ta
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pprint import pprint, pformat
from typing import List, Dict, Tuple, Any


"""
TODO:
    - agregando otras dummy_recomendations como features 
        - sacar ideas de trading strategies
    - mejorar el manual_trend_identifier:
        - usando "60min" en vez de "30min"?
        - agregandole lo de possible trend change (bull_bear o bear_bull)
        - repensar como agregar el bull_perc
        - mejrar la funcion de deteccion de trend (exportar vumanchu de trading view?)
"""


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


class DataRefiner(DataCleaner):

    load_parquet = [
        'y', 'X'
    ]
    load_pickle = [
        'outliers_dict',
    ]

    def __init__(
        self, 
        coin_name: str,
        intervals: str = Params.general_params.get("intervals"),
        overwrite: bool = True,
        mock: bool = False,
        **data_params
    ) -> None:
        # General params
        self.coin_name: str = coin_name
        self.intervals: str = intervals

        # Load param
        self.overwrite: bool = overwrite
        self.mock: bool = mock

        # Data params
        self.lag: int = None
        self.periods: int = None
        self.lag_periods: list = None
        self.rolling_windows: list = None

        self.new_n: int = None
        self.z_threshold: float = None
        self.mbp: int = None

        default_params = [
            'lag',
            'periods',
            'lag_periods',
            'rolling_windows',
            'new_n',
            'z_threshold',
            'mbp'
        ]
        for param in default_params:
            if param not in data_params.keys():
                data_params[param] = getattr(Params, 'data_params')[param]

            setattr(self, param, data_params[param])

        # # Extract intervals lag_periods & rolling_windows
        # self.lag_periods = self.lag_periods[self.intervals]
        # self.rolling_windows = self.rolling_windows[self.intervals]

        # Update periods
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < self.periods:
            # LOGGER.warning(
            #     '%s DR has less periods than required.\n'
            #     'Expected periods: %s\n'
            #     'raw_data_periods: %s\n'
            #     'Thus, self.periods will be reverted to %s.\n',
            #     self.coin_name, self.periods, raw_data_periods, raw_data_periods
            # )
            self.periods: int = raw_data_periods

        # Load parameters
        self.y: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.outliers_dict: Dict[str, tuple] = None

        self.load(debug=False)
    
    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/data_processing/data_refiner/{self.intervals}/{self.coin_name}"
        else:
            return f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/{self.coin_name}"

    @staticmethod
    def find_optimal_seasonal_period(
        time_series, 
        max_period=None,
        plot: bool = False
    ) -> float:
        LOGGER.info("Updating self.seasonal_period.")

        if max_period is None:
            max_period = len(time_series) // 2

        # Perform seasonal decomposition using STL
        decomposition = seasonal_decompose(time_series, period=max_period)

        # Get the seasonal component
        seasonal_component = decomposition.seasonal.dropna()

        # Compute the periodogram
        n = len(seasonal_component)
        fft_values = np.abs(np.fft.fft(seasonal_component)) ** 2
        fft_values = fft_values[:n // 2]
        frequencies = np.fft.fftfreq(n, 1)
        frequencies = frequencies[:n // 2]

        # Find the index of the maximum frequency
        max_index = np.argmax(fft_values)

        # Calculate the optimal seasonal period
        optimal_period = int(1 / frequencies[max_index])

        # Plot the periodogram (optional)
        if plot:
            plt.plot(1 / frequencies, fft_values)
            plt.xlabel('Seasonal Period')
            plt.ylabel('Periodogram')
            plt.title('Periodogram of Seasonal Component')
            plt.show()

        return optimal_period

    def update_seasonal_perod(
        self,
        target_price: pd.Series
    ) -> None:
        # Find periods per day
        periods_per_day = 1440 // self.mbp

        # Find seasonal period
        self.seasonal_period = self.find_optimal_seasonal_period(target_price, periods_per_day*31)

        # Update lag_periods
        self.lag_periods = [1, 2, 3, 4, 6, 12, 24, 48]
        self.lag_periods += [periods_per_day, periods_per_day*7, self.seasonal_period]
        self.lag_periods = [min([i-self.lag, periods_per_day*15]) for i in self.lag_periods if i > self.lag]
        self.lag_periods = sorted(list(set(self.lag_periods)))
        
        # Update rolling_windows
        self.rolling_windows = sorted(list(set([periods_per_day//4, periods_per_day//2, periods_per_day])))
        # self.rolling_windows = [i-self.lag for i in self.rolling_windows if i > self.lag]

        LOGGER.info(
            "self.seasonal_period: %s\n"
            "self.lag_periods: %s\n"
            "self.rolling_windows: %s\n",
            self.seasonal_period, self.lag_periods, self.rolling_windows
        )

    def update_fib_levels(
        self,
        df: pd.DataFrame = None,
        debug: bool = False
    ) -> None:
        LOGGER.info("Updating fib_levels.")

        price_min = df.coin_low.iloc[-self.periods:].min()
        price_max = df.coin_high.iloc[-self.periods:].max()
        diff = price_max - price_min

        self.fib_levels = {
            0: price_min + 1 * diff,
            1: price_min + 0.786 * diff,
            2: price_min + 0.618 * diff,
            3: price_min + 0.5 * diff,
            4: price_min + 0.382 * diff,
            5: price_min + 0.236 * diff,
            6: price_min + 0 * diff
        }

        if debug:
            price_min_index = df.index[df.coin_low == price_min].values[0]
            price_max_index = df.index[df.coin_high == price_max].values[0]
            print(f'price_min: {price_min} ({price_min_index})\n'
                  f'price_max: {price_max} ({price_max_index})\n'
                  f'diff: {diff}\n\n')
        
        LOGGER.info('self.fib_levels:\n%s\n', pformat(self.fib_levels))

    # @timing
    def add_trading_features(
        self, 
        df: pd.DataFrame, 
        debug: bool = False
    ) -> pd.DataFrame:
        # initial_cols = df.columns.tolist().copy()

        # Volume
        df['ta_log_volume'] = np.log(1. + df['ta_volume'])

        # CANDLE STICK
        df['candle_body'] = np.abs(df['coin_price'] - df['coin_open']) / df['coin_open']
        df['candle_range'] = np.abs(df['coin_high'] - df['coin_low']) / df['coin_open']

        df['candle_top_wick'] = (df['coin_high'] - np.maximum(df['coin_price'].copy(), df['coin_open'].copy())) / df['coin_open']
        df['candle_bot_wick'] = (np.minimum(df['coin_price'].copy(), df['coin_open'].copy()) - df['coin_low']) / df['coin_open']

        # HEIKEN ASHI
        #   - Open: (Open(previous candle) + Close(previous candle)) / 2
        #   - Close: (Open + Low + Close + High) / 4
        #   - High: the same of the actual candle
        #   - Low: the same of the actual candle
        df['ha_price'] = (df.coin_open + df.coin_high + df.coin_low + df.coin_price) / 4
        df['ha_open'] = 0
        idx = df.index.copy()

        df.at[idx[0], 'ha_open'] = (df['coin_open'].iloc[0] + df['coin_price'].iloc[0]) / 2
        for i in range(1, len(df)):
            df.loc[idx[i], 'ha_open'] = (df.at[idx[i-1], 'ha_open'] + df.at[idx[i-1], 'ha_price']) / 2

        df['ha_high'] = df[['ha_open', 'ha_price', 'coin_high']].max(axis=1)
        df['ha_low'] = df[['ha_open', 'ha_price', 'coin_low']].min(axis=1)

        # Calculate top price
        df['ha_top_price'] = np.where(
            df.ha_price > df.ha_open,
            df.ha_price, # True
            df.ha_open # False
        )
        
        # Calculate bot price
        df['ha_bot_price'] = np.where(
            df.ha_price < df.ha_open,
            df.ha_price, # True
            df.ha_open # False
        )

        # Calculate candle body
        df['ha_candle_body'] = np.where(
            df.ha_open != 0,
            (df.ha_top_price - df.ha_bot_price) / df.ha_open, # True
            0 # False
        )
        
        # Calculate top wick
        df['ha_top_wick'] = np.where(
            df.ha_open != 0,
            (df.ha_high - df.ha_top_price) / df.ha_open, # True
            0 # False
        )
        
        # Calculate bot wick
        df['ha_bot_wick'] = np.where(
            df.ha_open != 0,
            (df.ha_bot_price - df.ha_low) / df.ha_open, # True
            0 # False
        )

        # TECHNICAL INDICATORS
        for w in [10, 20, 30]:
            MACD = ta.trend.MACD(close=df.coin_price, window_slow=w * 2, window_fast=w, fillna=True)
            RSI = ta.momentum.RSIIndicator(close=df.coin_price, window=w, fillna=True)
            EMA = ta.trend.EMAIndicator(close=df.coin_price, window=w, fillna=True)
            ROC = ta.momentum.ROCIndicator(close=df.coin_price, window=w, fillna=True)
            BB = ta.volatility.BollingerBands(close=df.coin_price, window=w, fillna=True)

            df[f'ta_macd_{w}'] = MACD.macd()
            df[f'ta_rsi_{w}'] = RSI.rsi()
            df[f'ta_roc_{w}'] = ROC.roc()
            df[f'ta_ema_{w}'] = EMA.ema_indicator()

            df[f'ta_price_ema_{w}_diff'] = (df.coin_price - df[f'ta_ema_{w}']) / df.coin_price
            df[f'ta_price_high_bb_{w}_diff'] = (df.coin_price - BB.bollinger_hband()) / df.coin_price
            df[f'ta_price_low_bb_{w}_diff'] = (df.coin_price - BB.bollinger_lband()) / df.coin_price

            df[f'ta_cum_ret_{w}'] = np.exp(np.log(df['coin_return']+1).rolling(w).sum())-1

        def add_fib_features(df: pd.DataFrame, window: int):
            # Find rolling min, max & diff price
            df['rolling_low_min'] = df.coin_low.rolling(window=window, min_periods=1).min()
            df['rolling_high_max'] = df.coin_high.rolling(window=window, min_periods=1).max()
            df['rolling_diff'] = df['rolling_high_max'] - df['rolling_low_min']
            
            # Find fib_levels
            df['fib_level_6'] = df['rolling_low_min'] + 1 * df['rolling_diff']
            df['fib_level_5'] = df['rolling_low_min'] + 0.786 * df['rolling_diff']
            df['fib_level_4'] = df['rolling_low_min'] + 0.618 * df['rolling_diff']
            df['fib_level_3'] = df['rolling_low_min'] + 0.5 * df['rolling_diff']
            df['fib_level_2'] = df['rolling_low_min'] + 0.382 * df['rolling_diff']
            df['fib_level_1'] = df['rolling_low_min'] + 0.236 * df['rolling_diff']
            df['fib_level_0'] = df['rolling_low_min']
            
            # Find fib_group
            def find_fib_group_features(row: pd.Series):
                # Group 1 features
                if row.coin_price < row.fib_level_1:
                    return 1, row.fib_level_1 - row.coin_price, row.coin_price - row.fib_level_0
                
                # Group 2 features
                if row.coin_price < row.fib_level_2 and row.coin_price >= row.fib_level_1:
                    return 2, row.fib_level_2 - row.coin_price, row.coin_price - row.fib_level_1
                
                # Group 3 features
                if row.coin_price < row.fib_level_3 and row.coin_price >= row.fib_level_2:
                    return 3, row.fib_level_3 - row.coin_price, row.coin_price - row.fib_level_2
                
                # Group 4 features
                if row.coin_price < row.fib_level_4 and row.coin_price >= row.fib_level_3:
                    return 4, row.fib_level_4 - row.coin_price, row.coin_price - row.fib_level_3
                
                # Group 5 features
                if row.coin_price < row.fib_level_5 and row.coin_price >= row.fib_level_4:
                    return 5, row.fib_level_5 - row.coin_price, row.coin_price - row.fib_level_4
                
                # Group 6 features
                return 6, row.fib_level_6 - row.coin_price, row.coin_price - row.fib_level_5
                
                
            df[['fib_group', 'fib_upper_level_diff', 'fib_lower_level_diff']] = df.apply(
                lambda row: find_fib_group_features(row), axis='columns', result_type='expand'
            )
            
            # Cast 'fib_group' as int
            df['fib_group'] = df['fib_group'].astype(int)
            
            return df
        
        # Find window size
        periods_per_day = int(1440 // self.mbp)
        window_size = periods_per_day * 31

        # Add additional data required for fib_levels
        ini_idx = df.index.tolist().copy()
        required_n = window_size + self.new_n + 10
        if df.shape[0] < required_n:
            # LOGGER.warning(f'DR ({self.coin_name} | {self.intervals}) has not enough information to run add_fib_levels.\n'
            #       f'Thus, more data will be extracted.\n')
            # Extract cols
            extract_cols = df.columns.to_series().apply(
                lambda x: 'trading_' + x if x.split('_')[0] in ['candle', 'ha', 'ta', 'fib', 'rolling'] else x,
            ).tolist()

            rename_dict = dict(zip(extract_cols, df.columns.tolist()))

            # Extract more data
            add_df = (
                self.X
                .loc[
                    ~self.X.index.isin(df.index), 
                    extract_cols
                ]
                .rename(columns=rename_dict)
            )

            # Add new data to df
            df = (
                pd.concat([add_df, df], axis=0)
                .sort_index(ascending=True)
                .tail(required_n)
            )

            assert df.index.duplicated().sum() == 0

        # Add fib_features
        df = add_fib_features(df=df, window=window_size)

        # Reduce df
        df = df.loc[df.index.isin(ini_idx)]

        df.columns = df.columns.to_series().apply(
            lambda x: 'trading_' + x if x.split('_')[0] in ['candle', 'ha', 'ta', 'fib', 'rolling'] else x,
        )

        if debug:
            candle_ha_cols = ['coin_open', 'coin_price', 'coin_high', 'coin_low']
            candle_ha_cols += [c for c in df.columns if ('_candle_' in c or '_ha_' in c)]
            ta_cols = ['coin_price', 'coin_return'] + [c for c in df.columns if '_ta_' in c]
            fib_cols = ['coin_price'] + [c for c in df.columns if '_fib_' in c]

            print(f'candle_ha_df: \n{df[candle_ha_cols].tail(10)}\n\n'
                  f'technical_indicadors_df: \n{df[ta_cols].tail(10)}\n\n'
                  f'self.fib_levels: \n{self.fib_levels}\n\n'
                  f'fib_df: \n{df[fib_cols].tail(10)}\n\n')
            print(f'df shape after add_trading_features: {df.shape}\n')
        
        return df # [df.columns.difference(initial_cols)]

    # @timing
    def update_outliers_dict(
        self,
        df: pd.DataFrame,
        debug: bool = False
    ) -> None:
        LOGGER.info("Updating outliers_dict %s refined_data.", self.coin_name)
        self.outliers_dict = {}

        for column in ['coin_return', 'coin_acceleration', 'coin_jerk']:
            """
            Alternative method:

            # Define mean, required quantiles & Interquantile range
            mean = df[column].mean()
            q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
            iqr = q3 - q1

            # Populate outliers_dict
            self.outliers_dict[column] = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            """
            # Calculate mean & std
            mean, std = df[column].mean(), df[column].std()

            # Populate outliers_dict
            self.outliers_dict[column] = mean - 1.96 * std, mean + 1.96 * std

            if debug:
                print(f'column: {column}\n'
                      f'mean: {mean}\n'
                      f'std: {std}\n'
                      f'outliers_dict: {self.outliers_dict[column]}\n\n')
        
        LOGGER.info('self.outliers_dict:\n%s\n', pformat(self.outliers_dict))

        # Update expectations with new found self.outliers_dict
        try:
            expectations_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}/{self.coin_name}_X_expectations.json"
            
            expectations: dict = load_from_s3(path=expectations_path)
            expectations["outliers_dict"] = deepcopy(self.outliers_dict)

            # Save new expectations
            write_to_s3(asset=expectations, path=expectations_path)
        except Exception as e:
            LOGGER.error(
                'Unable to find DC %s expectations.\n'
                'Exception: %s.\n',
                self.coin_name, e
            )

    # @timing
    def add_manual_features(
        self, 
        df: pd.DataFrame, 
        debug: bool = False
    ) -> pd.DataFrame:
        # initial_cols = df.columns.tolist().copy()

        # FIND OUTLIERS
        for column in ['coin_return', 'coin_acceleration', 'coin_jerk']:
            bins = [
                -np.inf, 
                self.outliers_dict[column][0],
                self.outliers_dict[column][1],
                np.inf
            ]
            if debug:
                print(f'column: {column}\n'
                      f'bins: {bins}\n\n')

            df[f'{column.split("_")[1]}_outlier'] = pd.cut(
                df[column],
                bins=bins,
                labels=[f'{column.split("_")[1]}_low_outlier',
                        f'{column.split("_")[1]}_no_outlier',
                        f'{column.split("_")[1]}_top_outlier']
            )

            col_name = f'{column.split("_")[1]}_outlier'
            df[col_name] = df[col_name].fillna(f'{column.split("_")[1]}_no_outlier')
            # df[f'{column.split("_")[1]}_outlier'].fillna(f'{column.split("_")[1]}_no_outlier', inplace=True)

            df[f'{column.split("_")[1]}_outlier'] = df[f'{column.split("_")[1]}_outlier'].astype(str)

        # CANDLE & HEIKEN ASHI CANDLE TYPE
        def find_candle_type(row):
            tot = row.trading_candle_body + row.trading_candle_top_wick + row.trading_candle_bot_wick
            if row.trading_candle_body >= 0.78 * tot:
                if row['coin_return'] >= 0:
                    return 'strong_bull_candle'
                return 'strong_bear_candle'
            elif row.trading_candle_body >= 0.5 * tot:
                if row['coin_return'] >= 0:
                    return 'bull_candle'
                return 'bear_candle'
            elif row.trading_candle_bot_wick >= 0.6 * tot:
                if row['coin_return'] >= 0:
                    return 'hammer_candle'
                return 'hanging_man_candle'
            elif row.trading_candle_top_wick >= 0.6 * tot:
                if row['coin_return'] >= 0:
                    return 'inverted_hammer_candle'
                return 'shooting_star_candle'
            elif (row.trading_candle_top_wick >= 0.38 * tot) and (row.trading_candle_bot_wick >= 0.38 * tot):
                if row['coin_return'] >= 0:
                    return 'indecision_bull_candle'
                return 'indecision_bear_candle'
            else:
                if row['coin_return'] >= 0:
                    return 'undefined_bull_candle'
                return 'undefined_bear_candle'

        def find_ha_candle_type(row):
            tot = row.trading_ha_candle_body + row.trading_ha_top_wick + row.trading_ha_bot_wick
            if row['trading_ha_price'] >= row['trading_ha_open']:
                if (row.trading_ha_bot_wick <= 0.05 * tot) and (row.trading_ha_candle_body >= 0.5 * tot):
                    return 'strong_bull_ha_candle'
                if (row.trading_ha_top_wick >= 0.38 * tot) and (row.trading_ha_bot_wick >= 0.38 * tot):
                    return 'indecision_bull_ha_candle'
                return 'undefined_bull_ha_candle'
            else:
                if (row.trading_ha_top_wick <= 0.05 * tot) and (row.trading_ha_candle_body >= 0.5 * tot):
                    return 'strong_bear_ha_candle'
                if (row.trading_ha_top_wick >= 0.38 * tot) and (row.trading_ha_bot_wick >= 0.38 * tot):
                    return 'indecision_bear_ha_candle'
                return 'undefined_bear_ha_candle'

        df['is_candle_type'] = df.apply(find_candle_type, axis=1)
        df['is_ha_candle_type'] = df.apply(find_ha_candle_type, axis=1)

        # TREND IDENTIFIER
        replace_ha_names = {
            'strong_bull_ha_candle': 0,
            'strong_bear_ha_candle': 1,
            'undefined_bull_ha_candle': 2,
            'undefined_bear_ha_candle': 3,
            'indecision_bull_ha_candle': 4,
            'indecision_bear_ha_candle': 5
        }

        replace_trend_names = {
            0: 'bull_trend',
            1: 'bear_trend',
            2: 'indecision'
        }

        df['ha_candle'] = df['is_ha_candle_type'].replace(replace_ha_names).infer_objects(copy=False)
        # df['ha_candle'] = df['is_ha_candle_type'].replace(replace_ha_names)
        
        @jit(nopython=True)
        def trend_identifier(ha_candle):
            res = np.empty(ha_candle.shape)
            res[0] = 0
            for i in range(1, res.shape[0]):
                res[i] = res[i - 1]
                if res[i - 1] == 0:
                    if ha_candle[i] == 1 or (ha_candle[i] == ha_candle[i - 1] == 3):
                        res[i] = 1
                    elif (ha_candle[i] == ha_candle[i - 1] == 4) or (ha_candle[i] == 5):
                        res[i] = 2
                elif res[i - 1] == 1:
                    if ha_candle[i] == 0 or (ha_candle[i] == ha_candle[i - 1] == 2):
                        res[i] = 0
                    elif (ha_candle[i] == ha_candle[i - 1] == 5) or (ha_candle[i] == 4):
                        res[i] = 2
                elif res[i - 1] == 2:
                    if ha_candle[i] == 0 or (ha_candle[i] == ha_candle[i - 1] == 2):
                        res[i] = 0
                    elif ha_candle[i] == 1 or (ha_candle[i] == ha_candle[i - 1] == 3):
                        res[i] = 1
            return res

        df['trend_identifier'] = trend_identifier(*df[['ha_candle']].values.T)
        df['trend_identifier'] = df['trend_identifier'].replace(replace_trend_names)
        df.drop(columns=['ha_candle'], inplace=True)

        df['dummy_pred'] = df.apply(
            lambda row: 1 if row.trend_identifier == 'bull_trend'
            else -1 if row.trend_identifier == 'bear_trend'
            else 0, axis=1
        )

        manual_features_rename_dict = {
            'return_outlier': 'manual_return_outlier',
            'acceleration_outlier': 'manual_acceleration_outlier',
            'jerk_outlier': 'manual_jerk_outlier',

            'is_candle_type': 'manual_is_candle_type',
            'is_ha_candle_type': 'manual_is_ha_candle_type',

            'trend_identifier': 'manual_trend_identifier',
            'dummy_pred': 'manual_dummy_pred'
        }
        df.rename(columns=manual_features_rename_dict, inplace=True)

        if debug:
            outliers_cols = ['coin_return', 'coin_acceleration', 'coin_jerk',
                             'manual_return_outlier', 'manual_acceleration_outlier', 'manual_jerk_outlier']
            candle_type_cols = ['coin_price', 'coin_open', 'coin_high', 'coin_low',
                                'trading_candle_body', 'trading_candle_top_wick', 'trading_candle_bot_wick',
                                'manual_is_candle_type']
            ha_candle_type_cols = ['trading_ha_price', 'trading_ha_open', 'trading_ha_high', 'trading_ha_low',
                                   'trading_ha_candle_body', 'trading_ha_top_wick', 'trading_ha_bot_wick',
                                   'manual_is_ha_candle_type', 'manual_trend_identifier']
            outliers_df = df.loc[df.manual_return_outlier != 'return_no_outlier'][outliers_cols]
            print(f'outliers_df: \n{outliers_df.tail(10)}\n\n'
                  f'candle_type_df: \n{df[candle_type_cols].tail(10)}\n\n'
                  f'ha_candle_type_df: \n{df[ha_candle_type_cols].tail(10)}\n\n')
            print(f'df shape after add_manual_features: {df.shape}\n')

        # @jit
        # def bull_perc(trend):
        #     res = np.empty(trend.shape)
        #     res[:10] = 0
        #     for i in range(10, res.shape[0]):
        #         uneque, counts = np.unique(trend[i-10:i], return_counts=True)
        #         count = dict(zip(uneque, counts))
        #         res[i] = count[0] / count[1]
        #     return res
        #
        # features_data['bull_perc'] = bull_perc(*features_data[['trend_identifier']].values.T)

        return df # [df.columns.difference(initial_cols)]
    
    # @timing
    def add_derivatives(
        self,
        df: pd.DataFrame,
        ret: bool = True,
        accel: bool = True,
        jerk: bool = True,
        debug: bool = False
    ) -> pd.DataFrame:
        # initial_cols = df.columns.tolist().copy()

        col_categories = ['coin', 'long_short', 'other_coins', 'stock', 'economic', 'sentiment', 'on_chain',
                          'trading_candle', 'trading_ha', 'trading_ta', 'trading_fib', 'manual']

        total_cat = {cat: [c for c in df.columns if c.startswith(cat)] for cat in col_categories}

        return_diff_cols = [
            'trading_ta_macd_10', 'trading_ta_macd_20', 'trading_ta_macd_30',
            'trading_ta_rsi_10', 'trading_ta_rsi_20', 'trading_ta_rsi_30',
            'trading_ta_roc_10', 'trading_ta_roc_20', 'trading_ta_roc_30',
            'trading_ta_price_ema_10_diff', 'trading_ta_price_ema_20_diff', 'trading_ta_price_ema_30_diff',
            'trading_ta_price_high_bb_10_diff', 'trading_ta_price_high_bb_20_diff',
            'trading_ta_price_high_bb_30_diff',
            'trading_ta_price_low_bb_10_diff', 'trading_ta_price_low_bb_20_diff', 'trading_ta_price_low_bb_30_diff',
            'trading_ta_cum_ret_10', 'trading_ta_cum_ret_20', 'trading_ta_cum_ret_30',
            'trading_fib_upper_level_diff', 'trading_fib_lower_level_diff', 'manual_dummy_pred'
        ]
        return_diff_cols += total_cat['long_short'] + total_cat['sentiment'] + total_cat['on_chain']

        return_pct_cols = [
            'coin_high', 'coin_low', 'trading_ta_volume', 'trading_ta_log_volume',
            'trading_candle_body', 'trading_candle_range', 'trading_candle_top_wick', 'trading_candle_bot_wick',
            'trading_ha_price', 'trading_ha_open', 'trading_ha_high', 'trading_ha_low',
            'trading_ha_top_price', 'trading_ha_bot_price', 'trading_ha_candle_body',
            'trading_ha_top_wick', 'trading_ha_bot_wick',
            'trading_ta_ema_10', 'trading_ta_ema_20', 'trading_ta_ema_30'
        ]
        return_pct_cols += total_cat['other_coins'] + total_cat['stock']

        # Return
        if ret:
            # Calculate diff returns df
            diff_ret_df = pd.concat([
                df[ret_col].diff().rename(f'{ret_col}_return')
                for ret_col in return_diff_cols
            ], axis=1)

            # Calculate pct returns df
            pct_ret_df = pd.concat([
                df[ret_col].pct_change().rename(f'{ret_col}_return')
                for ret_col in return_pct_cols
            ], axis=1)

            # Concatenate return dfs to df
            df = pd.concat([df, diff_ret_df, pct_ret_df], axis=1)

            # for ret_col in return_diff_cols:
            #     df[f'{ret_col}_return'] = df[ret_col].diff()

            # for ret_col in return_pct_cols:
            #     df[f'{ret_col}_return'] = df[ret_col].pct_change()

            # Correct sentiment cols
            if self.intervals == '30min':
                sentiment_ret_cols = [s + '_return' for s in total_cat['sentiment']]
                if 'sentiment_btc_fgi_btc_fgi_class_return' in sentiment_ret_cols:
                    sentiment_ret_cols.remove('sentiment_btc_fgi_btc_fgi_class_return')
            
                df[sentiment_ret_cols] = df[sentiment_ret_cols].replace([0], np.nan).ffill().fillna(0)

            # Acceleration
            if accel:
                # Define acceleration columns
                acceleration_cols = return_diff_cols + return_pct_cols

                # Calculate accelerations df
                accel_df = pd.concat([
                    df[f'{accel_col}_return'].diff().rename(f'{accel_col}_acceleration')
                    for accel_col in acceleration_cols
                ], axis=1)

                # Concatenate return dfs to df
                df = pd.concat([df, accel_df], axis=1)

                # for accel_col in acceleration_cols:
                #     df[f'{accel_col}_acceleration'] = df[f'{accel_col}_return'].diff()

                # Correct sentiment cols
                if self.intervals == '30min':
                    sentiment_accel_cols = [s + '_acceleration' for s in total_cat['sentiment']]
                    if 'sentiment_btc_fgi_btc_fgi_class_acceleration' in sentiment_accel_cols:
                        sentiment_accel_cols.remove('sentiment_btc_fgi_btc_fgi_class_acceleration')
                
                    df[sentiment_accel_cols] = df[sentiment_accel_cols].replace([0], np.nan).ffill().fillna(0)

                # Jerk
                if jerk:
                    # Calculate jerk df
                    jerk_df = pd.concat([
                        df[f'{jerk_col}_acceleration'].diff().rename(f'{jerk_col}_jerk')
                        for jerk_col in acceleration_cols
                    ], axis=1)

                    # Concatenate return dfs to df
                    df = pd.concat([df, jerk_df], axis=1)

                    # for jerk_col in acceleration_cols:
                    #     df[f'{jerk_col}_jerk'] = df[f'{jerk_col}_acceleration'].diff()

                    # Correct sentiment cols
                    if self.intervals == '30min':
                        sentiment_jerk_cols = [s + '_jerk' for s in total_cat['sentiment']]
                        if 'sentiment_btc_fgi_btc_fgi_class_jerk' in sentiment_jerk_cols:
                            sentiment_jerk_cols.remove('sentiment_btc_fgi_btc_fgi_class_acceleration')
                        df[sentiment_jerk_cols] = df[sentiment_jerk_cols].replace([0], np.nan).ffill().fillna(0)

        if debug:
            rand_return_diff = random.choices(return_diff_cols, k=3)
            rand_return_pct = random.choices(return_pct_cols, k=3)

            print(f'return_diff_cols:')
            pprint(return_diff_cols)
            print(f'\nreturn_pct_cols:')
            pprint(return_pct_cols)
            print(f'\nacceleration_cols:')
            pprint(acceleration_cols)
            print(f'\njerk_cols:')
            pprint(acceleration_cols)
            print(f'\ndiff example: \n'
                    f'{df[[c for c in df.columns if any(c2 in c for c2 in rand_return_diff)]].tail()}\n\n'
                    f'pct example: \n'
                    f'{df[[c for c in df.columns if any(c2 in c for c2 in rand_return_pct)]].tail()}\n\n'
                    f'Features Shape (after adding derivates): {df.shape}\n\n')

        return (
            df # [df.columns.difference(initial_cols)]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )

    # @timing
    def update_primary_filter(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        q_thresh: float = 0.7,
        ff_thresh: float = 0.95,
        cat_perc: float = 0.1,
        debug: bool = False
    ) -> None:
        LOGGER.info("Updating self.primary_filter (%s).", self.coin_name)

        # Find numerical features
        filters = {
            col: self.target_feature_correl_filter(
                y=y[col].copy(),
                X=X.copy(),
                q_thresh=q_thresh,
                debug=debug
            ) for col in y.columns
        }

        filters = {
            col: self.colinear_feature_filter(
                y=y.copy(),
                X=X[filters[col]].copy(),
                thresh=ff_thresh
            ) for col in y.columns
        }

        self.primary_filter = {
            'num': [],
            'cat': []
        }
        for key in filters:
            self.primary_filter['num'].extend([c for c in filters[key] if c not in self.primary_filter['num']])

        # Find categorical features
        self.primary_filter['cat'].extend(self.categorical_features_filter(
            y=y['target_return'],
            X=X.copy(),
            perc=cat_perc
        ))

        print(f"New primary_filter:")
        print(f"len(self.primary_filter['cat']): {len(self.primary_filter['cat'])}\n"
              f"len(self.primary_filter['num']): {len(self.primary_filter['num'])}\n")
        pprint(self.primary_filter)
        print('\n\n')
    
    # @timing
    def find_lag_df(
        self,
        df: pd.DataFrame,
        accelerated: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        def lag_df(df_: pd.DataFrame, lag: int):
            if accelerated:
                df_ = df_.tail(lag+self.new_n+10)

            return (
                df_
                .shift(lag, axis=0)
                .rename(columns=lambda x: f"{x}_lag_{lag}" if x in df_.columns else x)
                .bfill()
            )

        # Add required data
        ini_idx = df.index.tolist().copy()
        required_periods: int = np.max(self.lag_periods) + self.new_n + 10
        if df.shape[0] < required_periods:
            # LOGGER.warning(
            #     'DR (%s | %s) has not enough information to run lag_df.\n'
            #     'Thus, more data will be extracted.\n',
            #     self.coin_name, self.intervals, required_periods, df.shape[0]
            # )

            # Extract more data
            add_df = self.X.loc[~self.X.index.isin(df.index), df.columns.tolist()]

            # Add new data to df
            df = pd.concat([add_df, df], axis=0).sort_index(ascending=True)

            assert df.index.duplicated().sum() == 0

        # Run lag_df
        df = pd.concat(
            [lag_df(df.copy(), lag=lag) for lag in self.lag_periods],
            axis=1
        )

        # Reduce df
        df = df.loc[df.index.isin(ini_idx)]

        if debug:
            print_cols = random.sample(df.columns.tolist(), 2)
            lag_df_print_cols = []
            for col in print_cols:
                lag_df_print_cols.extend([f'{col}_lag_1', f'{col}_lag_2', f'{col}_lag_{np.max(self.lag_periods)}'])

            print_df = pd.concat([
                df[print_cols], 
                df[lag_df_print_cols]
            ], axis=1)
            print(f'lag_df:\n {print_df.tail()}\n\n')
            print(f'lag_df shape: {print_df.shape}\n')
        
        return df

    # @timing
    def find_rolling_df(
        self,
        df: pd.DataFrame,
        accelerated: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        def rolling_df(
            df_: pd.DataFrame, 
            window: int
        ):
            if accelerated:
                df_ = df_.tail(window + self.new_n + 10)

            # Find rolling transformations
            df_ = (
                df_
                .rolling(window, min_periods=1)
                .agg({
                    column: ['mean', 'std', 'min', 'max'] 
                    for column in df_.columns
                })
                .bfill()
            )

            # Rename columns
            df_.columns = [f'_rolling_{window}_'.join(c) for c in df_.columns]

            # Add min-max DataFrame
            max_cols = [c for c in df_.columns if c.endswith('_max')]
            min_cols = [c for c in df_.columns if c.endswith('_min')]
            min_max_df_ = (
                df_
                .filter(items=max_cols) 
                - df_
                .filter(items=min_cols)
                .rename(columns=lambda c: c.replace('_min', '_max'))
            ).rename(columns=lambda c: c.replace('_max', '_range'))
            
            return pd.concat([df_, min_max_df_], axis=1)

        # Add required data
        ini_idx = df.index.tolist().copy()
        required_n = np.max(self.rolling_windows) + self.new_n + 10
        if df.shape[0] < required_n:
            # LOGGER.warning(
            #     'DR (%s | %s) has not enough information to run rolling_df.\n'
            #     'Thus, more data will be extracted.\n',
            #     self.coin_name, self.intervals
            # )

            # Extract more data
            add_df = self.X.loc[~self.X.index.isin(df.index), df.columns.tolist()]

            # Add new data to df
            df = (
                pd.concat([add_df, df], axis=0)
                .sort_index(ascending=True)
                .tail(required_n)
            )

            assert df.index.duplicated().sum() == 0
        
        # Run rolling_df
        df = pd.concat(
            [rolling_df(df.copy(), window=window) for window in self.rolling_windows],
            axis=1
        )

        # Reduce df
        df = df.loc[df.index.isin(ini_idx)]

        if debug:
            print(f'rolling_df shape: {df.shape}\n')
    
        return df

    # @timing
    def find_ema_df(
        self,
        df: pd.DataFrame,
        accelerated: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        def ema(df_: pd.DataFrame, window: int):
            if accelerated:
                df_ = df_.tail(window + self.new_n + 10)

            return (
                df_
                .ewm(span=window, adjust=False, min_periods=1)
                .mean()
                .rename(columns=lambda x: f"{x}_ema_{window}" if x in df_.columns else x)
                .bfill()
            )

        # Add required data
        ini_idx = df.index.tolist().copy()
        required_n = np.max(self.rolling_windows) + self.new_n + 10
        if df.shape[0] < required_n:
            # LOGGER.warning(
            #     'DR (%s | %s) has not enough information to run ema.\n'
            #     'Thus, more data will be extracted.\n',
            #     self.coin_name, self.intervals
            # )

            # Extract more data
            add_df = self.X.loc[~self.X.index.isin(df.index), df.columns.tolist()]

            # Add new data to df
            df = (
                pd.concat([add_df, df], axis=0)
                .sort_index(ascending=True)
                .tail(required_n)
            )

            assert df.index.duplicated().sum() == 0
        
        # Run ema
        df = pd.concat(
            [ema(df.copy(), window=window) for window in self.rolling_windows],
            axis=1
        )

        # Reduce df
        df = df.loc[df.index.isin(ini_idx)]

        if debug:
            print(f'ema_df shape: {df.shape}\n')

        return df

    @staticmethod
    def find_tef_df(
        df: pd.DataFrame, 
        hod: bool = True,
        dow: bool = True, 
        dom: bool = True,
        debug: bool = False
    ) -> pd.DataFrame:
        # Hour of Day
        if hod:
            df['hod_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hod_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        # Day of Week
        if dow:
            df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Day of Month
        if dom:
            df['dom_sin'] = np.sin(2 * np.pi * df.index.day / 31)
            df['dam_cos'] = np.cos(2 * np.pi * df.index.day / 31)

        df = pd.concat([
            df.filter(like='sin', axis=1), 
            df.filter(like='cos', axis=1)
        ], axis=1)

        if debug:
            print(f'tef_df shape: {df.shape}\n')
            
        return df

    @staticmethod
    def find_tbf_df(
        df: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day'] = df.index.day

        df = df[['hour', 'day_of_week', 'day']]

        if debug:
            print(f'tbf_df shape: {df.shape}\n')
        
        return df

    @staticmethod
    def find_hbf_df(
        df: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        spain_holidays = CountryHoliday('US', observed=True)
        df['is_holiday'] = df.index.to_series().apply(lambda x: x.date() in spain_holidays)

        df = df[['is_holiday']]

        if debug:
            print(f'hbf_df shape: {df.shape}\n')
        
        return df

    # @timing
    def refiner_pipeline(
        self,
        df: pd.DataFrame,
        update_outliers_dict: bool = False,
        reset_expectations: bool = False,
        validate_data: bool = True,
        accelerated: bool = False,
        save_mock: bool = False,
        debug: bool = False,
        **update_expectations: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Save mock input df
        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='refiner_pipeline_input'
            )

        # Shorten raw_data
        # df = df.iloc[-self.periods-self.lag:]

        # Define Target
        y: pd.DataFrame = df[Params.fixed_params.get("target_columns")]

        # Define Features
        X: pd.DataFrame = df.drop(columns=Params.fixed_params.get("target_columns"))

        # Define forecast DataFrame
        X_forecast = pd.DataFrame(
            index=X.index[-self.lag:] + pd.Timedelta(minutes=self.mbp),
            columns=X.columns.tolist()
        )

        # Append forecast DtaFrame & shift featuers
        X = (
            pd.concat([X, X_forecast])
            .shift(self.lag)
            .bfill()
            .sort_index(ascending=True)
        )

        # Find keep_idx
        keep_idx = X.index.copy()
        
        # Add Trading Features: 
        #   - Candles
        #   - Heiken-Ashe Candles
        #   - Trading Indicators
        #   - Fib-Retracements
        X = self.add_trading_features(
            df=X.copy(), 
            debug=debug
        )

        # Update outliers_dict
        if self.outliers_dict is None or update_outliers_dict:
            self.update_outliers_dict(
                df=X.copy(),
                debug=debug
            )
        
        # Add manual features
        X = self.add_manual_features(
            df=X.copy(),
            debug=debug
        )

        # Add Derivative Features
        X = self.add_derivatives(
            df=X.copy(),
            ret=True,
            accel=True,
            jerk=False,
            debug=debug
        )
            
        # Define input_dfs
        # .filter(items=self.primary_filter['num'] + self.primary_filter['cat'])
        lag_input_df_cols = [c for c in X.columns if not c.endswith('acceleration') and not c.endswith('jerk')]
        lag_input_df = X.filter(items=lag_input_df_cols)
        
        # .filter(items=self.primary_filter['num'])
        rol_input_df_cols = [c for c in lag_input_df_cols if not c.startswith('manual')] + ['manual_dummy_pred', 'manual_dummy_pred_return']
        rol_input_df = X.filter(items=rol_input_df_cols)
        
        # Find lagged Features
        lag_df = self.find_lag_df(
            df=lag_input_df.copy(),
            accelerated=accelerated,
            debug=debug
        )
        
        # Find:
        #   - Simple Moving Average
        #   - Simple Moving Standard Deviation
        #   - Simple Moving Max
        #   - Simple Moving Min
        #   - Simple Moving Min-Max
        rolling_df = self.find_rolling_df(
            df=rol_input_df.copy(),
            accelerated=accelerated,
            debug=debug
        )
        
        # Exponential Moving Average
        ema_df = self.find_ema_df(
            df=rol_input_df.copy(),
            accelerated=accelerated,
            debug=debug
        )
        
        # Temporal Embedding Features
        tef_df = self.find_tef_df(
            df=X.copy(),
            hod=True,
            dow=True, 
            dom=True,
            debug=debug
        )
        
        # Time-Based Features
        tbf_df = self.find_tbf_df(
            df=X.copy(),
            debug=debug
        )
        
        # Holiday-Based Features
        hbf_df = self.find_hbf_df(
            df=X.copy(),
            debug=debug
        )

        # Concatenate partial DataFrames
        X = (
            pd.concat([
                X, 
                lag_df.loc[lag_df.index.isin(keep_idx)],
                rolling_df.loc[rolling_df.index.isin(keep_idx)],
                ema_df.loc[ema_df.index.isin(keep_idx)],
                tef_df,
                tbf_df,
                hbf_df
            ], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )

        # Verify there are no nulls left
        # if X.isnull().sum().sum() > 0:
        #     # Find Columns
        #     str_cols = list(X.select_dtypes(include=['object', 'category']).columns)
        #     num_cols = list(X.select_dtypes(include=['number']).columns)

        #     # Fill nulls
        #     X[num_cols] = X[num_cols].fillna(0)
        #     X[str_cols] = (
        #         X[str_cols]
        #         .bfill()
        #         .ffill()
        #     )
        
        # Update expectations
        if reset_expectations:
            self.update_expectations(
                X=X,
                debug=debug
            )
             
        # Validate data
        if validate_data:
            # Validate y
            y = self.validate_data(
                df=y.copy(),
                df_name='y',
                repair=True,
                debug=debug,
                **update_expectations
            )

            # Validate X
            X = self.validate_data(
                df=X.copy(),
                df_name='X',
                repair=True,
                debug=debug,
                **update_expectations
            )
        
        if debug:
            print(f'concat_df.shape: {df.shape}\n')
            print(f'concat_df column duplicates: {df.columns.duplicated().sum()}\n')

        # Save mock output df
        if save_mock:
            # Save y
            self.save_mock_asset(
                asset=y,
                asset_name='refiner_pipeline_output_y'
            )

            # Save X
            self.save_mock_asset(
                asset=X,
                asset_name='refiner_pipeline_output_X'
            )
        
        return y, X
    
    # @timing
    def update(
        self,
        cleaned_data_shift: pd.DataFrame,
        debug: bool = False,
        **update_params
    ) -> None:
        # Set Up Update Parameters
        complete_update_params = {
            'update_data': False,
            'rewrite_data': False,
            'update_expectations': False,
            'update_outliers_dict': False,
            'validate_data': False,
            'save': False
        }
        for k, v in complete_update_params.items():
            if k not in update_params.keys():
                update_params[k] = v

        # Update y & X
        if (
            self.y is None 
            or self.X is None 
            or update_params['rewrite_data']
            or (
                cleaned_data_shift.index[-1] > self.X.index[-2] 
                and update_params['update_data']
            )
        ):
            self.y, self.X = self.refiner_pipeline(
                df=cleaned_data_shift.copy(),
                update_outliers_dict=update_params['update_outliers_dict'],
                reset_expectations=update_params['rewrite_data'],
                validate_data=update_params['validate_data'],
                debug=debug,
                **{
                    'expected_periods': self.periods
                }
            )

            if self.overwrite:
                LOGGER.info(
                    'self.y.shape: %s\n'
                    'self.X.shape: %s\n',
                    self.y.shape, self.X.shape
                )

        if debug:
            print(f'self.enrich_features has been ran.\n'
                  f'self.X.shape: {self.X.shape}\n'
                  f'refined_data columns:')
            pprint(self.X.columns.tolist())

        # Update expectations
        if update_params['update_expectations']:
            self.update_expectations(debug=debug)

        # Validate Data
        if update_params['validate_data']:
            # Validate "y"
            self.y = self.validate_data(
                df=self.y.copy(),
                df_name='y',
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )

            # Validate "X"
            self.X = self.validate_data(
                df=self.X.copy(),
                df_name='X',
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )

        """
        Save Data Refiner
        """
        if update_params['save']:
            self.save(debug=debug)

    def update_expectations(
        self,
        X: pd.DataFrame = None,
        debug: bool = False
    ) -> None:
        # Validate input datasets
        if X is None:
            X = self.X.copy()

        # Define asset_paths
        base_path = f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/{self.coin_name}"
        s3_y_asset_path = f"{base_path}/{self.coin_name}_y.parquet"
        s3_X_asset_path = f"{base_path}/{self.coin_name}_X.parquet"

        # Extract y expected cols
        y_expected_cols: List[str] = [
            'target_price', 'target_return', 'target_acceleration', 'target_jerk'
        ]

        # Extract X expected cols
        X_expected_cols: List[str] = X.columns.tolist()
        # X_expected_cols = list(filter(
        #     lambda col: not(col.startswith('manual_')), 
        #     list(set(X_expected_cols))
        # ))

        # Define y expected schema
        y_expected_schema = {
            col: 'float' for col in y_expected_cols
        }

        # Define X expected schema
        X_expected_schema = {
            col: 'float' for col in list(X.select_dtypes(include=['number']).columns)
        }
        X_expected_schema.update(**{
            col: 'str' for col in list(X.select_dtypes(include=['object']).columns)
        })

        # Define y max values allowed
        y_max_values_allowed = None

        # Define X max values allowed
        X_max_values_allowed = {
            'long_short_global_long_perc': 1.0,
            'long_short_top_traders_long_perc': 1.0
        }

        # Define y min values allowed
        y_min_values_allowed = None

        # Define X min values allowed
        X_min_values_allowed = {
            col: 0.0 for col in list(X.select_dtypes(include=['number']).columns)
            if not(
                'return' in col 
                or 'acceleration' in col
                or 'jerk' in col
                or 'dummy_pred' in col
                or '_diff' in col
                or '_macd_' in col
                or '_roc_' in col
                or 'cum_ret' in col
                or 'sin' in col
                or 'cos' in col
            )
        }

        # Unique y values allowed
        y_unique_values_allowed = None

        # Unique X values allowed
        X_unique_values_allowed = None

        # Null y values percentage allowed            
        y_null_perc_allowed: Dict[str, float] = {
            col: 0.0 for col in y_expected_cols
        }

        # Null X values percentage allowed            
        X_null_perc_allowed: Dict[str, float] = {
            col: 0.0 for col in X_expected_cols
        }
        
        # Define y duplicate rows subset
        y_duplicate_rows_subset = None

        # Define X duplicate rows subset
        X_duplicate_rows_subset = X_expected_cols.copy()

        # Expected periods
        expected_periods = Params.data_params.get("periods")
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < expected_periods:
            expected_periods = raw_data_periods

        # Load DataCleaner attrs
        base_data_cleaner_attr_path = f"{Params.bucket}/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"
        pickle_attrs: dict = load_from_s3(
            path=f"{base_data_cleaner_attr_path}/{self.coin_name}_data_cleaner_attr.pickle"
        )

        # Load outliers_dict
        X_outliers_dict: dict = pickle_attrs.get("outliers_dict")

        for col in y_expected_cols:
            if col in X_outliers_dict.keys():
                X_outliers_dict.pop(col)

        X_outliers_dict['trading_ta_volume'] = X_outliers_dict.pop('ta_volume')

        # Other Coins
        other_coins_n = Params.data_params.get("other_coins_n")
        other_coins = Params.other_coins_json[self.intervals][:other_coins_n]

        # Define y expectations
        s3_y_expectations = {
            "asset_path": s3_y_asset_path,
            "check_new_missing_data": False,
            "check_missing_cols": True,
            "check_unexpected_cols": True,
            "check_missing_rows": True,
            "check_null_values": True,
            "check_duplicated_idx": True,
            "check_duplicates_rows": True,
            "check_duplicated_cols": True,
            "check_max_values_allowed": True,
            "check_min_values_allowed": True,
            "check_unique_values_allowed": True,
            "check_inconsistent_prices": False,
            "check_extreme_values": False,
            "check_excess_features": True,
            "check_short_length": True,

            "expected_cols": y_expected_cols,
            "expected_schema": y_expected_schema,
            "max_values_allowed": y_max_values_allowed,
            "min_values_allowed": y_min_values_allowed,
            "unique_values_allowed": y_unique_values_allowed,
            "null_perc_allowed": y_null_perc_allowed,

            "duplicate_rows_subset": y_duplicate_rows_subset,
            "outliers_dict": None,
            "max_features_perc": Params.data_params.get("max_features"),
            "other_coins": None,
            "expected_periods": expected_periods
        }
        
        # Define X expectations
        s3_X_expectations = {
            "asset_path": s3_X_asset_path,
            "check_new_missing_data": False,
            "check_missing_cols": True,
            "check_unexpected_cols": False,            
            "check_missing_rows": True,
            "check_null_values": True,
            "check_duplicated_idx": True,
            "check_duplicates_rows": True,
            "check_duplicated_cols": True,
            "check_max_values_allowed": True,
            "check_min_values_allowed": True,
            "check_unique_values_allowed": True,
            "check_inconsistent_prices": False,
            "check_extreme_values": True,
            "check_excess_features": False,
            "check_short_length": True,

            "expected_cols": X_expected_cols,
            "expected_schema": X_expected_schema,
            "max_values_allowed": X_max_values_allowed,
            "min_values_allowed": X_min_values_allowed,
            "unique_values_allowed": X_unique_values_allowed,
            "null_perc_allowed": X_null_perc_allowed,

            "duplicate_rows_subset": X_duplicate_rows_subset,
            "outliers_dict": X_outliers_dict,
            "max_features_perc": None,
            "other_coins": other_coins,
            "expected_periods": expected_periods
        }

        if debug:
            LOGGER.debug('y_expectations:\n%s\n', pformat(s3_y_expectations))
            
            show_s3_X_expectations = deepcopy(s3_X_expectations)
            show_s3_X_expectations['expected_cols'] = len(X_expected_cols)
            LOGGER.debug('X_expectations:\n%s\n', pformat(show_s3_X_expectations))

        # Save Expectations
        s3_expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"

        write_to_s3(
            asset=s3_y_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_y_expectations.json"
        )

        write_to_s3(
            asset=s3_X_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_X_expectations.json"
        )

    def diagnose_data(
        self,
        df: pd.DataFrame = None,
        df_name: str = 'y',
        debug: bool = False,
        **update_expectations: dict
    ) -> Dict[str, bool]:
        # Validate df_name
        if df_name not in ['y', 'X']:
            LOGGER.critical('Invalid "df_name" parameter: %s', df_name)
            raise Exception(f'Invalid "df_name" parameter: {df_name}\n\n')

        # Extract Datasets
        if df is None:
            if df_name == 'y':
                df = self.y.copy()
            else:
                df = self.X.copy()

        # Find Diagnostics Dict
        diagnostics_dict = find_data_diagnosis_dict(
            df_name=df_name,
            intervals=self.intervals,
            coin_name=self.coin_name,
            df=df,
            debug=debug,
            **update_expectations
        )

        if debug:
            print(f'{self.coin_name} {df_name} diagnostics_dict:')
            pprint(diagnostics_dict)
            print(f'\n\n')
        
        return diagnostics_dict
    
    # @timing
    def validate_data(
        self,
        df: pd.DataFrame = None,
        df_name: str = 'y',
        repair: bool = True,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        # Validate df_name
        if df_name not in ['y', 'X']:
            LOGGER.critical('Invalid "df_name" parameter: %s', df_name)
            raise Exception(f'Invalid "df_name" parameter: {df_name}\n\n')

        # Extract Datasets
        if df is None:
            if df_name == 'y':
                df = self.y.copy()
            else:
                df = self.X.copy()

        # Find Diagnostics Dict
        diagnostics_dict = self.diagnose_data(
            df=df,
            df_name=df_name,
            debug=debug,
            **update_expectations
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                "%s %s needs repair.\n"
                "diagnostics_dict:\n%s\n",
                self.coin_name, df_name, 
                pformat(diagnostics_dict)
            )

            if repair:
                LOGGER.info("Repairing %s %s...", self.coin_name, df_name)
                """
                Diagnostics Dict:
                    - has_missing_new_data
                    - has_missing_columns
                    - has_unexpected_columns
                    - has_missing_rows
                    - has_null_values
                    - has_duplicated_idx
                    - has_duplicated_columns
                    - has_unexpected_negative_values
                    - has_negative_prices
                    - has_extreme_values
                    - has_excess_features
                    - has_short_length
                }
                """
                # Load DataShifter dataset
                cleaned_data_shift: pd.DataFrame = load_from_s3(
                    path=f"{Params.bucket}/data_processing/data_shifter/{self.intervals}/{self.coin_name}/{self.coin_name}_cleaned_data_shift.parquet"
                )

                # Re-calculate y & X
                y, X = self.refiner_pipeline(
                    df=cleaned_data_shift,
                    update_outliers_dict=False,
                    reset_expectations=False,
                    validate_data=False,
                    accelerate=False,
                    debug=False
                )

                # Re-assign df
                if df_name == 'y':
                    df = y.tail(self.periods).copy()
                else:
                    df = X.tail(self.periods).copy()

                del y
                del X
            else:
                LOGGER.warning(
                    '%s cleaned_data needed repair, but "repair" parameter was set to False.\n'
                    'Thus, %s cleaned_data will NOT be repaired.\n',
                    self.coin_name, self.coin_name
                )
        
        return df

    def save_mock_asset(
        self,
        asset: Any,
        asset_name: str
    ) -> None:
        # print(f'Saving {asset_name} - [shape: {asset.shape}]')

        # Define base_path
        base_path = f"{Params.bucket}/mock/data_processing/data_refiner/{self.intervals}/{self.coin_name}"

        # Define save_path
        if asset_name == 'refiner_pipeline_input':
            save_path = f"{base_path}/refiner_pipeline_input.parquet"
        elif asset_name == 'refiner_pipeline_output_y':
            save_path = f"{base_path}/refiner_pipeline_output_y.parquet"
        elif asset_name == 'refiner_pipeline_output_X':
            save_path = f"{base_path}/refiner_pipeline_output_X.parquet"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Save dataset in S3
        write_to_s3(asset=asset, path=save_path, overwrite=True)
    
    def load_mock_asset(
        self,
        asset_name: str,
        re_create: bool = False,
        re_create_periods: int = None
    ) -> pd.DataFrame:
        # Define base_paths
        re_create_base_path = f"{Params.bucket}/data_processing/data_shifter/{self.intervals}/{self.coin_name}"
        base_path = f"{Params.bucket}/mock/data_processing/data_refiner/{self.intervals}/{self.coin_name}"

        # Define load_path
        if asset_name == 'refiner_pipeline_input':
            if re_create:
                load_path = f"{re_create_base_path}/{self.coin_name}_cleaned_data_shift.parquet"
            else:
                load_path = f"{base_path}/refiner_pipeline_input.parquet"
        elif asset_name == 'refiner_pipeline_output_y':
            load_path = f"{base_path}/refiner_pipeline_output_y.parquet"
        elif asset_name == 'refiner_pipeline_output_X':
            load_path = f"{base_path}/refiner_pipeline_output_X.parquet"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Load asset from S3
        asset = load_from_s3(path=load_path, ignore_checks=True)

        if (
            re_create
            and re_create_periods is not None 
            and isinstance(asset, pd.DataFrame)
        ):
            asset = asset.tail(re_create_periods)
        
        # print(f'Loaded {asset_name} - [shape: {asset.shape}]')

        return asset

    def save(
        self, 
        debug: bool = False
    ) -> None:
        """
        Save .pickle files
        """
        pickle_attrs = {key: value for (key, value) in self.__dict__.items() if key in self.load_pickle}

        # Save pickled attrs
        write_to_s3(
            asset=pickle_attrs,
            path=f"{self.save_path}/{self.coin_name}_data_refiner_attr.pickle"
        )

        if debug:
            for attr_key, attr_value in pickle_attrs.items():            
                print(f'Saved pickle {attr_key}:')
                pprint(attr_value)
                print('\n')

        """
        Save .parquet files
        """
        for attr_name in self.load_parquet:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                # Save parquet file
                write_to_s3(
                    asset=df,
                    path=f"{self.save_path}/{self.coin_name}_{attr_name}.parquet",
                    overwrite=self.overwrite
                )
            else:
                LOGGER.warning('%s (%s) is None!', attr_name, self.coin_name)

    def load(
        self, 
        debug: bool = False
    ) -> None:
        """
        Load .pickle files
        """
        pickle_attrs = None
        try:
            # Load pickled attrs
            pickle_attrs: dict = load_from_s3(
                path=f"{self.save_path}/{self.coin_name}_data_refiner_attr.pickle"
            )

            for attr_key, attr_value in pickle_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

                    if debug:
                        print(f'Loaded pickle {attr_key}:')
                        pprint(attr_value)
                        print('\n')
        except Exception as e:
            LOGGER.critical(
                'Unable to load data_refiner (%s: %s).\n'
                'Exception: %s\n',
                self.coin_name, self.intervals, e
            )

        """
        Load .parquet files
        """
        if self.overwrite:
            load_reduced_dataset = False
        else:
            load_reduced_dataset = True

        for attr_name in self.load_parquet:
            # Find periods to keep
            keep_periods = self.periods
            if attr_name == 'X':
                keep_periods += self.lag

            try:
                # Load parquet files
                setattr(self, attr_name, load_from_s3(
                    path=f"{self.save_path}/{self.coin_name}_{attr_name}.parquet",
                    load_reduced_dataset=load_reduced_dataset
                ).iloc[-keep_periods:])
            except Exception as e:
                LOGGER.error(
                    'Unable to load %s: %s.\n'
                    'Exception: %s\n', 
                    attr_name, self.coin_name, e
                )

        # Update periods if required
        if load_reduced_dataset and self.y is not None:
            self.periods = self.y.shape[0]

