from PyTradeX.config.params import Params
from PyTradeX.utils.data_processing.repair_helper import (
    repair_coin_cols,
    repair_stock_and_on_chain_cols,
    drop_duplicates
)
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair
)
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing

from sklearn.impute import SimpleImputer
# from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted
# from missingpy import MissForest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from copy import deepcopy
from pprint import pprint, pformat
from collections.abc import Iterable
from typing import List, Dict, Tuple, Any
from sklearn.exceptions import InconsistentVersionWarning
import warnings

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


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


class DataCleaner:

    load_parquet = [
        'cleaned_data'
    ]
    load_pickle = [
        'price_columns',
        'str_imputer',
        'num_imputer',
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

        # Data Params
        self.lag: int = None
        self.periods: int = None
        self.other_coins_n: int = None

        self.new_n: int = None
        self.z_threshold: float = None
        self.mbp: int = None

        default_params = [
            'lag',
            'periods',
            'other_coins_n',
            'new_n',
            'z_threshold',
            'mbp'
        ]
        for param in default_params:
            if param not in data_params.keys():
                data_params[param] = getattr(Params, 'data_params')[param]

            setattr(self, param, data_params[param])

        # Update periods
        if coin_name is not None:
            raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
            if raw_data_periods < self.periods:
                # LOGGER.warning(
                #     '%s DC has less periods than required.\n'
                #     'Expected periods: %s\n'
                #     'raw_data_periods: %s\n'
                #     'Thus, self.periods will be reverted to %s.\n',
                #     self.coin_name, self.periods, raw_data_periods, raw_data_periods
                # )
                self.periods: int = raw_data_periods
        
        # Extract other_coins
        if self.intervals in Params.other_coins_json.keys():
            self.other_coins: List[str] = Params.other_coins_json[self.intervals][:self.other_coins_n]
        else:
            self.other_coins: List[str] = Params.other_coins_json['30min'][:self.other_coins_n]
            LOGGER.warning(
                'Unable to find other coins in DC %s %s.\n'
                'Setting self.other_coins to %s.\n',
                self.coin_name, self.intervals, self.other_coins
            )

        # Load params
        self.cleaned_data: pd.DataFrame = None
        self.price_columns: List[str] = None

        self.str_imputer: SimpleImputer = None
        self.num_imputer: SimpleImputer = None
        # self.mf_imputer = None

        self.outliers_dict: Dict[str, Tuple[float, float]] = None

        self.load(debug=False)

    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"
        else:
            return f"{Params.bucket}/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"

    def find_price_columns(
        self,
        df: pd.DataFrame
    ) -> None:
        # Re-set self.price_columns
        self.price_columns = [
            c for c in df.columns if (
                c.endswith('_price')
                or c.endswith('_open')
                or c.endswith('_high')
                or c.endswith('_low')
            )
        ]

    def remove_unexpected_neg_values(
        self,
        df: pd.DataFrame,
        non_neg_cols: Iterable = None
    ) -> pd.DataFrame:
        """
        Replace Negative Values in Non-Negative columns
        """
        if non_neg_cols is None:
            # Find self.price_columns if not already found
            if self.price_columns is None:
                self.find_price_columns(df=df)

            # Find volume columns
            vol_cols = [c for c in df.columns if c.endswith('_volume')]

            # Find other non_neg columns
            other_non_neg_cols = [
                c for c in df.columns if (
                    c.startswith('long_short_')
                    or c.startswith('on_chain_')
                    or c.startswith('sentiment_')
                ) and c in Params.fixed_params.get('raw_data_columns')
            ]

            # Re-define non_neg_cols
            non_neg_cols = self.price_columns + vol_cols + other_non_neg_cols

            # print(f'Found non_neg_cols:')
            # pprint(non_neg_cols)
            # print('\n\n')
        
        # Replace Negative Values
        for col in non_neg_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def remove_inconsistent_prices(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Repalce Inconsistent Prices
        """
        # Validate self.price_cols
        if self.price_columns is None:
            self.find_price_columns(df=df)

        for col in self.price_columns:
            # Calculate dummy derivartes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['dummy_ret'] = df[col].pct_change() # fill_method=None).bfill()

            df['dummy_accel'] = df['dummy_ret'].diff()
            df['dummy_jerk'] = df['dummy_accel'].diff()

            # Chalculate mask
            mask = (
                (df['dummy_ret'] == 0) &
                (df['dummy_accel'] == 0) &
                (df['dummy_jerk'] == 0)
            )
            if mask.sum().sum() > 3:
                # Find change_cols
                if col == 'target_price':
                    change_cols = [
                        'target_price', 
                        'target_return',
                        'target_acceleration',
                        'target_jerk'
                    ]
                elif col == 'coin_price':
                    change_cols = [
                        'coin_price', 
                        'coin_return',
                        'coin_acceleration',
                        'coin_jerk'
                    ]
                else:
                    change_cols = [
                        col, 
                        f'{col}_return',
                        f'{col}_acceleration',
                        f'{col}_jerk'
                    ]

                change_cols = [c for c in change_cols if c in df.columns]

                # Replace values
                # print(f"Pre-replaced values:\n {df.loc[mask, change_cols]}\n\n")
                df.loc[mask, change_cols] = np.nan
                # print(f"Post-replaced values:\n {df.loc[mask, change_cols]}\n\n")

            # Drop dummy_cols
            df.drop(columns=['dummy_ret', 'dummy_accel', 'dummy_jerk'], errors='ignore', inplace=True)

        return df

    def add_missing_rows(
        self,
        df: pd.DataFrame,
        intervals: str = None
    ) -> pd.DataFrame:
        if intervals is None:
            intervals = self.intervals

        freq = {
            '30min': '30min',
            '60min': '60min',
            '1d': '1D'
        }[intervals]

        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        return df.reindex(full_idx).fillna(np.nan)

    def handle_expected_cols(
        self,
        df: pd.DataFrame,
        expected_cols: Iterable = None,
        new_data: pd.DataFrame = None,
        coin_name: str = None,
        df_name: str = None
    ) -> pd.DataFrame:
        if coin_name is None:
            coin_name = self.coin_name

        if df_name is None:
            df_name = 'cleaned_data'

        # Find Expected Columns
        if expected_cols is None:
            expected_cols: List[str] = Params.fixed_params.get("raw_data_columns").copy()
            for coin in self.other_coins:
                # if coin != self.coin_name:
                expected_cols.append(f'other_coins_{coin}_price')

        # Drop extra columns
        drop_cols = [c for c in df.columns if c not in expected_cols]
        if len(drop_cols) > 0:
            LOGGER.info("%s %s drop_cols:\n%s\n", coin_name, df_name, pformat(drop_cols))

            df.drop(columns=drop_cols, inplace=True)

        # Add required columns
        added_cols = [c for c in expected_cols if c not in df.columns]
        if len(added_cols) > 0:
            LOGGER.info("%s %s added_cols:\n%s\n", coin_name, df_name, pformat(added_cols))

            if new_data is None:
                df = repair_coin_cols(
                    df=df.copy(),
                    other_coins=self.other_coins, # [c for c in self.other_coins if c != self.coin_name],
                    intervals=self.intervals,
                    price_only=True
                )
            else:
                new_data = (
                    new_data
                    .loc[new_data.index.isin(df.index)]
                    .filter(items=added_cols)
                )
                df = pd.concat([df, new_data], axis=1)
            
        return df

    def update_outliers_dict(
        self,
        df: pd.DataFrame,
        z_threshold: float = None
    ) -> None:
        LOGGER.info('Updating outliers attrs on %s DC.', self.coin_name)

        # Verify self.price_columns
        if self.price_columns is None:
            self.find_price_columns(df=df)

        # Verify z_threshold
        if z_threshold is None:
            z_threshold = self.z_threshold

        # Define num_cols
        num_cols = list(df.select_dtypes(include=['number']).columns)
        
        # Find self.outliers_dict
        def find_threshold(col: str):
            # Calculate mean & std
            mean, std = df[col].mean(), df[col].std()

            # Return thresholds
            if col in self.price_columns:
                if col.endswith('_high'):
                    return 0, mean + 10.0 * std
                return 0, mean + 9.5 * std
            return mean - z_threshold * std, mean + z_threshold * std
        
        self.outliers_dict = {
            col: find_threshold(col) for col in num_cols
        }
        
        try:
            # Update expectations with new found self.outliers_dict
            expectations_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}/{self.coin_name}_cleaned_data_expectations.json"

            expectations: dict = load_from_s3(path=expectations_path)
            expectations["outliers_dict"] = deepcopy(self.outliers_dict)

            # Save new expectations
            write_to_s3(asset=expectations, path=expectations_path)
        except Exception as e:
            LOGGER.error(
                'Unable to update expectations in DC %s %s.\n'
                'Exception: %s.\n',
                self.coin_name, self.intervals, e
            )

    def correct_outliers(
        self,
        df: pd.DataFrame,
        z_threshold: float = None
    ) -> pd.DataFrame:
        # Validate z_threshold, outliers_dict & outliers_dict
        if z_threshold is None:
            z_threshold = self.z_threshold

        if self.outliers_dict is None:
            self.update_outliers_dict(
                df=df.copy(),
                z_threshold=z_threshold
            )

        """
        Correct price outliers
        """
        for col in self.price_columns:
            # Find extreme outliers mask
            extreme_outliers_mask = np.logical_or(
                df[col] < self.outliers_dict[col][0],
                df[col] > self.outliers_dict[col][1]
            )

            # Remove extreme outliers
            df[col] = np.where(extreme_outliers_mask, np.nan, df[col])

            # Interpolate obvious outliers
            df[col] = df[col].interpolate(method='linear')

            # Find dummy pred
            df['dummy_ma'] = (
                df[col]
                # .rolling(10, min_periods=1)
                .ewm(span=10, adjust=False, min_periods=1)
                .mean()
                .bfill()
            )

            # Find dummy residual
            df['dummy_resid'] = df[col] - df['dummy_ma']

            # Find bandwiths
            resid_std = df['dummy_resid'].std()

            df['lower_bound'] = df['dummy_ma'] - 7.0 * resid_std
            df['upper_bound'] = df['dummy_ma'] + 7.0 * resid_std

            # Find outliers_mask
            outliers_mask = np.logical_or(
                df[col] < df['lower_bound'],
                df[col] > df['upper_bound']
            )

            # Remove outliers
            df[col] = np.where(outliers_mask, np.nan, df[col])

            # Interpolate obvious outliers
            df[col] = df[col].interpolate(method='linear')

            # Remove unnecessary columns
            df.drop(columns=['dummy_ma', 'dummy_resid', 'lower_bound', 'upper_bound'], inplace=True)

        """
        Correct returns, accelerations & jerks
        """
        # Re-create coin_return, coin_acceleration & coin_jerk
        if 'coin_return' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['coin_return'] = df['coin_price'].pct_change() # fill_method=None).bfill()
        if 'coin_acceleration' in df.columns:
            df['coin_acceleration'] = df['coin_return'].diff()
        if 'coin_jerk' in df.columns:
            df['coin_jerk'] = df['coin_acceleration'].diff()

        # Re-create target_return, coin_return & coin_acceleration
        if 'target_return' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['target_return'] = df['target_price'].pct_change() # fill_method=None).bfill()
        if 'target_acceleration' in df.columns:
            df['target_acceleration'] = df['target_return'].diff()
        if 'target_jerk' in df.columns:
            df['target_jerk'] = df['target_acceleration'].diff()

        """
        Correct open_price diff
        """
        if 'coin_open' in df.columns:
            # Find prev_coin_price & prev_price_open_diff
            df['prev_coin_price'] = df['coin_price'].shift(1)
            df['prev_price_open_diff'] = (df['coin_open'] - df['prev_coin_price']).abs()

            # Find extreme outliers diff_mask
            outliers_diff_mask = (
                df['prev_price_open_diff'] > 0.04 * df['coin_price']
            )

            # Correct extreme outliers in diff
            df.loc[outliers_diff_mask, 'coin_open'] = df.loc[outliers_diff_mask, 'prev_coin_price']

            # Delete added columns
            df.drop(columns=['prev_coin_price', 'prev_price_open_diff'], inplace=True)

        """
        Correct rest of outliers
        """
        for col, thresholds in self.outliers_dict.items():
            # Find low outliers mask
            low_outliers_mask = df[col] < thresholds[0]

            # Correct low outliers
            df.loc[low_outliers_mask, col] = thresholds[0]

            # Find high outliers mask
            high_outliers_mask = df[col] > thresholds[1]

            # Replace high outliers
            df.loc[high_outliers_mask, col] = thresholds[1]

        return df

    def update_imputers(
        self,
        df: pd.DataFrame
    ) -> None:
        # Update str_imputer
        str_cols = list(df.select_dtypes(include=['object', 'category']).columns)

        self.str_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        if len(str_cols) > 0:
            self.str_imputer.fit(df[str_cols])

        # Update num_imputer
        num_cols = list(df.select_dtypes(include=['number']).columns)

        self.num_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.num_imputer.fit(df[num_cols])
        
        # Update MissForest Imputer
        # self.mf_imputer = MissForest(
        #     criterion='squared_error',
        #     max_features='sqrt',
        #     # max_features=1.0,
        #     n_estimators=90,
        #     max_iter=8
        # )
        # self.mf_imputer.fit(df)
    
    def fill_nulls(
        self,
        df: pd.DataFrame, 
        str_imputer: SimpleImputer = None,
        num_imputer: SimpleImputer = None,
        coin_name: str = None,
        df_name: str = None,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Validate Inputs
        """
        if str_imputer is None:
            str_imputer = self.str_imputer

        if num_imputer is None:
            num_imputer = self.num_imputer

        if coin_name is None:
            coin_name = self.coin_name

        if df_name is None:
            df_name = 'cleaned_data'

        """
        Fill null values
        """
        warnings.filterwarnings("ignore")
    
        if debug:
            print(f"Filling null values in {coin_name} {df_name}.\n")
            print(f'{df_name}.shape: {df.shape}\n'
                  f'{df_name}.index: \n{df.index}\n\n')

        last_obs = df.index[-1]

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

        category_columns = {cat: [c for c in df.columns if c.startswith(cat)] for cat in category_names}

        # Handle Null values
        #   - drop the NaN and zero columns
        # df.dropna(axis='columns', how="all", inplace=True)
        # df = df.loc[:, (df != 0).any(axis=0)]

        # Ffill Columns
        ffill_cols = [
            'target_price',
            'coin_price',
            'coin_open',
            'coin_high',
            'coin_low',
            'ta_volume',
        ]

        # Add Other Coins & Stock Cols
        ffill_cols.extend(category_columns['other_coins'])
        ffill_cols.extend(category_columns['stock'])

        ffill_cols = list(filter(lambda col: col in df.columns, ffill_cols))
        
        if len(ffill_cols) > 0 and df[ffill_cols].isnull().sum().sum() > 0:
            df[ffill_cols] = (
                df[ffill_cols]
                .ffill()
                .bfill()
            )
        
        # Sentiment Columns
        if len(category_columns['sentiment_lc']) > 0:
            for col in category_columns['sentiment_lc']:
                if df[[col]].isnull().sum().sum() > 0:
                    dummy = df[col] # [category_columns['sentiment_lc']]

                    first_valid_idx = dummy.first_valid_index()
                    if first_valid_idx is None:
                        first_valid_idx = dummy.index[-1]

                    if dummy.index[0] < first_valid_idx: # and dummy.shape[0] > self.new_n * 10:
                        mean_value = np.mean(self.outliers_dict[col])
                        # scaler = SimpleImputer(strategy='median', keep_empty_features=True)
                        # scaler.fit(dummy)

                        dummy.loc[
                            dummy.index < first_valid_idx # , col # category_columns['sentiment_lc']
                        ] = mean_value
                        # scaler.transform(
                        #     dummy.loc[
                        #         dummy.index < first_valid_idx # , col # category_columns['sentiment_lc']
                        #     ]
                        # )

                    dummy = dummy.bfill().ffill()
                    df[col] = dummy # df[category_columns['sentiment_lc']] = dummy

        # Replace inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Interpolate num cols
        num_cols = list(df.select_dtypes(include=['number']).columns)
        df[num_cols] = df[num_cols].interpolate(method='linear')
        
        # Fill the rest
        if df.isnull().sum().sum() > 0:
            str_cols = list(df.select_dtypes(include=['object', 'category']).columns)
            if len(str_cols) > 0:
                df[str_cols] = str_imputer.transform(df[str_cols])

            if len(num_cols) > 0:
                df[num_cols] = num_imputer.transform(df[num_cols])

        # df = pd.DataFrame(
        #     mf_imputer.transform(df),
        #     columns=df.columns.tolist(),
        #     index=df.index.tolist()
        # )

        assert df.isnull().sum().sum() == 0

        if debug:
            print(f'cleaned_data.shape: {df.shape}\n')
            # msno.matrix(df)
            # plt.show()

        return df.loc[:last_obs]
    
    @staticmethod
    def correct_highs_and_lows(
        df: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        # Correct highs
        # df['coin_high'] = np.maximum(df['coin_high'], df['coin_price'], df['coin_open'])
        df['coin_high'] = df[['coin_high', 'coin_price', 'coin_open']].max(axis=1)

        if debug:
            wrong_highs = df.loc[
                df['coin_high'] < np.maximum(df['coin_price'].copy(), df['coin_open'].copy()), 
                ['coin_price', 'coin_open', 'coin_high']
            ]
            print(f"wrong highs:\n{wrong_highs}\n\n")
        
        # Correct lows
        # df['coin_low'] = np.minimum(df['coin_low'], df['coin_price'], df['coin_open'])
        df['coin_low'] = df[['coin_low', 'coin_price', 'coin_open']].min(axis=1)

        if debug:
            wrong_lows = df.loc[
                df['coin_low'] > np.minimum(df['coin_price'].copy(), df['coin_open'].copy()), 
                ['coin_price', 'coin_open', 'coin_low']
            ]
            print(f"wrong lows:\n{wrong_lows}\n\n")
        
        return df

    # @timing
    def cleaner_pipeline(
        self,
        df: pd.DataFrame,
        unused_data: pd.DataFrame = None,
        remove_unexpected_neg_values: bool = True,
        non_neg_cols: list = None,
        remove_inconsistent_prices: bool = True,
        handle_rows_and_columns: bool = True,
        expected_cols: list = None,
        new_data: pd.DataFrame = None,
        remove_outliers: bool = True,
        update_outliers_dict: bool = False,
        z_threshold: float = None,
        impute_nulls: bool = True,
        update_imputers: bool = False,
        validate_data: bool = True,
        save_mock: bool = False,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        # Save mock input df
        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='cleaner_pipeline_input'
            )

        # Check Missing New Data
        # if has_missing_new_data(
        #     df=df.copy(),
        #     intervals=self.intervals,
        #     coin_name=self.coin_name,
        #     df_name='raw_data'
        # ):
        #     repair_missing_new_data(self.coin_name, self.intervals)

        # Shorten raw_data
        # df = df.iloc[-self.periods-self.lag:]
        
        # Concat raw_data & unused_data
        keep_idx = df.index.copy()
        if unused_data is not None:
            df = (
                pd.concat([df, unused_data], axis=0)
                .sort_index(ascending=True)
                .copy()
            )
        
        # Find price columns
        self.find_price_columns(df=df)
        
        # Remove Unexpected Negative Values
        if remove_unexpected_neg_values:
            df = self.remove_unexpected_neg_values(
                df=df,
                non_neg_cols=non_neg_cols
            )
        
        # Remove Inconsistent Prices
        if remove_inconsistent_prices:
            df = self.remove_inconsistent_prices(df=df)
        
        # Handle Rows & Columns
        if handle_rows_and_columns:
            # Drop Duplicate Rows & Cols
            df = drop_duplicates(df=df)

            # Add Missig Rows with np.nan
            df = self.add_missing_rows(df=df)

            # Handle Missing Cols
            df = self.handle_expected_cols(
                df=df,
                expected_cols=expected_cols,
                new_data=new_data
            )
        
        # Remove outliers
        if remove_outliers:
            # Update outliers attrs
            if update_outliers_dict:
                self.update_outliers_dict(
                    df=df.copy(),
                    z_threshold=z_threshold
                )
            
            # Correct outliers
            df = self.correct_outliers(
                df=df,
                z_threshold=z_threshold
            )
        
        # Impute nulls
        if impute_nulls:
            # Update imputers & outliers_dict
            is_fitted = True
            try:
                check_is_fitted(self.str_imputer)
                check_is_fitted(self.num_imputer)
            except:
                is_fitted = False

            if (
                self.str_imputer is None or 
                self.num_imputer is None or 
                not is_fitted or
                update_imputers
            ):
                self.update_imputers(df=df.copy())

            # Fill Missing Values
            df = self.fill_nulls(
                df=df,
                debug=debug
            )
        
        # Correct coin_high & coin_low
        df = self.correct_highs_and_lows(
            df=df,
            debug=debug
        )
        
        # Correct DF Columns Order
        # df = self.correct_columns_order(
        #     df=df.copy()
        # )

        # Keep only needed index
        if keep_idx is not None:
            df = (
                df.loc[df.index.isin(keep_idx)]
                .sort_index(ascending=True)
            )

        # Validate data
        if validate_data:
            df = self.validate_data(
                df=df.copy(),
                repair=True,
                keep_idx=keep_idx,
                debug=debug,
                **update_expectations
            )
        
        # Save mock output df
        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='cleaner_pipeline_output'
            )
        
        return df.tail(self.periods)

    def correct_columns_order(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        ordered_cols: List[str] = Params.fixed_params.get("raw_data_columns").copy()
        for coin in self.other_coins:
            # if coin != self.coin_name:
            ordered_cols.append(f'other_coins_{coin}_price')
        ordered_cols = [c for c in ordered_cols if c in df.columns]
        return df.loc[:, ordered_cols]

    def correct_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Repair Coin Cols
        df = repair_coin_cols(
            df=df.copy(),
            other_coins=self.other_coins, # [c for c in self.other_coins if c != self.coin_name],
            intervals=self.intervals,
            price_only=True
        )
        
        # Repair Stock & On-Chain Cols
        df = repair_stock_and_on_chain_cols(
            df=df.copy(),
            intervals=self.intervals
        )

        # Correct column order
        df = self.correct_columns_order(
            df=df.copy()
        )

        return df

    # @timing
    def update(
        self,
        raw_data: pd.DataFrame,
        unused_data: pd.DataFrame = None,
        debug: bool = False,
        **update_params: dict
    ) -> None:
        # Set Up Update Parameters
        complete_update_params = {
            'update_data': False,
            'rewrite_data': False,
            'update_expectations': False,
            'update_outliers_dict': False,
            'update_imputers': False,
            'validate_data': False,
            'save': False
        }
        for k, v in complete_update_params.items():
            if k not in update_params.keys():
                update_params[k] = v

        # Update Expectations
        if update_params['update_expectations']:
            self.update_expectations(debug=debug)

        # Update Raw Data (and Unused Data)
        if self.cleaned_data is None or update_params['rewrite_data']:
            if update_params['rewrite_data']:
                LOGGER.warning('%s (%s) self.cleaned_data will be re-written.', self.coin_name, self.intervals)
            else:
                LOGGER.warning('%s (%s): self.cleaned_data is None, thus it will be re-written.', self.coin_name, self.intervals)

            assert raw_data.shape[0] >= self.periods, f"{self.coin_name} raw data inputed to the DC is too short ({raw_data.shape} periods).\n\n"
            
            self.cleaned_data: pd.DataFrame = self.cleaner_pipeline(
                df=raw_data.copy(),
                unused_data=unused_data,
                remove_unexpected_neg_values=True,
                non_neg_cols=None,
                remove_inconsistent_prices=True,
                handle_rows_and_columns=True,
                expected_cols=None,
                new_data=None,
                remove_outliers=True,
                update_outliers_dict=update_params['update_outliers_dict'],
                z_threshold=self.z_threshold,
                impute_nulls=True,
                update_imputers=update_params['update_imputers'],
                validate_data=update_params['validate_data'],
                debug=debug
            )
            
            LOGGER.info('self.cleaned_data.shape: %s', self.cleaned_data.shape)
        elif raw_data.index[-1] > self.cleaned_data.index[-1] and update_params['update_data']:
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_periods = int((utc_now - self.cleaned_data.index[-1]).seconds / (60 * self.mbp)) + self.new_n
            new_raw_data = raw_data.tail(new_periods)

            new_cleaned_data: pd.DataFrame = self.cleaner_pipeline(
                df=new_raw_data.copy(),
                unused_data=unused_data,
                remove_unexpected_neg_values=True,
                non_neg_cols=None,
                remove_inconsistent_prices=True,
                handle_rows_and_columns=True,
                expected_cols=None,
                new_data=None,
                remove_outliers=True,
                update_outliers_dict=False,
                z_threshold=self.z_threshold,
                impute_nulls=True,
                update_imputers=False,
                validate_data=update_params['validate_data'],
                debug=debug,
                **{
                    'expected_periods': new_periods
                }
            )
            col_diff = list(set(new_cleaned_data.columns).symmetric_difference(set(self.cleaned_data.columns)))
            if len(col_diff) > 0:                
                LOGGER.warning(
                    'self.cleaned_data and new_cleaned_data have different columns (%s).\n'
                    'new_cleaned_data.columns: %s\n'
                    'self.cleaned_data.columns: %s\n'
                    'col_diff: %s\n',
                    self.coin_name, new_cleaned_data.columns, self.cleaned_data.columns, col_diff
                )

                self.cleaned_data = self.correct_columns(
                    df=self.cleaned_data.copy()
                )

            new_cols = [c for c in new_cleaned_data.columns.tolist() if c in self.cleaned_data]

            self.cleaned_data = (
                self.cleaned_data[new_cols].iloc[:-24]
                .combine_first(new_cleaned_data)
                .combine_first(self.cleaned_data[new_cols])
                .sort_index(ascending=True)
            )

            # self.cleaned_data = self.cleaned_data[list(new_cleaned_data.columns)]
            self.cleaned_data = self.cleaned_data.loc[~self.cleaned_data.index.duplicated(keep='last')]

        """
        Validate Data
        """
        if update_params['validate_data']:
            self.cleaned_data = self.validate_data(
                df=self.cleaned_data.copy(),
                unused_data=unused_data,
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )
        
        """
        Save Data Cleaner
        """
        if update_params['save']:
            self.save(debug=debug)

    def update_expectations(
        self,
        debug: bool = False
    ) -> None:
        # Define asset_path
        s3_asset_path = f"{Params.bucket}/data_processing/data_cleaner/{self.intervals}/{self.coin_name}/{self.coin_name}_cleaned_data.parquet"

        # Extract expected cols
        expected_cols: List[str] = Params.fixed_params.get("raw_data_columns").copy()
        for coin in self.other_coins:
            # if coin != self.coin_name:
            expected_cols.append(f'other_coins_{coin}_price')

        # Define expected schema
        expected_schema = {
            col: 'float' for col in expected_cols
        }

        # Define max values allowed
        max_values_allowed = {
            'long_short_global_long_perc': 1.0,
            'long_short_top_traders_long_perc': 1.0
        }

        # Define min values allowed
        min_values_allowed = {
            col: 0.0 for col in expected_cols
            if not(col.endswith('return') or col.endswith('acceleration') or col.endswith('jerk'))
        }

        # Unique values allowed
        unique_values_allowed = None

        # Null values percentage allowed            
        null_perc_allowed: Dict[str, float] = {
            col: 0.0 for col in expected_cols
        }

        # Define duplicate rows subset
        duplicate_rows_subset = expected_cols.copy()

        # Expected periods
        expected_periods = Params.data_params.get("periods")
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < expected_periods:
            expected_periods = raw_data_periods

        # Define Expectations
        s3_expectations = {
            "asset_path": s3_asset_path,
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
            "check_extreme_values": True,
            "check_excess_features": True,
            "check_short_length": True,

            "expected_cols": expected_cols,
            "expected_schema": expected_schema,
            "max_values_allowed": max_values_allowed,
            "min_values_allowed": min_values_allowed,
            "unique_values_allowed": unique_values_allowed,
            "null_perc_allowed": null_perc_allowed,

            "duplicate_rows_subset": duplicate_rows_subset,
            "outliers_dict": deepcopy(self.outliers_dict),
            "max_features_perc": Params.data_params.get("max_features"),
            "other_coins": self.other_coins,
            "expected_periods": expected_periods
        }

        if debug:
            LOGGER.debug('cleaned_data expectations:\n%s\n', pformat(s3_expectations))

        # Save Expectations
        s3_expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
        write_to_s3(
            asset=s3_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_cleaned_data_expectations.json"
        )
    
    def diagnose_data(
        self,
        df: pd.DataFrame = None,
        debug: bool = False,
        **update_expectations: dict
    ) -> Dict[str, bool]:
        if df is None:
            df = self.cleaned_data.copy()

        # Find Diagnostics Dict
        diagnostics_dict = find_data_diagnosis_dict(
            df_name="cleaned_data",
            intervals=self.intervals,
            coin_name=self.coin_name,
            df=df,
            debug=debug,
            **update_expectations
        )

        if debug:
            print(f'{self.coin_name} cleaned_data diagnostics_dict:')
            pprint(diagnostics_dict)
            print('\n\n')

        return diagnostics_dict        
    
    def validate_data(
        self,
        df: pd.DataFrame = None,
        unused_data: pd.DataFrame = None,
        repair: bool = True,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        if df is None:
            df = self.cleaned_data.copy()

        # Find Diagnostics Dict
        diagnostics_dict = self.diagnose_data(
            df=df,
            debug=debug,
            **update_expectations
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                "%s cleaned_data needs repair.\n"
                "diagnostics_dict:\n%s\n",
                self.coin_name, pformat(diagnostics_dict)
            )

            if repair:
                LOGGER.info("Repairing %s cleaned_data...", self.coin_name)
                """
                Diagnostics Dict:
                    - has_missing_new_data
                    - has_missing_columns
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
                # File System
                # expectations: dict = json.load(open(os.path.join(
                #     Params.base_cwd, Params.bucket, "utils", "expectations", self.intervals, self.coin_name, 
                #     f"{self.coin_name}_cleaned_data_expectations.json"
                # )))

                # S3
                s3_expectations_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
                expectations: dict = load_from_s3(
                    path=f"{s3_expectations_path}/{self.coin_name}_cleaned_data_expectations.json"
                )

                df: pd.DataFrame = self.cleaner_pipeline(
                    df=df,
                    unused_data=unused_data,
                    remove_unexpected_neg_values=True,
                    non_neg_cols=None,
                    remove_inconsistent_prices=True,
                    handle_rows_and_columns=True,
                    expected_cols=expectations.get("expected_cols"),
                    new_data=None,
                    remove_outliers=True,
                    update_outliers_dict=False,
                    z_threshold=self.z_threshold,
                    impute_nulls=True,
                    update_imputers=False,
                    validate_data=False,
                    debug=debug
                )
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
        base_path = f"{Params.bucket}/mock/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"

        # Define save_path
        if asset_name == 'cleaner_pipeline_input':
            save_path = f"{base_path}/cleaner_pipeline_input.parquet"
        elif asset_name == 'cleaner_pipeline_output':
            save_path = f"{base_path}/cleaner_pipeline_output.parquet"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Save asset to S3
        write_to_s3(asset=asset, path=save_path, overwrite=True)
    
    def load_mock_asset(
        self,
        asset_name: str,
        re_create: bool = False,
        re_create_periods: int = None
    ) -> pd.DataFrame:
        # Define base_path
        re_create_base_path = f"{Params.bucket}/data_processing/data_extractor/{self.intervals}/{self.coin_name}"
        base_path = f"{Params.bucket}/mock/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"

        # Define load_path
        if asset_name == 'cleaner_pipeline_input':
            if re_create:
                load_path = f"{re_create_base_path}/{self.coin_name}_raw_data.parquet"
            else:
                load_path = f"{base_path}/cleaner_pipeline_input.parquet"
        elif asset_name == 'cleaner_pipeline_output':
            load_path = f"{base_path}/cleaner_pipeline_output.parquet"
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
        pickle_attrs = {key: value for key, value in self.__dict__.items() if key in self.load_pickle}
        
        # Write pickled attrs
        write_to_s3(
            asset=pickle_attrs,
            path=f"{self.save_path}/{self.coin_name}_data_cleaner_attr.pickle"
        )
        
        if debug:
            for attr_key, attr_value in pickle_attrs.items():            
                print(f'Saved pickle {attr_key}:')
                pprint(attr_value)
                print('\n')
        
        """
        Save cleaned_data
        """
        write_to_s3(
            asset=self.cleaned_data,
            path=f"{self.save_path}/{self.coin_name}_cleaned_data.parquet",
            overwrite=self.overwrite
        )
    
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
                path=f"{self.save_path}/{self.coin_name}_data_cleaner_attr.pickle"
            )

            for attr_key, attr_value in pickle_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

                    if debug:
                        print(f'Loaded pickle {attr_key}:')
                        pprint(attr_value)
                        print('\n')
        except Exception as e:
            LOGGER.error(
                'Unable to load data_cleaner pickle_attrs (%s: %s).\n'
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
        
        try:
            self.cleaned_data: pd.DataFrame = load_from_s3(
                path=f"{self.save_path}/{self.coin_name}_cleaned_data.parquet",
                load_reduced_dataset=load_reduced_dataset
            ).iloc[-self.periods:]

            # Update periods if required
            if load_reduced_dataset:
                self.periods = self.cleaned_data.shape[0]
        except Exception as e:
            LOGGER.error(
                'Unable to load cleaned_data: %s.\n'
                'Exception: %s\n', 
                self.coin_name, e
            )

    