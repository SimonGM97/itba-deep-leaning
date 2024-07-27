from PyTradeX.config.params import Params
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair
)
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any
from pprint import pprint, pformat


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


class DataShifter(DataCleaner):

    load_parquet = [
        'cleaned_data_shift'
    ]
    load_pickle = [
        'consistency_storage',
        'abs_diff_dfs',
        'shift_dict'
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
        self.new_n: int = None
        self.save_distance: int = None
        self.compare_distance: int = None
        self.shift_threshold: float = None
        self.max_shift: int = None
        self.mbp: int = None

        default_params = [
            'lag',
            'periods',
            'new_n',
            'save_distance',
            'compare_distance',
            'shift_threshold',
            'max_shift',
            'mbp'
        ]
        for param in default_params:
            if param not in data_params.keys():
                data_params[param] = getattr(Params, 'data_params')[param]

            setattr(self, param, data_params[param])

        # Update periods
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < self.periods:
            # LOGGER.warning(
            #     '%s DS has less periods than required.\n'
            #     'Expected periods: %s\n'
            #     'raw_data_periods: %s\n'
            #     'Thus, self.periods will be reverted to %s.\n',
            #     self.coin_name, self.periods, raw_data_periods, raw_data_periods
            # )
            self.periods: int = raw_data_periods

        # Load params
        self.cleaned_data_shift: pd.DataFrame = None

        self.consistency_storage: List[pd.DataFrame] = None
        self.abs_diff_dfs: List[pd.DataFrame] = None
        self.shift_dict: Dict[str, float] = None

        self.load(debug=False)
    
    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/data_processing/data_shifter/{self.intervals}/{self.coin_name}"
        else:
            return f"{Params.bucket}/data_processing/data_shifter/{self.intervals}/{self.coin_name}"

    def sort_consistency_storage(self) -> None:
        def sort_dates_fun(df: pd.DataFrame):
            ts: pd.Timestamp = df.index[-1]
            return ts.year, ts.month, ts.day, ts.minute
        
        # Sort DataFrames
        self.consistency_storage.sort(key=sort_dates_fun, reverse=False)

    def update_consistency_storage(
        self,
        df: pd.DataFrame,
        reset_consistency_storage: bool = False,
        debug: bool = False
    ) -> None:
        # Reset consistency (if needed)
        if self.consistency_storage is None or reset_consistency_storage:
            LOGGER.warning('Re-Setting self.consistency_storage (%s).', self.coin_name)
            self.consistency_storage = []

        # Filter out old dfs
        last_valid_time = datetime.now(timezone.utc).replace(tzinfo=None) - pd.Timedelta(minutes=self.mbp * self.save_distance * 2)
        self.consistency_storage = list(filter(
            lambda df_: df_.index[-1] > last_valid_time, 
            self.consistency_storage
        ))

        # Assert that all old dataframes have been filtered out
        assert all([df_.index[-1] >= last_valid_time for df_ in self.consistency_storage])

        # Sort consistency_storage
        self.sort_consistency_storage()

        # Append New DF
        append_df = df.iloc[-150:].copy()
        saved_idxs = [df_.index[-1] for df_ in self.consistency_storage]

        if debug:
            LOGGER.debug('First initial saved_idxs:\n%s\n', pformat(saved_idxs[:5]))
            LOGGER.debug('First last saved_idxs:\n%s\n', pformat(saved_idxs[-5:]))

        # Remove excess DataFrames
        if append_df.index[-1] not in saved_idxs:            
            self.consistency_storage = self.consistency_storage + [append_df]
            self.consistency_storage = self.consistency_storage[-self.save_distance + 1:]

        if debug:
            LOGGER.debug('Final initial saved_idxs:\n%s\n', pformat(saved_idxs[:5]))
            LOGGER.debug('Final last saved_idxs:\n%s\n', pformat(saved_idxs[-5:]))

    def update_diff_dfs(
        self,
        debug: bool = False
    ) -> None:
        # Reset abs_diff_dfs & Find the list of percent comparison DataFrames
        LOGGER.warning('Re-Setting self.abs_diff_dfs %s.', self.coin_name)
        self.abs_diff_dfs: List[pd.DataFrame] = []

        # Sort consistency_storage
        self.sort_consistency_storage()

        # Extract saved idxs
        saved_idxs = [df_.index[-1] for df_ in self.consistency_storage]

        if debug:
            LOGGER.debug('Initial saved_idxs:\n%s\n', pformat(saved_idxs[:5]))
            LOGGER.debug('Last saved_idxs:\n%s\n', pformat(saved_idxs[-5:]))

        show_cols = None

        for n in range(len(self.consistency_storage)):
            # Find old df
            old_df = self.consistency_storage[n].copy()
            
            # Find New df
            search_idx = old_df.index[-1] + pd.Timedelta(minutes=self.mbp * self.compare_distance)
            
            if search_idx in saved_idxs:
                # print(f'found {search_idx}')
                new_df = next(df_ for df_ in self.consistency_storage if df_.index[-1] == search_idx).copy()

                # Prepare Dataframes for Comparison
                old_num_cols = list(old_df.select_dtypes(include=['number']).columns)
                new_num_cols = list(new_df.select_dtypes(include=['number']).columns)
                num_cols = list(set(old_num_cols).intersection(set(new_num_cols)))
                idx_itersect = old_df.index.intersection(new_df.index)

                daily_periods = {
                    '60min': 24,
                    '30min': 48
                }.get(self.intervals)

                old_df = old_df.loc[old_df.index.isin(idx_itersect), num_cols].tail(daily_periods)
                new_df = new_df.loc[new_df.index.isin(idx_itersect), num_cols].tail(daily_periods)

                # Add diff comparison
                abs_diff_df = (old_df.subtract(new_df)).abs().reset_index(drop=True)

                if debug:
                    if show_cols is None:
                        show_cols = [c for c in abs_diff_df.columns if abs_diff_df[c].iat[-1] > 0][:5]

                    LOGGER.debug('old_df:\n%s\n', old_df[show_cols].tail(5))
                    LOGGER.debug('new_df (assumed source of truth):\n%s\n', new_df[show_cols].tail(5))
                    LOGGER.debug('abs_diff_df:\n%s\n', abs_diff_df[show_cols].tail(5))
                
                self.abs_diff_dfs.append(abs_diff_df)

    def update_shift_dict(
        self,
        df: pd.DataFrame,
        debug: bool = False
    ) -> None:
        # Calculate mean_abs_diff_df
        mean_abs_diff_df = pd.concat(self.abs_diff_dfs).groupby(level=0).mean()

        if debug:
            LOGGER.debug('mean_abs_diff_df:\n%s\n', mean_abs_diff_df.tail())
        
        # Divide by the mean abs ret
        for col in mean_abs_diff_df.columns:
            mean_abs_diff_df[col] = mean_abs_diff_df[col] / df[col].abs().mean() # .pct_change()
        
        if debug:
            LOGGER.debug('perc mean_abs_diff_df:\n%s\n', mean_abs_diff_df.tail())
        
        # Update self.shift_dict
        def find_shift_n(col: str, threshold: float = 0.1):
            index = next((i for i, x in enumerate(mean_abs_diff_df[col]) if x > threshold), None)
            if index is not None:
                return mean_abs_diff_df.shape[0] - index
            return 0
        
        LOGGER.warning('Re-Setting self.shift_dict (%s).', self.coin_name)

        self.shift_dict = {
            col: find_shift_n(col, self.shift_threshold) for col in mean_abs_diff_df.columns
        }

        # Filter out 0 observations
        self.shift_dict = {k: v for k, v in self.shift_dict.items() if v > 0}

        LOGGER.info('self.shift_dict:\n%s\n', pformat(self.shift_dict))

    def shift_df(
        self,
        df: pd.DataFrame,
        accelerated: bool = False,
        category_features: Dict[str, List[str]] = None
    ) -> pd.DataFrame:
        # Populate required features
        if accelerated:
            required_features: List[str] = []
            for _, features in category_features.items():
                required_features.extend(features)
        else:
            required_features: List[str] = None

        if len(self.shift_dict.keys()) > 0:
            # Loop over columns to shift
            for col, shift_n in self.shift_dict.items():
                if accelerated and col not in required_features:
                    continue

                if shift_n > self.max_shift:
                    # LOGGER.warning(
                    #     '%s has a high self.shift_dict: %s.\n'
                    #     'Max allowed: %s.\n',
                    #     col, shift_n, self.max_shift
                    # )
                    df[col] = df[col].shift(self.max_shift)
                else:
                    df[col] = df[col].shift(shift_n)
            
            df.bfill(inplace=True)
        
        return df

    def placeholder(self) -> None:
        # Define shift_dict
        if self.intervals == '30min':
            self.shift_dict = {
                # 'economic': 1,
                'on_chain_transaction_rate_per_second': 2,
                'sentiment_lc_circulating_supply': 5, #
                'sentiment_lc_galaxy_score': 9,
                'sentiment_lc_volatility': 5, #
                'sentiment_lc_alt_rank': 3,
                # 'sentiment_lc_market_dominance': 5,
                'sentiment_lc_contributors_active': 5, #
                'sentiment_lc_posts_active': 5, #
                'sentiment_lc_interactions': 5, #
                'sentiment_lc_social_dominance': 5, #
            }
        else:
            self.shift_dict = {
                # 'economic': 1,
                'on_chain_transaction_rate_per_second': 1,
                'sentiment_lc_circulating_supply': 3, #
                'sentiment_lc_galaxy_score': 5,
                'sentiment_lc_volatility': 3, #
                'sentiment_lc_alt_rank': 2,
                # 'sentiment_lc_market_dominance': 3,
                'sentiment_lc_contributors_active': 3, #
                'sentiment_lc_posts_active': 3, #
                'sentiment_lc_interactions': 3, #
                'sentiment_lc_social_dominance': 3, #
            }

    # @timing
    def shifter_pipeline(
        self,
        df: pd.DataFrame,
        record_df: bool = True,
        reset_consistency_storage: bool = False,
        update_shift_dict: bool = False,
        placeholder: bool = True,
        validate_data: bool = True,
        accelerated: bool = False,
        category_features: Dict[str, List[str]] = None,
        save_mock: bool = False,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        # Save mock input df
        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='shifter_pipeline_input'
            )

        # Record new DataFrame
        if record_df:
            self.update_consistency_storage(
                df=df.copy(),
                reset_consistency_storage=reset_consistency_storage,
                debug=debug
            )
        
        # Update shift dict
        if placeholder:
            self.placeholder()
        else:
            if update_shift_dict:
                # Update diff_dfs
                self.update_diff_dfs(debug=debug)
                
                # Update shift_dict
                self.update_shift_dict(
                    df=df.copy(),
                    debug=debug
                )

        # Shift DataFrame
        df = self.shift_df(
            df=df,
            accelerated=accelerated,
            category_features=category_features
        )

        # Validate Data
        if validate_data:
            df = self.validate_data(
                df=df.copy(),
                repair=True,
                debug=False,
                **update_expectations
            )
        
        # Save mock output df
        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='shifter_pipeline_output'
            )

        return df.tail(self.periods)

    # @timing
    def update(
        self,
        cleaned_data: pd.DataFrame,
        debug: bool = False,
        **update_params
    ) -> None:
        # Set Up Update Parameters
        complete_update_params = {
            'update_data': False,
            'rewrite_data': False,
            'update_expectations': False,
            'update_shift_dict': False,
            'record_df': False,
            'validate_data': False,
            'save': False
        }
        for k, v in complete_update_params.items():
            if k not in update_params.keys():
                update_params[k] = v

        # Update Expectations
        if update_params['update_expectations']:
            self.update_expectations(debug=debug)

        # Update Shifted Data
        if self.cleaned_data_shift is None or update_params['rewrite_data']:
            if update_params['rewrite_data']:
                LOGGER.warning('%s (%s) self.cleaned_data_shift will be re-written.', self.coin_name, self.intervals)
            else:
                LOGGER.warning(
                    '%s (%s): self.cleaned_data_shift is None, thus it will be re-written.\n'
                    'self.cleaned_data_shift: %s\n',
                    self.coin_name, self.intervals, self.cleaned_data_shift
                )
                
            self.cleaned_data_shift: pd.DataFrame = self.shifter_pipeline(
                df=cleaned_data.copy(),
                record_df=update_params['record_df'],
                reset_consistency_storage=False,
                update_shift_dict=update_params['update_shift_dict'],
                placeholder=True,
                validate_data=update_params['validate_data'],
                debug=debug
            )

            LOGGER.info('self.cleaned_data_shift.shape: %s', self.cleaned_data_shift.shape)
        elif cleaned_data.index[-1] > self.cleaned_data_shift.index[-1] and update_params['update_data']:
            # Clean Raw Data (using unused_data to interpolate if necessary!)
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_periods = int((utc_now - self.cleaned_data_shift.index[-1]).seconds / (60 * self.mbp)) + self.new_n
            new_cleaned_data = cleaned_data.iloc[-new_periods:]

            new_cleaned_data_shift: pd.DataFrame = self.shifter_pipeline(
                df=new_cleaned_data.copy(),
                record_df=update_params['record_df'],
                reset_consistency_storage=False,
                update_shift_dict=False,
                placeholder=True,
                validate_data=update_params['validate_data'],
                debug=debug,
                **{
                    'expected_periods': new_periods
                }
            )

            col_diff = list(set(new_cleaned_data_shift.columns).symmetric_difference(set(self.cleaned_data_shift.columns)))
            if len(col_diff) > 0:                
                LOGGER.warning(
                    'self.cleaned_data_shift and new_cleaned_data_shift have different columns (%s).\n'
                    'new_cleaned_data_shift.columns: %s\n'
                    'self.cleaned_data_shift.columns: %s\n'
                    'col_diff: %s\n',
                    self.coin_name, new_cleaned_data_shift.columns, self.cleaned_data_shift.columns, col_diff
                )

                self.cleaned_data_shift = self.correct_columns(
                    df=self.cleaned_data_shift.copy()
                )

            new_cols = [c for c in new_cleaned_data_shift.columns.tolist() if c in self.cleaned_data_shift]

            self.cleaned_data_shift = (
                self.cleaned_data_shift[new_cols].iloc[:-24]
                .combine_first(new_cleaned_data_shift)
                .combine_first(self.cleaned_data_shift[new_cols])
                .sort_index(ascending=True)
            )
            
            self.cleaned_data_shift = self.cleaned_data_shift.loc[~self.cleaned_data_shift.index.duplicated(keep='last')]

        """
        Validate Data
        """
        if update_params['validate_data']:
            self.cleaned_data_shift = self.validate_data(
                df=self.cleaned_data_shift.copy(),
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )

        """
        Save Data Shifter
        """
        if update_params['save']:
            self.save(debug=debug)

    def update_expectations(
        self,
        debug: bool = False
    ) -> None:
        # Define asset_path
        s3_asset_path = f"{Params.bucket}/data_processing/data_shifter/{self.intervals}/{self.coin_name}_cleaned_data_shift.parquet"

        # Find Other Coins
        other_coins_n = Params.data_params.get("other_coins_n")
        other_coins = Params.other_coins_json[self.intervals][:other_coins_n]

        # Extract expected cols
        expected_cols: List[str] = Params.fixed_params.get("raw_data_columns").copy()
        for coin in other_coins:
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
        
        # Load DataCleaner outliers_dict
        base_data_cleaner_path = f"{Params.bucket}/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"
        pickle_attrs: dict = load_from_s3(
            path=f"{base_data_cleaner_path}/{self.coin_name}_data_cleaner_attr.pickle"
        )

        outliers_dict = pickle_attrs.get("outliers_dict")

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
            "outliers_dict": outliers_dict,
            "max_features_perc": Params.data_params.get("max_features"),
            "other_coins": other_coins,
            "expected_periods": expected_periods
        }

        if debug:
            LOGGER.debug('cleaned_data_shift expectations:\n%s\n', pformat(s3_expectations))
        
        # Save Expectations
        s3_expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
        write_to_s3(
            asset=s3_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_cleaned_data_shift_expectations.json"
        )

    def diagnose_data(
        self,
        df: pd.DataFrame = None,
        debug: bool = False,
        **update_expectations: dict
    ) -> Dict[str, bool]:
        if df is None:
            df = self.cleaned_data_shift.copy()

        # Find Diagnostics Dict
        diagnostics_dict = find_data_diagnosis_dict(
            df_name="cleaned_data_shift",
            intervals=self.intervals,
            coin_name=self.coin_name,
            df=df,
            debug=debug,
            **update_expectations
        )

        if debug:
            print(f'{self.coin_name} cleaned_data_shift diagnostics_dict:')
            pprint(diagnostics_dict)
            print('\n\n')

        return diagnostics_dict        

    def validate_data(
        self,
        df: pd.DataFrame = None,
        repair: bool = True,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        if df is None:
            df = self.cleaned_data_shift.copy()

        # Find Diagnostics Dict
        diagnostics_dict = self.diagnose_data(
            df=df,
            debug=debug,
            **update_expectations
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                "%s cleaned_data_shift needs repair.\n"
                "diagnostics_dict:\n%s\n", 
                self.coin_name, pformat(diagnostics_dict)
            )

            if repair:
                LOGGER.info("Repairing %s cleaned_data_shift...", self.coin_name)
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
                # Load expectations
                s3_expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
                expectations: dict = load_from_s3(
                    path=f"{s3_expectations_base_path}/{self.coin_name}_cleaned_data_shift_expectations.json"
                )

                DC = DataCleaner(
                    coin_name=self.coin_name,
                    intervals=self.intervals,
                    **Params.data_params.copy()
                )

                self.other_coins = DC.other_coins
                self.outliers_dict = DC.outliers_dict

                self.num_imputer = DC.num_imputer
                self.str_imputer = DC.str_imputer

                self.periods = DC.periods
                self.new_n = DC.new_n

                df = self.cleaner_pipeline(
                    df=df.copy(),
                    remove_unexpected_neg_values=True,
                    non_neg_cols=expectations.get("greater_than_cero_cols"),
                    remove_inconsistent_prices=True,
                    handle_rows_and_columns=True,
                    expected_cols=expectations.get("expected_cols"),
                    new_data=None,
                    remove_outliers=True,
                    update_outliers_dict=False,
                    z_threshold=DC.z_threshold,
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
        base_path = f"{Params.bucket}/mock/data_processing/data_shifter/{self.intervals}/{self.coin_name}"

        # Define save_path
        if asset_name == 'shifter_pipeline_input':
            save_path = f"{base_path}/shifter_pipeline_input.parquet"
        elif asset_name == 'shifter_pipeline_output':
            save_path = f"{base_path}/shifter_pipeline_output.parquet"
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
        # Define base_paths
        re_create_base_path = f"{Params.bucket}/data_processing/data_cleaner/{self.intervals}/{self.coin_name}"
        base_path = f"{Params.bucket}/mock/data_processing/data_shifter/{self.intervals}/{self.coin_name}"

        # Define load_path
        if asset_name == 'shifter_pipeline_input':
            if re_create:
                load_path = f"{re_create_base_path}/{self.coin_name}_cleaned_data.parquet"
            else:
                load_path = f"{base_path}/shifter_pipeline_input.parquet"
        elif asset_name == 'shifter_pipeline_output':
            load_path = f"{base_path}/shifter_pipeline_output.parquet"
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
            path=f"{self.save_path}/{self.coin_name}_data_shifter_attr.pickle"
        )
        
        if debug:
            for attr_key, attr_value in pickle_attrs.items():            
                LOGGER.debug('Saved pickle %s:\n%s\n', attr_key, pformat(attr_value))

        """
        Save .parquet files
        """
        for attr_name in self.load_parquet:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                # Save parquet files
                write_to_s3(
                    asset=df,
                    path=f"{self.save_path}/{self.coin_name}_{attr_name}.parquet",
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
                path=f"{self.save_path}/{self.coin_name}_data_shifter_attr.pickle"
            )

            for attr_key, attr_value in pickle_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

                    if debug:
                        LOGGER.debug('Loaded pickle %s:\n%s\n', attr_key, pformat(attr_value))
        except Exception as e:
            LOGGER.critical(
                'Unable to load data_shifter (%s: %s).\n'
                'Exception: %s\n',
                self.coin_name, self.intervals, e
            )

        """
        Load .parquet files
        """
        if self.overwrite:
            load_reduced_dataset=False
        else:
            load_reduced_dataset=True
        
        try:
            self.cleaned_data_shift: pd.DataFrame = load_from_s3(
                path=f"{self.save_path}/{self.coin_name}_cleaned_data_shift.parquet",
                load_reduced_dataset=load_reduced_dataset
            ).iloc[-self.periods:]

            # Update periods if required
            if load_reduced_dataset:
                self.periods = self.cleaned_data_shift.shape[0]
        except Exception as e:
            LOGGER.critical(
                'Unable to load cleaned_data_shift: %s.\n'
                'Exception: %s\n', self.coin_name, e
            )

    