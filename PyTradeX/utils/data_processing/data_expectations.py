from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import load_from_s3
from PyTradeX.utils.general.logging_helper import get_logger
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pprint import pformat
from typing import Iterable, Dict
import warnings

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


"""
Data Checks
"""
# Missing New Data
def has_missing_new_data(
    df: pd.DataFrame,
    intervals: str,
    coin_name: str = None,
    df_name: str = None
):
    # Check Missing New Data
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    if df.index[-1] < utc_now - pd.Timedelta(minutes=Params.data_params.get("mbp") * 12):
        time_since_last_period = utc_now - df.index[-1]
        LOGGER.warning(
            "%s %s (%s) is missing new data.\n"
            "Time since last period: %s\n",
            coin_name, df_name, intervals, time_since_last_period
        )
        return True
    return False


# Expected Columns
def has_missing_columns(
    df: pd.DataFrame,
    expected_cols: Iterable,
    coin_name: str,
    df_name: str
):
    # difference = set(expected_cols).symmetric_difference(set(df.columns.tolist()))
    difference = set(expected_cols) - set(df.columns.tolist())
    if len(difference) > 0:
        LOGGER.warning(
            "%s %s has missing columns.\n"
            "difference:\n%s\n",
            coin_name, df_name, pformat(difference)
        )
        return True
    return False


def has_unexpected_columns(
    df: pd.DataFrame,
    expected_cols: Iterable,
    coin_name: str,
    df_name: str
):
    # difference = set(expected_cols).symmetric_difference(set(df.columns.tolist()))
    difference = set(df.columns.tolist()) - set(expected_cols)
    if len(difference) > 0:
        LOGGER.warning(
            "%s %s has unexpected columns.\n"
            "difference:\n%s\n",
            coin_name, df_name, pformat(difference)
        )
        return True
    return False


# Missing Rows
def has_missing_rows(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    intervals: str
):
    # Check if there are missing rows
    freq = {
        '30min': '30min',
        '60min': '60min',
        '1d': '1D'
    }[intervals]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    
    if len(list(set(full_idx) - set(df.index))) > 0:
        missing_rows_idx = sorted(list(set(full_idx) - set(df.index)))
        LOGGER.warning(
            '%s %s has %s missing rows.\n'
            'missing_rows_idx: %s\n',
            coin_name, df_name, len(full_idx) - len(df.index), missing_rows_idx
        )
        return True
    return False


# Null values
def has_exess_null_values(
    df: pd.DataFrame, 
    null_perc_allowed: dict = None,
    coin_name: str = None,
    df_name: str = None,    
    debug: bool = False
):
    # Validate parameters
    if null_perc_allowed is None:
        null_perc_allowed = {
            col: 0.0 for col in df.columns.tolist()
        }

    # Collect null values
    null_perc = df.isnull().sum() / df.shape[0]
    
    if debug:
        LOGGER.debug('null_perc:\n%s\n', pformat(null_perc))

    has_exess_null_values_ = False

    for idx in null_perc.index:
        if null_perc[idx] > null_perc_allowed[idx]:
            LOGGER.warning(
                '%s %s "%s" column has too many null values: %s [max allowed: %s].',
                coin_name, df_name, idx, round(null_perc[idx], 3), null_perc_allowed[idx]
            )
            has_exess_null_values_ = True
    return has_exess_null_values_


# Duplicates
def has_duplicated_idx(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str
):
    # Check if there are duplicated idxs
    duplicated_idxs = df.index.duplicated()
    if sum(duplicated_idxs) > 0:
        idx = df.index[df.index.duplicated()]
        LOGGER.warning(
            "%s %s has %s duplicated_idxs.\n"
            "duplicated_idxs:\n%s\n",
            coin_name, df_name, sum(duplicated_idxs), pformat(idx)
        )
        return True
    return False


def has_duplicates_rows(
    df: pd.DataFrame,
    df_name: str,
    duplicate_rows_subset: list
):
    if duplicate_rows_subset is not None:
        # Extract duplicated_rows
        duplicate_rows_subset = [d for d in duplicate_rows_subset if d in df.columns]
        duplicate_rows = df.loc[df[duplicate_rows_subset].duplicated()]

        if duplicate_rows.shape[0] > 0:
            LOGGER.warning('%s contains duplicate rows:\n%s\n', df_name, duplicate_rows.to_string())
            return True
    return False


def has_duplicated_columns(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str
):
    duplicated_cols = df.columns[df.columns.duplicated()]
    if len(duplicated_cols) > 0:
        LOGGER.warning('%s %s had duplicated columns:\n%s\n', coin_name, df_name, pformat(duplicated_cols))
        return True
    return False


# Max & Min values
def has_values_above_max_allowed(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    max_values_allowed: dict = None,
    debug: bool = False
):
    if max_values_allowed is not None:
        # Collect max values
        max_values = df.filter(items=list(max_values_allowed.keys())).max()
        
        if debug:
            LOGGER.debug('max_values:\n%s\n', max_values)

        has_values_above_max_allowed_ = False

        for idx in max_values.index:
            if max_values[idx] > max_values_allowed[idx]:
                # Warn
                if coin_name is None:
                    LOGGER.warning(
                        '%s "%s" column max value (%s) is above max allowed (%s).',
                        df_name, idx, max_values[idx], max_values_allowed[idx]
                    )
                else:
                    LOGGER.warning(
                        '%s %s "%s" column max value (%s) is above max allowed (%s).',
                        coin_name, df_name, idx, max_values[idx], max_values_allowed[idx]
                    )
                has_values_above_max_allowed_ = True
        return has_values_above_max_allowed_
    return False


def has_values_below_min_allowed(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    min_values_allowed: dict = None,
    debug: bool = False
):
    if min_values_allowed is not None:
        # Collect min values
        min_values = df.filter(items=list(min_values_allowed.keys())).min()
        
        if debug:
            LOGGER.debug('min_values:\n%s\n', min_values)

        has_values_below_min_allowed_ = False

        for idx in min_values.index:
            if min_values[idx] < min_values_allowed[idx]:
                if coin_name is None:
                    LOGGER.warning(
                        '%s "%s" column min value (%s) is below min allowed (%s).',
                        df_name, idx, min_values[idx], min_values_allowed[idx]
                    )
                else:
                    LOGGER.warning(
                        '%s %s "%s" column min value (%s) is below min allowed (%s).',
                        coin_name, df_name, idx, min_values[idx], min_values_allowed[idx]
                    )
                has_values_below_min_allowed_ = True
        return has_values_below_min_allowed_
    return False


# Unique values
def has_unique_values_not_allowed(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    unique_values_allowed: Dict[str, list] = None,
    debug: bool = False
):
    if unique_values_allowed is not None:
        # Collect unique values
        unique_values = {
            col: df[col].unique().tolist()
            for col in unique_values_allowed.keys()
        }

        if debug:
            LOGGER.debug('unique_values:\n%s\n', pformat(unique_values))
            LOGGER.debug('unique_values_allowed:\n%s\n', pformat(unique_values_allowed))

        has_unique_values_not_allowed_ = False

        for key in unique_values.keys():
            # Find unexpected values
            unexpected_values = [
                c for c in unique_values[key] 
                if c not in unique_values_allowed[key]
            ]

            if len(unexpected_values) > 0:
                if coin_name is None:
                    LOGGER.warning(
                        '%s "%s" column unexpected values.\n'
                        '   - unexpected_values: %s\n'
                        '   - unique_values_allowed: %s\n',
                        df_name, key, unexpected_values, unique_values_allowed[key]
                    )
                else:
                    LOGGER.warning(
                        '%s %s "%s" column unexpected values.\n'
                        '   - unexpected_values: %s\n'
                        '   - unique_values_allowed: %s\n',
                        coin_name, df_name, key, unexpected_values, unique_values_allowed[key]
                    )
                has_unique_values_not_allowed_ = True
        return has_unique_values_not_allowed_
    return False


# Column Values above 0
def has_unexpected_negative_values(
    df: pd.DataFrame,
    columns: Iterable = None,
    coin_name: str = None,
    df_name: str = None
):
    if columns is not None:
        for col in columns:
            below_cero_vals = df.loc[df[col] < 0, col]
            if below_cero_vals.sum() > 0:
                LOGGER.warning(
                    "%s %s contains %s values in %s.", 
                    coin_name, df_name, len(below_cero_vals), col
                )
                return True
    return False


# Negative Prices
def is_standardized(
    s: pd.Series, 
    multiplier: int = 100
):
    """
    A standardized column will have roughly mean == 0 and std == 1. Therefore, if the actual mean, 
    multiplied by "multiplier is greater than the std, then we'll assume it's not standardized.
    """
    mean, std = s.mean(), s.std()
    if mean * multiplier > std:
        return False
    return True


def has_negative_prices(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    other_coins: list = None
):
    # Find price Cols
    price_cols = [
        'target_price',
        'coin_price',
        'coin_open',
        'coin_high',
        'coin_low',
        'ta_volume'
    ]

    # Add Other Cols
    if other_coins is not None:
        price_cols.extend([f'other_coins_{coin}_price' for coin in other_coins])

    # Add Stock Cols
    price_cols.extend([f'stock_{stock}_price' for stock in Params.fixed_params.get("full_stock_list")])
    
    price_cols = list(filter(lambda col: col in df.columns, set(price_cols)))

    for col in price_cols:
        if not is_standardized(df[col]):
            neg_prices = df.loc[df[col] < 0, col]
            if len(neg_prices) > 0:
                LOGGER.warning(
                    '%s %s had negative prices:\n'
                    'First example found:\n%s\n',
                    coin_name, df_name, neg_prices
                )
                return True
    return False


# Inconsistent Prices
def has_inconsistent_prices(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    other_coins: list = None
):
    # Find price Cols
    price_cols = [
        'target_price',
        'coin_price',
        'coin_open',
        'coin_high',
        'coin_low',
        'ta_volume'
    ]

    # Add Other Cols
    price_cols.extend([f'other_coins_{coin}_price' for coin in other_coins])
    
    price_cols = list(filter(lambda col: col in df.columns, set(price_cols)))

    for col in price_cols:
        # Calculate dummy derivartes
        dummy = df[[col]].copy()

        dummy.dropna(inplace=True)

        dummy['dummy_ret'] = dummy[col].pct_change()
        dummy['dummy_accel'] = dummy['dummy_ret'].diff()
        dummy['dummy_jerk'] = dummy['dummy_accel'].diff()

        # Chalculate mask
        mask = (
            (dummy['dummy_ret'] == 0) &
            (dummy['dummy_accel'] == 0) &
            (dummy['dummy_jerk'] == 0)
        )

        if mask.sum().sum() > 5:
            print(f"{coin_name} {df_name} has {mask.sum().sum()} inconsistent_prices.\n"
                  f"{dummy.loc[mask, [col, 'dummy_ret', 'dummy_accel', 'dummy_jerk']]}\n\n")
            return True
    return False


# Extreme Values
def has_extreme_values(
    df: pd.DataFrame,
    coin_name: str,
    df_name: str,
    outliers_dict: dict = None,
    debug: bool = False
):
    # Check outliers dict
    if outliers_dict is not None:
        for col, thresholds in outliers_dict.items():
            # Find outliers mask
            outliers_mask = np.logical_or(
                df[col] < thresholds[0],
                df[col] > thresholds[1]
            )

            if outliers_mask.sum() > 0:
                LOGGER.warning(
                    'Outliers found in %s %s %s.\n'
                    'Thresholds: %s\n'
                    '%s\n.',
                    col, coin_name, df_name, thresholds, 
                    df.loc[outliers_mask, col].to_string()
                )
                return True
            
    return False


# Excess Features
def has_excess_features(
    df: pd.DataFrame,
    max_features_perc: float = None,
    coin_name: str = None,
    df_name: str = None
):
    # Check Excess Features
    if max_features_perc is not None:
        max_cols = int(Params.data_params.get('periods') * max_features_perc)
        if df.shape[1] > max_cols:
            LOGGER.warning(
                "%s %s has excess features.\n"
                "Number of features: %s, max allowed: %s\n",
                coin_name, df_name, df.shape[1], max_cols
            )
            return True
    return False


# Expected Lenght
def has_short_length(
    df: pd.DataFrame,
    expected_periods: int,
    coin_name: str,
    df_name: str
):
    # Check data len
    if df.shape[0] < expected_periods * 0.95:
        LOGGER.warning(
            "%s %s is too 'short'.\n"
            "df.shape[0]: %s\n"
            "expected_periods: %s\n"
            "df.tail():\n%s\n",
            coin_name, df_name, df.shape[0], expected_periods, df.tail(10) # .to_string()
        )        
        return True
    return False


"""
Data Inspection & Repair
"""
def needs_repair(d: dict):
    for value in d.values():
        if value:
            return True
    return False


def find_data_diagnosis_dict(
    df_name: pd.DataFrame,
    intervals: str,
    coin_name: str = None,
    df: pd.DataFrame = None,
    debug: bool = False,
    **update_expectations: dict
) -> dict:
    # Load Expectations
    if df_name in [
        # Dim tables
        'fees', 'clients', 'accounts', 'models',

        # Fact tables
        'futures_balances', 'asset_valuations', 'futures_income_history',
        'futures_orders_history', 'inferences', 'delays', 
        
        # BI tables
        'trading_returns'
    ]:
        expectations: dict = load_from_s3(
            path=f"{Params.bucket}/utils/expectations/financials/{df_name}_expectations.json"
        )
    elif coin_name is not None:
        expectations: dict = load_from_s3(
            path=f"{Params.bucket}/utils/expectations/{intervals}/{coin_name}/{coin_name}_{df_name}_expectations.json"
        )
    else:
        expectations: dict = load_from_s3(
            path=f"{Params.bucket}/utils/expectations/{intervals}/collective_data_expectations.json"
        )

    if expectations is not None:
        # Update Expectations
        if update_expectations is not None:
            expectations.update(**update_expectations)

        # Extract DF
        if df is None and expectations.get("asset_path") is not None:
            df: pd.DataFrame = load_from_s3(path=expectations.get("asset_path"))
        
        # Define Diagnosis Dict
        diagnosis_dict = {}
        
        # Check Missing New Data
        if expectations.get('check_new_missing_data', False):
            diagnosis_dict['has_missing_new_data'] = has_missing_new_data(
                df=df.copy(),
                intervals=intervals,
                coin_name=coin_name,
                df_name=df_name
            )

        # Check Missing Columns
        if expectations.get('check_missing_cols', False):
            diagnosis_dict['has_missing_columns'] = has_missing_columns(
                df=df.copy(), 
                expected_cols=expectations['expected_cols'],
                coin_name=coin_name,
                df_name=df_name
            )

        # Check Unexpected Columns
        if expectations.get('check_unexpected_cols', False):
            diagnosis_dict['has_unexpected_columns'] = has_unexpected_columns(
                df=df.copy(), 
                expected_cols=expectations['expected_cols'],
                coin_name=coin_name,
                df_name=df_name
            )

        # Check Missing Rows
        if expectations.get('check_missing_rows', False):
            diagnosis_dict['has_missing_rows'] = has_missing_rows(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name,
                intervals=intervals
            )

        # Check Null Percentage
        if expectations.get('check_null_values', False):
            diagnosis_dict['has_exess_null_values'] = has_exess_null_values(
                df=df.copy(),
                null_perc_allowed=expectations['null_perc_allowed'],
                coin_name=coin_name,
                df_name=df_name,                
                debug=debug
            )

        # Check Duplicated IDX
        if expectations.get('check_duplicated_idx', False):
            diagnosis_dict['has_duplicated_idx'] = has_duplicated_idx(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name
            )

        # Check duplicated rows
        if expectations.get('check_duplicates_rows', False):
            diagnosis_dict['has_duplicates_rows'] = has_duplicates_rows(
                df=df.copy(),
                df_name=df_name,
                duplicate_rows_subset=expectations['duplicate_rows_subset']
            )

        # Check Duplicated Cols
        if expectations.get('check_duplicated_cols', False):
            diagnosis_dict['has_duplicated_columns'] = has_duplicated_columns(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name
            )

        # Check Max values allowed
        if expectations.get('check_max_values_allowed', False):
            diagnosis_dict['has_values_above_max_allowed'] = has_values_above_max_allowed(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name,
                max_values_allowed=expectations['max_values_allowed']
            )

        # Check Min values allowed
        if expectations.get('check_min_values_allowed', False):
            diagnosis_dict['has_values_below_min_allowed'] = has_values_below_min_allowed(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name,
                min_values_allowed=expectations['min_values_allowed']
            )

        # Check unique values allowed
        if expectations.get('check_unique_values_allowed', False):
            diagnosis_dict['has_unique_values_not_allowed'] = has_unique_values_not_allowed(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name,
                unique_values_allowed=expectations['unique_values_allowed']
            )

        # Check Inconsistent Prices
        if expectations.get('check_inconsistent_prices', False):
            diagnosis_dict['has_inconsistent_prices'] = has_inconsistent_prices(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name,
                other_coins=expectations['other_coins']
            )

        # Check Extreme Values
        if expectations.get('check_extreme_values', False):
            diagnosis_dict['has_extreme_values'] = has_extreme_values(
                df=df.copy(), 
                coin_name=coin_name,
                df_name=df_name,
                outliers_dict=expectations['outliers_dict'],
                debug=debug
            )
        
        # Check Exess Features
        if expectations.get('check_excess_features', False):
            diagnosis_dict['has_excess_features'] = has_excess_features(
                df=df.copy(), 
                max_features_perc=expectations['max_features_perc'],
                coin_name=coin_name,
                df_name=df_name
            )

        # Check Expected Periods
        if expectations.get('check_short_length', False):
            diagnosis_dict['has_short_length'] = has_short_length(
                df=df.copy(), 
                expected_periods=expectations['expected_periods'],
                coin_name=coin_name,
                df_name=df_name
            )
    else:
        diagnosis_dict = {'unable_to_find_expectations': True}
    
    return diagnosis_dict

