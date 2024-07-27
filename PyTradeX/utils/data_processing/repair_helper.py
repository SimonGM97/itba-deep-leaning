from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.utils.data_processing.data_expectations import (
    has_missing_new_data,
    is_standardized,

)
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.general.logging_helper import get_logger

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone
from tqdm import tqdm
from typing import Dict, Iterable
from pprint import pprint, pformat
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


# Missing New Data
def repair_missing_new_data(
    coin_name: str,
    intervals: str
):
    LOGGER.info("%s will be added to usdt_coins.json", coin_name)

    # Load usdt_coins
    s3_usdt_coins_path = f"{Params.bucket}/utils/usdt_coins/usdt_coins.json"
    usdt_coins: Dict[str, list] = load_from_s3(path=s3_usdt_coins_path)

    assert isinstance(usdt_coins[intervals], list)

    # Add coin_name to file
    usdt_coins[intervals].append(coin_name)
    usdt_coins[intervals] = list(set(usdt_coins[intervals]))
    
    # Save file

    # Update BinanceClient
    setattr(BinanceClient, 'usdt_coins', usdt_coins)


def reset_usdt_coins():
    print(f'Re-setting and updating usdt_coins.\n\n')

    # Instanciate BinanceClient and reset usdt_coins
    client = BinanceClient()

    # S3
    s3_usdt_coins_path = f"{Params.bucket}/utils/usdt_coins/usdt_coins.json"

    empty_usdt_coins = {
        '30min': [],
        '60min': [],
        '1d': []
    }

    # S3
    write_to_s3(
        asset=empty_usdt_coins,
        path=s3_usdt_coins_path
    )

    # Update BinanceClient
    setattr(BinanceClient, 'usdt_coins', empty_usdt_coins)

    for coin in tqdm(Params.fixed_params.get("full_coin_list")):
        for intervals in ['30min', '60min', '1d']:
            try:
                df = client.get_data(
                    coin_name=coin, 
                    intervals=intervals, 
                    periods=300,
                    ignore_last_period=True
                )
                time_since_last_period = datetime.now(timezone.utc).replace(tzinfo=None) - df.index[-1]
                # print(f"Time since last period ({coin} {intervals}): {time_since_last_period}.\n")
                # print(f"{df.tail()}\n\n")

                repair = has_missing_new_data(
                    df, 
                    intervals=intervals, 
                    coin_name=coin,
                    df_name='raw_data'
                )

                if repair:
                    repair_missing_new_data(coin, intervals)
            except Exception as e:
                LOGGER.warning(
                    "Unable to extract %s %s.\n"
                    "Thus, it will be added to usdt_coins.\n"
                    "Exception: %s\n",
                    coin, intervals, e
                )
                repair_missing_new_data(coin, intervals)


# Negative Prices
def is_price_col(c):
    if 'price' in c:
        if 'diff' not in c and 'return' not in c and 'acceleration' not in c and 'jerk' not in c:
            return True
    return False


# Replace Negative Values
def replace_negative_values(
    df: pd.DataFrame,
    non_neg_cols: list,
    coin_name: str = None,
    df_name: str = None,
    debug: bool = False
):
    if debug:
        print(f"Repairing negative values in {coin_name} {df_name}.\n")

    for col in non_neg_cols:
        df.loc[df[col] < 0, col] = np.nan

    return df


# Replace Negative Values
def replace_negative_prices(
    df: pd.DataFrame,
    coin_name: str = None,
    df_name: str = None,
    debug: bool = False
):
    if debug:
        print(f"Repairing negative prices in {coin_name} {df_name}.\n")

    price_cols = [c for c in df.columns if is_price_col(c)]
    for col in price_cols:
        if not is_standardized(df[col]):
            if df.loc[df[col] < 0, col].shape[0] > 0:
                df.loc[df[col] < 0, col] = np.nan
                # df[col] = df[col].interpolate().fillna(0).replace([np.inf, -np.inf], 0)
                # df.loc[df[col] < 0, col] = 0
    return df


# Repair Coin Cols
def load_other_coin_data(
    coin_name: str,
    intervals: str,
    periods: int,
    df_index: pd.Index = None,
    client: BinanceClient = None,
    price_only: bool = False,
    debug: bool = False
):
    # File System
    # base_path = os.path.join(
    #     Params.base_cwd, Params.bucket, "data_processing", "data_extractor", intervals
    # )

    # S3
    # s3_base_path = f"{Params.bucket}/data_processing/data_extractor/{intervals}/{coin_name}"

    if price_only:
        rename_cols = {'coin_price': f'other_coins_{coin_name}_price'}
    else:
        rename_cols = {
            'coin_open': f'other_coins_{coin_name}_open',
            'coin_high': f'other_coins_{coin_name}_high', 
            'coin_low': f'other_coins_{coin_name}_low',
            'coin_price': f'other_coins_{coin_name}_price',
            'ta_volume': f'other_coins_{coin_name}_volume'
        }

    # File System
    # raw_data_path = os.path.join(base_path, f"{coin_name}_raw_data.parquet")
    # loaded_fill_df = pd.read_parquet(raw_data_path)

    # S3
    s3_raw_data_path = f"{Params.bucket}/data_processing/data_extractor/{intervals}/{coin_name}/{coin_name}_raw_data.parquet"
    loaded_fill_df: pd.DataFrame = load_from_s3(path=s3_raw_data_path)
    
    loaded_fill_df = (
        loaded_fill_df
        # .loc[loaded_fill_df.index.isin(df_index)]
        .filter(items=list(rename_cols.keys()))
        .rename(columns=rename_cols)
    )

    fill_df = client.get_data(
        coin_name=coin_name,
        intervals=intervals,
        periods=periods,
        forced_stable_coin='USDT',
        update_last_prices=False,
        ignore_last_period=True
    )
    fill_df.rename(
        columns={
            'open': 'coin_open',
            'high': 'coin_high',
            'low': 'coin_low',
            'price': 'coin_price',
            'volume': 'ta_volume'
        },
        inplace=True
    )
    
    fill_df = (
        fill_df
        # .loc[fill_df.index.isin(df_index)]
        .filter(items=list(rename_cols.keys()))
        .rename(columns=rename_cols)
    )

    if loaded_fill_df is not None:
        fill_df = fill_df.combine_first(loaded_fill_df)

    # if debug:
    print(f'{coin_name} fill_df {fill_df.shape}: \n{fill_df.tail()}\n\n')
    
    if df_index is not None:
        return fill_df.loc[fill_df.index.isin(df_index)]
    else:
        return fill_df.iloc[-periods:]


def repair_coin_cols(
    df: pd.DataFrame, 
    other_coins: list, 
    intervals: str,
    price_only: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    # Real Columns
    df_coins_cols = [c for c in df.columns if c.startswith('other_coins')]

    # Expected Columns
    expected_cols = []
    def extend_expected_coin_cols(coin_):
        if price_only:
            expected_cols.extend([
                f'other_coins_{coin_}_price'
            ])
        else:
            expected_cols.extend([
                f'other_coins_{coin_}_open',
                f'other_coins_{coin_}_high',
                f'other_coins_{coin_}_low',
                f'other_coins_{coin_}_price',
                f'other_coins_{coin_}_volume'
            ])

    for coin in other_coins:
        extend_expected_coin_cols(coin)

    col_diff = list(set(df_coins_cols).symmetric_difference(set(expected_cols)))
    if len(col_diff) > 0:
        fill_cols = [c for c in expected_cols if c not in df_coins_cols]
        drop_cols = [c for c in df_coins_cols if c not in expected_cols]

        LOGGER.warning(
            'Other coins expected columns dont match actual seen cols.\n'
            'df_coins_cols:\n%s\n',
            'expected_cols:\n%s\n',
            'fill_cols:\n%s\n',
            'drop_cols:\n%s\n',
            pformat(df_coins_cols),
            pformat(expected_cols),
            pformat(fill_cols),
            pformat(drop_cols)
        )
        
        if len(fill_cols) > 0:
            client =  BinanceClient()
            concat_dfs = []

            for coin_name in set([c.split('_')[2] for c in fill_cols]):
                LOGGER.info('Loading %s data.', coin_name)
                concat_dfs.append(
                    load_other_coin_data(
                        coin_name=coin_name,
                        intervals=intervals,
                        periods=df.shape[0],
                        df_index=df.index,
                        client=client,
                        price_only=price_only,
                        debug=debug
                    )
                )
            
            if debug:
                for df_ in concat_dfs:
                    print(f'concat df: \n{df_}\n\n')
            
            full_fill = pd.concat(concat_dfs, axis=1)
            df = full_fill.combine_first(df)

        if len(drop_cols) > 0:
            df.drop(columns=drop_cols, inplace=True)

        if debug:
            print(f'Corrected df:\n {df[expected_cols]}\n\n')

        print('Corrected coin_cols:')
        pprint([c for c in df.columns if c.startswith('other_coins')])
        print('\n\n')
    
    return df


# Repair Stock & On-Chain Cols
def repair_stock_and_on_chain_cols(
    df: pd.DataFrame,
    intervals: str
) -> pd.DataFrame:
    find_cols = ['stock_' + stock + '_price' for stock in Params.fixed_params.get("full_stock_list")] + ['on_chain_transaction_rate_per_second']

    missing_cols = set(find_cols) - set(df.columns.tolist())
    if len(missing_cols) > 0:
        LOGGER.warning(
            "collective_data is missing %s.\n"
            "Thus, columns will be filled.\n",
            missing_cols
        )
        i = 0
        while len(missing_cols) > 0 and i < 5:
            coin = Params.fixed_params.get("full_coin_list")[1+i]
            # File System
            # fill_df: pd.DataFrame = pd.read_parquet(os.path.join(
            #     Params.base_cwd, Params.bucket, "data_processing", "data_extractor", intervals,
            #     f"{coin}_raw_data.parquet"
            # ))

            # S3
            fill_df: pd.DataFrame = load_from_s3(
                path=f"{Params.bucket}/data_processing/data_extractor/{intervals}/{coin}/{coin}_raw_data.parquet"
            )

            fill_df = fill_df.loc[fill_df.index.isin(df.index), missing_cols]
            
            if fill_df.shape[0] >= df.shape[0] * 0.95:
                df: pd.DataFrame = pd.concat([df, fill_df], axis=1)
                missing_cols = set(find_cols) - set(df.columns.tolist())

    return df


# Drop Duplicated IDX & Cols
def drop_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    # Drop Duplicated idx & Columns
    return df.loc[
        ~df.index.duplicated(keep='first'), 
        ~df.columns.duplicated(keep='first')
    ]


# Add Missing Rows
def add_missing_rows(
    df: pd.DataFrame,
    intervals: str
):
    freq = {
        '30min': '30min',
        '60min': '60min',
        '1d': '1D'
    }[intervals]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return df.reindex(full_idx).fillna(np.nan)


# Drop Unexpected Columns
def drop_unexpected_columns(
    df: pd.DataFrame,
    expected_columns: Iterable = None,
    coin_name: str = None,
    df_name: str = None
):
    if expected_columns is not None:
        drop_cols = [c for c in df.columns if c not in expected_columns]
        if len(drop_cols) > 0:
            print(f"{coin_name} {df_name} drop_cols:")
            pprint(drop_cols)
            print('\n\n')

            df.drop(columns=drop_cols, inplace=True)

    return df


# Find Dummy Collective Data
def find_dummy_collective_data(
    other_coins: list,
    intervals: str,
    collective_data_idx: pd.DatetimeIndex
):
    # Find & concat dummy_df
    dummy_list = []
    eth_raw_data = None

    # Fill with Raw Data
    for coin_name in other_coins:
        # File System
        # raw_data = pd.read_parquet(os.path.join(
        #     Params.base_cwd, Params.bucket, "data_processing", "data_extractor", intervals, 
        #     f"{coin_name}_raw_data.parquet"
        # ))

        # S3
        raw_data: pd.DataFrame = load_from_s3(
            path=f"{Params.bucket}/data_processing/data_extractor/{intervals}/{coin_name}/{coin_name}_raw_data.parquet"
        )

        if coin_name == 'ETH':
            eth_raw_data = raw_data.copy()
        
        dummy_list.append(
            raw_data
            .filter(items=['coin_price', 'coin_open', 'coin_high', 'coin_low', 'ta_volume'])
            .rename(columns={
                'coin_price': f'other_coins_{coin_name}_price', 
                'coin_open': f'other_coins_{coin_name}_open', 
                'coin_high': f'other_coins_{coin_name}_high', 
                'coin_low': f'other_coins_{coin_name}_low',
                'ta_volume': f'other_coins_{coin_name}_volume'
            })
        )
        
    # Fill Stock Data
    stock_cols = ['stock_' + stock + '_price' for stock in Params.fixed_params.get("full_stock_list")]
    dummy_list.append(eth_raw_data.filter(items=stock_cols))

    # Concat Data
    dummy_df = pd.concat(dummy_list, axis=1)

    # Filter Data
    dummy_df = dummy_df.loc[dummy_df.index.isin(collective_data_idx)]

    return dummy_df


def add_year_quarter_column(
    df: pd.DataFrame, 
    show_message: bool = True,
    df_name: str = None
):
    # Insert year_quarter column
    df['year_quarter'] = (
        df.index.to_series().dt.year.astype(str) + "_" 
        + df.index.to_series().dt.quarter.astype(str)
    )
    return df