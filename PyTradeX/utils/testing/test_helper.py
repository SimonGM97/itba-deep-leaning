from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.utils.others.s3_helper import (
    load_from_s3,
    write_to_s3
)
import pandas as pd


def dfs_are_equal(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    df1_name: str = None, 
    df2_name: str = None,
    tolerance: float = 0,
    debug: bool = False
):
    if df1.equals(df2):
        if debug:
            print(f"{df1_name} and {df2_name} are equal.\n")
        return True
    else:
        if debug:
            print(f"{df1_name} and {df2_name} are NOT equal.\n")
            try:
                compare_df = df1.compare(df2)
                print(f'compare_df: \n{compare_df}\n\n')
            except Exception as e:
                print(f'[WARNING] Unable to compare {df1_name} and {df2_name}.\n'
                      f'Exception: {e}\n\n')
            cols1 = df1.columns.tolist()
            cols2 = df2.columns.tolist()
            
            message = f"""
Running manual analysis:
{df1_name} shape: {df1.shape}
{df2_name} shape: {df2.shape}

{df1_name} size: {round(df1.memory_usage().sum() / 1024, 1)}
{df2_name} size: {round(df2.memory_usage().sum() / 1024, 1)}

{df1_name} columns: {cols1}
{df2_name} columns: {cols2}
column difference: {set(cols1).symmetric_difference(set(cols2))}

{df1_name} head: \n{df1.iloc[:10, :10]}\n
{df2_name} head: \n{df2.iloc[:10, :10]}\n
            """
            print(message)

        num_cols = list(df1.select_dtypes(include=['number']).columns)

        are_equal = True
        for col in num_cols:
            mape = (df1[col] - df2[col]).abs().sum() / df1[col].abs().sum()
                    
            if mape > tolerance:
                if debug:
                    abs_pct_diff_df = (df1[[col]] - df2[[col]]).abs() / df1[col].abs()
                    print(f'{col} mape ({mape}) is greated than allowed: {tolerance}!\n'
                        f'abs_pct_diff_df:\n{abs_pct_diff_df.loc[abs_pct_diff_df[col] > tolerance]}\n\n')
                are_equal = False

        return are_equal


def find_mocked_asset(
    target: str, 
    target_params: dict = None,
    re_create: bool = False
):
    def re_create_asset():
        print(f'Re-creating {target}')
        # Instanciate BinanceClient
        client = BinanceClient()

        # Define default params
        default_coin_name = 'ETH'
        default_futures = True
        default_update_last_prices = True
        default_ignore_last_period = True

        # Extract Params attrs
        default_intervals = Params.general_params.get('intervals')
        default_periods = Params.data_params.get('periods')

        # Find get_data
        if target == 'get_data':
            asset = client.get_data(
                coin_name=target_params.get('coin_name', default_coin_name),
                intervals=target_params.get('intervals', default_intervals),
                periods=target_params.get('periods', default_periods),
                futures=target_params.get('futures', default_futures),
                update_last_prices=target_params.get('update_last_prices', default_update_last_prices),
                ignore_last_period=target_params.get('ignore_last_period', default_ignore_last_period)
            )

        # Find get_long_short_data
        if target == 'get_long_short_data':
            asset = client.get_long_short_data(
                coin_name=target_params.get('coin_name', default_coin_name),
                from_=target_params.get('from_'),
                intervals=target_params.get('intervals', default_intervals),
                periods=target_params.get('periods', default_periods)
            )

        # Find raw_data
        if target == 'raw_data':
            # Define s3_base_backup_path
            coin_name = target_params.get('coin_name', default_coin_name)
            intervals = target_params.get('intervals', default_intervals)
            s3_base_backup_path = f"{Params.bucket}/backup/raw_data/{intervals}/{coin_name}"

            # Load raw_data_backup
            asset: pd.DataFrame = load_from_s3(
                path=f'{s3_base_backup_path}/{coin_name}_raw_data_backup.parquet',
                load_reduced_dataset=True
            )

            # Shorten data
            periods = target_params.get('periods', default_periods)
            asset = asset.iloc[-periods:]

        if target == 'cleaned_data':
            # Define s3_base_path
            coin_name = target_params.get('coin_name', default_coin_name)
            intervals = target_params.get('intervals', default_intervals)
            s3_base_path = f"{Params.bucket}/data_processing/data_cleaner/{intervals}/{coin_name}"

            # Load cleaned_data
            asset: pd.DataFrame = load_from_s3(
                path=f'{s3_base_path}/{coin_name}_cleaned_data.parquet',
                load_reduced_dataset=True
            )

            # Shorten data
            periods = target_params.get('periods', default_periods)
            asset = asset.iloc[-periods:]
        
        if target == 'cleaned_data_shift':
            # Define s3_base_path
            coin_name = target_params.get('coin_name', default_coin_name)
            intervals = target_params.get('intervals', default_intervals)
            s3_base_path = f"{Params.bucket}/data_processing/data_shifter/{intervals}/{coin_name}"

            # Load cleaned_data_shift
            asset = load_from_s3(
                path=f'{s3_base_path}/{coin_name}_cleaned_data_shift.parquet',
                load_reduced_dataset=True
            )

            # Shorten data
            periods = target_params.get('periods', default_periods)
            asset = asset.iloc[-periods:]

        if target == 'y':
            # Define s3_base_path
            coin_name = target_params.get('coin_name', default_coin_name)
            intervals = target_params.get('intervals', default_intervals)
            s3_base_path = f"{Params.bucket}/data_processing/data_transformer/{intervals}/{coin_name}"

            # Load cleaned_data_shift
            asset = load_from_s3(
                path=f'{s3_base_path}/{coin_name}_y.parquet',
                load_reduced_dataset=True
            )

            # Shorten data
            periods = target_params.get('periods', default_periods)
            asset = asset.iloc[-periods:]

        if target == 'X':
            # Define s3_base_path
            coin_name = target_params.get('coin_name', default_coin_name)
            intervals = target_params.get('intervals', default_intervals)
            s3_base_path = f"{Params.bucket}/data_processing/data_transformer/{intervals}/{coin_name}"

            # Load cleaned_data_shift
            asset = load_from_s3(
                path=f'{s3_base_path}/{coin_name}_X.parquet',
                load_reduced_dataset=True
            )

            # Shorten data
            periods = target_params.get('periods', default_periods)
            asset = asset.iloc[-periods:]
        
        # Save re_created dataset
        if isinstance(asset, pd.DataFrame):
            file_format = 'parquet'
        else:
            file_format = 'pickle'

        write_to_s3(
            asset=asset,
            path=f"{base_path}/{target}.{file_format}",
            overwrite=True
        )
        
        return asset
    
    # Define base_path
    base_path = f"{Params.bucket}/mock/data_processing/{Params.general_params.get('intervals')}"
    
    # Re-create datasets (if specified)
    if re_create:
        return re_create_asset()
    else:
        try:
            return load_from_s3(
                path=f"{base_path}/{target}.parquet",
                load_reduced_dataset=False
            )
        except Exception as e:
            print(f'[WARNING] Unable to load {base_path}/{target}.parquet.\n'
                  f'Re-creating mocked dataset.\n'
                  f'Exception: {e}\n\n')
            return re_create_asset()


def find_expected_asset(
    default_asset,
    asset_name: str,
    reset: bool = False
):
    # Define expected_asset_base_path
    expected_asset_base_path = f"{Params.bucket}/mock/data_processing/{Params.general_params.get('intervals')}"

    # Find file_format
    if isinstance(default_asset, pd.DataFrame):
        file_format = 'parquet'

    # Define expected_asset_path
    expected_asset_path = f"{expected_asset_base_path}/{asset_name}.{file_format}"

    if reset:
        # Save new default_asset
        write_to_s3(
            asset=default_asset,
            path=expected_asset_path,
            overwrite=True
        )

        # Replace expected_cleaned_data
        asset = default_asset.copy()
    else:
        try:
            asset = load_from_s3(
                path=expected_asset_path,
                load_reduced_dataset=False
            )
        except Exception as e:
            print(f"[WARNING] Unable to load {expected_asset_path}.\n"
                f"Thus, a new {asset_name} will be saved.\n"
                f"Exception: {e}\n\n")
            # Save new default_asset
            write_to_s3(
                asset=default_asset,
                path=expected_asset_path,
                overwrite=True
            )

            # Replace expected_cleaned_data
            asset = default_asset.copy()

    # Correct datetime index
    if isinstance(asset, pd.DataFrame):
        asset.index = pd.to_datetime(asset.index)
    
    return asset