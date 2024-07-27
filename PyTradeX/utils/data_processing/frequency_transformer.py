from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
import pandas as pd
import numpy as np
from tqdm import tqdm


def transform_frequency(
    df: pd.DataFrame, 
    input_frequency: str,
    output_frequency: str
):
    if input_frequency == '30min':
        def find_agg_fun(col: str):
            if col.endswith('_high'):
                return 'max'
            if col.endswith('_low'):
                return 'min'
            if col.endswith('_open'):
                return 'first'
            if col.endswith('_price'):
                return 'last'
            if col.endswith('_volume'):
                return 'sum'
            if col.startswith('on_chain_'):
                return 'mean'
            if col.startswith('long_short_') and col != 'long_short_top_global_ratio':
                return 'first'
            if col.startswith('sentiment_lc_'):
                return 'first'
            return 'ignore'

        # Find initial cols
        ini_cols = df.columns.tolist().copy()

        # Define agg_funs
        agg_funs = {
            col: find_agg_fun(col) for col in df.columns 
            if find_agg_fun(col) != 'ignore'
        }
        
        # Apply agg_funs
        df = df.groupby(pd.Grouper(freq=output_frequency)).agg(agg_funs)

        # Add derivatives
        if 'coin_return' in ini_cols:
            df['coin_return'] = df['coin_price'].pct_change()
        if 'coin_acceleration' in ini_cols:
            df['coin_acceleration'] = df['coin_return'].diff()
        if 'coin_jerk' in ini_cols:
            df['coin_jerk'] = df['coin_acceleration'].diff()

        if 'target_return' in ini_cols:
            df['target_return'] = df['target_price'].pct_change()
        if 'coin_acceleration' in ini_cols:
            df['target_acceleration'] = df['target_return'].diff()
        if 'coin_jerk' in ini_cols:
            df['target_jerk'] = df['target_acceleration'].diff()

        # Add "long_short_top_global_ratio"
        if 'long_short_top_global_ratio' in ini_cols:
            df['long_short_top_global_ratio'] = (
                df['long_short_top_traders_long_short_ratio'] / df['long_short_global_long_short_ratio']
            )

        return df
    else:
        raise Exception(f'Invalid "input_frequency": {input_frequency}.\n\n')
    

# source .pytradex_venv/bin/activate
# .pytradex_venv/bin/python PyTradeX/utils/data_processing/frequency_transformer.py
if __name__ == '__main__':
    # Define input & output frequencies
    input_frequency = '30min'
    output_frequency = '60min'

    # Load collective_data
    collective_data_path = f"{Params.bucket}/utils/collective_data/{input_frequency}/collective_data.parquet"
    collective_data: pd.DataFrame = load_from_s3(
        path=collective_data_path,
        load_reduced_dataset=False
    )

    # Transform collective_data
    trans_collective_data: pd.DataFrame = transform_frequency(
        df=collective_data.copy(),
        input_frequency=input_frequency,
        output_frequency=output_frequency
    )

    # Save transformed collective_data
    write_to_s3(
        asset=trans_collective_data,
        path=f"{Params.bucket}/utils/collective_data/{output_frequency}/collective_data.parquet",
        overwrite=True
    )

    for coin_name in tqdm(Params.fixed_params.get('full_coin_list')):
        # Load coin raw_data
        load_raw_data_path = f"{Params.bucket}/data_processing/data_extractor/{input_frequency}/{coin_name}/{coin_name}_raw_data.parquet"
        raw_data: pd.DataFrame = load_from_s3(
            path=load_raw_data_path,
            load_reduced_dataset=False
        )

        # Transform raw_data
        trans_raw_data: pd.DataFrame = transform_frequency(
            df=raw_data.copy(),
            input_frequency=input_frequency,
            output_frequency=output_frequency
        )

        # Save transformed raw_data
        write_to_s3(
            asset=trans_raw_data,
            path=f"{Params.bucket}/data_processing/data_extractor/{output_frequency}/{coin_name}/{coin_name}_raw_data.parquet",
            overwrite=True
        )