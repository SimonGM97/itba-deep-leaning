from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import load_from_s3
import pandas as pd
from typing import Dict, List
import logging
from tqdm import tqdm


def load_modeling_datasets(
    intervals: str = None,
    full_coin_list: list = None,
    methods: List[str] = None,
    data_params: dict = None,
    reduced_datasets: bool = False,
    from_local: bool = False,
    to_local: bool = False,
    logger: logging.Logger = None,
    debug: bool = False
) -> Dict[str, Dict[str, pd.DataFrame]]:
    # Define datasets dict
    datasets: Dict[str, Dict[str, pd.DataFrame]] = {
        coin_name: {} for coin_name in full_coin_list
    }

    # Define periods to keep
    periods = data_params.get("periods")

    # Load DataTransformer datasets
    logger.info('Loading modeling datasets:')
    for coin_name in tqdm(full_coin_list):
        if from_local:
            # Load y_trans dataset
            y_trans: pd.DataFrame = pd.read_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_y_trans.parquet"
            )

            # Load X_trans dataset
            X_trans: pd.DataFrame = pd.read_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_X_trans.parquet"
            )

            # Load X_trans_pca dataset
            X_trans_pca: pd.DataFrame = pd.read_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_X_trans_pca.parquet"
            )

            # Load cleaned_data dataset
            cleaned_data: pd.DataFrame = pd.read_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_cleaned_data.parquet"
            )
        else:
            # Define base_data_paths
            base_dt_path = f"{Params.bucket}/data_processing/data_transformer/{intervals}/{coin_name}"
            base_dc_path = f"{Params.bucket}/data_processing/data_cleaner/{intervals}/{coin_name}"

            # Load y_trans dataset
            y_trans: pd.DataFrame = load_from_s3(
                path=f"{base_dt_path}/{coin_name}_y_trans.parquet",
                load_reduced_dataset=reduced_datasets
            )

            # Load X_trans dataset
            X_trans: pd.DataFrame = load_from_s3(
                path=f"{base_dt_path}/{coin_name}_X_trans.parquet",
                load_reduced_dataset=reduced_datasets
            )

            # Load X_trans_pca dataset
            X_trans_pca: pd.DataFrame = load_from_s3(
                path=f"{base_dt_path}/{coin_name}_X_trans_pca.parquet",
                load_reduced_dataset=reduced_datasets
            )

            # Load cleaned_data dataset
            cleaned_data: pd.DataFrame = load_from_s3(
                path=f"{base_dc_path}/{coin_name}_cleaned_data.parquet",
                load_reduced_dataset=reduced_datasets
            )

        # Save to local file
        if to_local:
            y_trans.to_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_y_trans.parquet"
            )
            X_trans.to_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_X_trans.parquet"
            )
            X_trans_pca.to_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_X_trans_pca.parquet"
            )
            cleaned_data.to_parquet(
                path=f"dummy_bucket/{intervals}/{coin_name}_cleaned_data.parquet"
            )

        # Find intersection index
        intersection = (
            y_trans.index
            .intersection(X_trans.index)
            .intersection(X_trans_pca.index)
            .intersection(cleaned_data.index)
        )

        # Populate dataset dict with y_trans
        datasets[coin_name]['y_trans'] = (
            y_trans
            .loc[intersection]
            .tail(periods)
        )

        # Populate dataset dict with X_trans
        datasets[coin_name]['X_trans'] = (
            X_trans
            .loc[intersection]
            .tail(periods)
        )

        # Populate dataset dict with X_trans
        datasets[coin_name]['X_trans_pca'] = (
            X_trans_pca
            .loc[intersection]
            .tail(periods)
        )

        # Populate dataset dict with cleaned_data
        cleaned_data_cols = [f'target_{method}' for method in methods] + ['coin_return', 'coin_open', 'coin_high', 'coin_low', 'coin_price']
        datasets[coin_name]['cleaned_data'] = (
            cleaned_data
            .loc[intersection]
            .filter(items=cleaned_data_cols)
            .tail(periods)
        )

        # Delete datasets from memory
        del y_trans
        del X_trans
        del X_trans_pca
        del cleaned_data

    print('\n')

    if debug:
        print('datasets:')
        for coin_name in full_coin_list:
            print(f'    datasets[{coin_name}]["y_trans"].shape: {datasets[coin_name]["y_trans"].shape}')
            print(f'    datasets[{coin_name}]["X_trans"].shape: {datasets[coin_name]["X_trans"].shape}')
            print(f'    datasets[{coin_name}]["X_trans_pca"].shape: {datasets[coin_name]["X_trans_pca"].shape}')
            print(f'    datasets[{coin_name}]["cleaned_data"].shape: {datasets[coin_name]["cleaned_data"].shape}\n')
        print('\n\n')

    return datasets