from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
from nancorrmp.nancorrmp import NaNCorrMp
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List
from pprint import pprint


def update_GFM_train_coins(
    ml_datasets: Dict[str, Dict[str, pd.DataFrame]],
    intervals: str,
    logger: logging.Logger = None,
    debug: bool = False
):
    # Extract coin_names & methods
    coin_names = ml_datasets.keys()
    methods = Params.ml_params.get('methods')

    # Define Correlation Matrix dict
    cm_dict = {}

    # Populate Correlation Matrix dict
    logger.info('Preparing Correlation Matrix dict:')
    for method in tqdm(methods):
        # Define method datasets
        method_datasets = pd.concat([
            ml_datasets[coin_name]['y_trans'][[f'target_{method}']].rename(columns={f'target_{method}': f'{coin_name}_target_{method}'})
            for coin_name in coin_names
        ], axis=1).iloc[:-1]

        # Calculate method Correlation Matrix
        cm: pd.DataFrame = (
            NaNCorrMp
            .calculate(method_datasets, n_jobs=Params.cpus)
            .abs()
        )

        # Populate dict
        cm_dict[method] = cm

        if debug:
            print(f'{method} Correlation Matrix:\n{cm_dict[method]}\n\n')

    # Define train_coins_dict
    train_coins_dict = {
        coin_name: {method: [] for method in methods}
        for coin_name in coin_names
    }

    # Populate train_coins_dict
    logger.info('\nPopulating train coins dict:')
    for coin_name in tqdm(coin_names):
        for method in methods:
            # Find correlation quantile & thresold
            cq = np.quantile(cm_dict[method][f'{coin_name}_target_{method}'], 0.25)
            threshold = min([0.6, cq])

            if debug:
                print(f'Correlation quantile {coin_name} {method}: {cq}\n'
                      f'Correlation threshold: {coin_name} {method}: {threshold}\n\n')
            
            # Filter CM & sort by target
            filtered_cm: pd.DataFrame = cm_dict[method][[f'{coin_name}_target_{method}']]
            filtered_cm.sort_values(by=f'{coin_name}_target_{method}', ascending=False, inplace=True)

            # Find top columns
            top_cols: List[str] = (
                filtered_cm
                .loc[filtered_cm[f'{coin_name}_target_{method}'] > threshold]
                .index
                .tolist()
            )
                
            # Populate train_coins_dict with top coins
            train_coins_dict[coin_name][method] = [col.split('_')[0] for col in top_cols]

            if debug:
                print(f'Top coins for {coin_name} {method}:')
                pprint(train_coins_dict[coin_name][method])
                print('\n\n')

    # Save train_coins_dict
    write_to_s3(
        asset=train_coins_dict,
        path=f"{Params.bucket}/utils/train_coins_dict/{intervals}/train_coins_dict.json"
    )


def load_GFM_train_coins(intervals: str):
    return load_from_s3(
        path=f"{Params.bucket}/utils/train_coins_dict/{intervals}/train_coins_dict.json"
    )