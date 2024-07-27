from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.utils.data_processing.collective_data import (
    get_collective_data,
    update_coin_correlations, 
    update_collective_data_expectations
)
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing

from PyTradeX.data_processing.data_extractor import DataExtractor
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.data_processing.data_shifter import DataShifter
from PyTradeX.data_processing.data_refiner import DataRefiner
from PyTradeX.data_processing.data_transformer import DataTransformer
from PyTradeX.data_processing.feature_selector import FeatureSelector

import pandas as pd
import requests
from typing import Tuple, List
from tqdm import tqdm
from copy import deepcopy
from pprint import pformat


def load_data_classes(
    coin_name: str,
    client: BinanceClient,
    intervals: str,
    overwrite: bool,
    data_params: dict = None,
    data_extractor_update_params: dict = None,
    data_cleaner_update_params: dict = None,
    data_shifter_update_params: dict = None,
    data_refiner_update_params: dict = None,
    selected_features_update_params: dict = None,
    data_transformer_update_params: dict = None
) -> Tuple[
    DataExtractor, 
    DataCleaner, 
    DataShifter, 
    DataRefiner, 
    FeatureSelector,
    DataTransformer
]:
    # Validate inputs
    if data_extractor_update_params is None:
        data_extractor_update_params = {}
    if data_cleaner_update_params is None:
        data_cleaner_update_params = {}
    if data_shifter_update_params is None:
        data_shifter_update_params = {}
    if data_refiner_update_params is None:
        data_refiner_update_params = {}
    if selected_features_update_params is None:
        selected_features_update_params = {}
    if data_transformer_update_params is None:
        data_transformer_update_params = {}

    # Define base classes
    DE, DC, DS, DR, FS, DT  = None, None, None, None, None, None

    # DataExtractor
    if (
        data_extractor_update_params.get('update_data', False)
        or data_extractor_update_params.get('validate_data', False)

        or data_cleaner_update_params.get('update_data', False)
        or data_cleaner_update_params.get('rewrite_data', False)
        or data_cleaner_update_params.get('validate_data', False)
    ):
        DE = DataExtractor(
            coin_name=coin_name,
            client=client,
            intervals=intervals,
            overwrite=overwrite,
            **data_params.copy()
        )
    
    # DataCleaner
    if (
        data_cleaner_update_params.get('update_data', False)
        or data_cleaner_update_params.get('rewrite_data', False)
        or data_cleaner_update_params.get('validate_data', False)

        or data_shifter_update_params.get('update_data', False)
        or data_shifter_update_params.get('rewrite_data', False)
        or data_shifter_update_params.get('validate_data', False)
    ):
        DC = DataCleaner(
            coin_name=coin_name,
            intervals=intervals,
            overwrite=overwrite,
            **data_params.copy()
        )

    # DataShifter
    if (
        data_shifter_update_params.get('update_data', False)
        or data_shifter_update_params.get('rewrite_data', False)
        or data_shifter_update_params.get('validate_data', False)

        or data_refiner_update_params.get('update_data', False)
        or data_refiner_update_params.get('rewrite_data', False)
        or data_refiner_update_params.get('validate_data', False)
    ):
        DS = DataShifter(
            coin_name=coin_name,
            intervals=intervals,
            overwrite=overwrite,
            **data_params.copy()
        )
    
    # DataRefiner
    if (
        data_refiner_update_params.get('update_data', False)
        or data_refiner_update_params.get('rewrite_data', False)
        or data_refiner_update_params.get('validate_data', False)

        or data_transformer_update_params.get('update_data', False)
        or data_transformer_update_params.get('rewrite_data', False)
        or data_transformer_update_params.get('validate_data', False)
    ):
        DR = DataRefiner(
            coin_name=coin_name,
            intervals=intervals,
            overwrite=overwrite,
            **data_params.copy()
        )

    # FeatureSelector
    if (
        selected_features_update_params.get('update_primary_filter', False)
        or selected_features_update_params.get('update_selected_features', False)
        or selected_features_update_params.get('validate_selected_features', False)

        or data_transformer_update_params.get('update_data', False)
        or data_transformer_update_params.get('rewrite_data', False)
        or data_transformer_update_params.get('validate_data', False)
    ):
        FS = FeatureSelector(
            intervals=intervals,
            **data_params.copy()
        )

    # DataTransformer
    if (
        data_transformer_update_params.get('update_data', False)
        or data_transformer_update_params.get('rewrite_data', False)
        or data_transformer_update_params.get('validate_data', False)
    ):
        DT = DataTransformer(
            coin_name=coin_name,
            intervals=intervals,
            overwrite=overwrite,
            **data_params.copy()
        )

    return DE, DC, DS, DR, FS, DT 


def update_data_extractor(
    DE: DataExtractor,
    data_extractor_update_params: dict = None,
    debug: bool = False
) -> None:
    if (
        data_extractor_update_params.get('update_data')
        or data_extractor_update_params.get('validate_data')
    ):
        try:
            DE.update(
                debug=debug,
                **data_extractor_update_params
            )
        except Exception as e:
            LOGGER.critical(
                'Unable to update DataExtractor (%s).\n'
                'Exception: %s\n'
                'Reverting to backup.\n\n',
                DE.coin_name, e
            )

            DE.revert_to_backup()

            DE.update(
                debug=debug,
                **data_extractor_update_params
            )


def update_data_cleaner(
    DC: DataCleaner,
    DE: DataExtractor,
    data_cleaner_update_params: dict = None,
    debug: bool = False
) -> None:
    if (
        data_cleaner_update_params.get('update_data') 
        or data_cleaner_update_params.get('rewrite_data')
        or data_cleaner_update_params.get('validate_data')
    ):
        # try:
        DC.update(
            raw_data=DE.raw_data.copy(),
            unused_data=DE.unused_data.copy(),
            debug=debug,
            **data_cleaner_update_params
        )
        # except Exception as e:
        #     LOGGER.critical(
        #         'Unable to update DataCleaner (%s).\n'
        #         'Exception: %s\n'
        #         'Re-creating DataCleaner.\n\n',
        #         DC.coin_name, e
        #     )

        #     DC.update(
        #         raw_data=DE.raw_data.copy(),
        #         unused_data=DE.unused_data.copy(),
        #         debug=debug,
        #         **{
        #             'update_data': False,
        #             'rewrite_data': True,
        #             'update_expectations': True,
        #             'update_outliers_dict': True,
        #             'update_imputers': True,
        #             'validate_data': True,
        #             'save': True
        #         }
        #     )


def update_data_shifter(
    DS: DataShifter,
    DC: DataCleaner,
    data_shifter_update_params: dict = None,
    debug: bool = False
) -> None:
    if (
        data_shifter_update_params.get('update_data')
        or data_shifter_update_params.get('rewrite_data')
        or data_shifter_update_params.get('validate_data')
    ):
        # try:
        DS.update(
            cleaned_data=DC.cleaned_data.copy(),
            debug=debug,
            **data_shifter_update_params
        )
        # except Exception as e:
        #     LOGGER.critical(
        #         'Unable to update DataShifter (%s).\n'
        #         'Exception: %s\n'
        #         'Re-creating DataShifter.\n\n',
        #         DS.coin_name, e
        #     )

        #     DS.update(
        #         cleaned_data=DC.cleaned_data.copy(),
        #         debug=debug,
        #         **{
        #             'update_data': False,
        #             'rewrite_data': True,
        #             'update_expectations': True,
        #             'update_shift_dict': False,
        #             'record_df': True,
        #             'validate_data': True,
        #             'save': True
        #         }
        #     )


def update_data_refiner(
    DR: DataRefiner,
    DS: DataShifter,
    data_refiner_update_params: dict = None,
    debug: bool = False
) -> None:
    if (
        data_refiner_update_params.get('update_data')
        or data_refiner_update_params.get('rewrite_data')
        or data_refiner_update_params.get('validate_data')
    ):
        # try:
        DR.update(
            cleaned_data_shift=DS.cleaned_data_shift.copy(),
            debug=debug,
            **data_refiner_update_params
        )
        # except Exception as e:
        #     LOGGER.critical(
        #         'Unable to update DataRefiner (%s).\n'
        #         'Exception: %s\n'
        #         'Re-creating DataRefiner.\n\n',
        #         DR.coin_name, e
        #     )

        #     DR.update(
        #         cleaned_data_shift=DS.cleaned_data_shift.copy(),
        #         debug=debug,
        #         **{
        #             'update_data': False,
        #             'rewrite_data': True,
        #             'update_expectations': True,
        #             'update_outliers_dict': True,
        #             'validate_data': True,
        #             'save': True
        #         }
        #     )


def load_combined_refined_data(
    full_coin_list: list,
    intervals: str,
    re_create_comb_datasets: bool = False,
    reduce_comb_datasets: float = 1.0,
    local: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Define lists to append with datasets
    comb_y: List[pd.DataFrame] = []
    comb_X: List[pd.DataFrame] = []

    # Define base_path
    base_path = f"{Params.bucket}/data_processing/combined_datasets/30min"

    if re_create_comb_datasets:
        LOGGER.info('\nRe-creating comb_y & comb_X:')
        for coin_name in tqdm(full_coin_list):
            # Extract periods & lags
            periods = Params.data_params.get('periods')
            lag = Params.data_params.get('lag')

            # Load coin y
            y_path = f"{Params.bucket}/data_processing/data_refiner/{intervals}/{coin_name}/{coin_name}_y.parquet"
            y: pd.DataFrame = load_from_s3(
                path=y_path,
                load_reduced_dataset=False
            ).iloc[-periods:]
            y['coin_name'] = coin_name

            # Load coin X
            X_path = f"{Params.bucket}/data_processing/data_refiner/{intervals}/{coin_name}/{coin_name}_X.parquet"
            X: pd.DataFrame = load_from_s3(
                path=X_path,
                load_reduced_dataset=False
            ).iloc[-periods-lag:]
            X['coin_name'] = coin_name

            # Find Intersection
            intersection = y.index.intersection(X.index)
            y = y.loc[intersection]
            X = X.loc[intersection]

            # Reduce datasets
            if reduce_comb_datasets / 0.75 < 1.0:
                y = y.tail(int(0.75 * y.shape[0]))
                X = X.tail(int(0.75 * X.shape[0]))

                # Sample datasets
                y = y.sample(
                    frac=reduce_comb_datasets / 0.75,
                    replace=False,
                    random_state=23111997
                ).sort_index(ascending=True)

                X = X.loc[y.index]

            # Run asserts
            assert len(y.loc[y.index.duplicated()]) == 0
            assert len(X.loc[X.index.duplicated()]) == 0
            assert len(set(y.index).symmetric_difference(set(X.index))) == 0
            assert y.shape[0] == X.shape[0]

            # Append intersected datasets
            comb_y.append(y)
            comb_X.append(X)

        # Combine datasets
        comb_y: pd.DataFrame = pd.concat(comb_y)
        comb_X: pd.DataFrame = pd.concat(comb_X)
        
        # Save datasets
        if local:
            comb_y.to_parquet(f"dummy_bucket/{intervals}/comb_y.parquet")
            comb_X.to_parquet(f"dummy_bucket/{intervals}/comb_X.parquet")
        else:
            write_to_s3(
                asset=comb_y,
                path=f"{base_path}/comb_y.parquet",
                partition_cols=['coin_name'],
                overwrite=True
            )
            write_to_s3(
                asset=comb_X,
                path=f"{base_path}/comb_X.parquet",
                partition_cols=['coin_name'],
                overwrite=True
            )
    else:
        LOGGER.info('Loading comb_y & comb_X:\n')
        if local:
            comb_y: pd.DataFrame = pd.read_parquet(f"dummy_bucket/{intervals}/comb_y.parquet")
            comb_X: pd.DataFrame = pd.read_parquet(f"dummy_bucket/{intervals}/comb_X.parquet")
        else:
            comb_y: pd.DataFrame = load_from_s3(
                path=f"{base_path}/comb_y.parquet",
                load_reduced_dataset=False
            )
            comb_X: pd.DataFrame = load_from_s3(
                path=f"{base_path}/comb_X.parquet",
                load_reduced_dataset=False
            )

    LOGGER.info(
        'comb_y.shape: %s\n'
        'comb_X.shape: %s\n\n',
        comb_y.shape, comb_X.shape
    )
    
    return comb_y, comb_X


def update_feature_selector(
    FS: FeatureSelector,
    comb_y: pd.DataFrame,
    comb_X: pd.DataFrame,
    selected_features_update_params: dict = None,
    debug: bool = False
) -> None:
    if (
        selected_features_update_params.get('update_primary_filter')
        or selected_features_update_params.get('update_selected_features')
        or selected_features_update_params.get('validate_selected_features')
    ):
        # try:
        FS.update(
            y=comb_y,
            X=comb_X,
            debug=debug,
            **selected_features_update_params
        )
        # except Exception as e:
        #     LOGGER.critical(
        #         'Unable to update FeatureSelector (%s).\n'
        #         'Exception: %s\n'
        #         'Re-creating FeatureSelector.\n\n',
        #         FS.intervals, e
        #     )

        #     FS.update(
        #         y=comb_y,
        #         X=comb_X,
        #         debug=debug,
        #         **{
        #             'update_primary_filter': True,
        #             'update_selected_features': True,
        #             'validate_selected_features': True,
        #             'save': True
        #         }
        #     )


def update_data_transformer(
    DR: DataRefiner,
    FS: FeatureSelector,
    DT: DataTransformer,
    data_transformer_update_params: dict = None,
    debug: bool = False
) -> None:
    if (
        data_transformer_update_params.get('update_data')
        or data_transformer_update_params.get('rewrite_data')
        or data_transformer_update_params.get('validate_data')
    ):
        # try:
        DT.update(
            y=DR.y.copy(),
            X=DR.X.copy(),
            selected_features=deepcopy(FS.selected_features),
            debug=debug,
            **data_transformer_update_params
        )
        # except Exception as e:
        #     print(f'[WARNING] Unable to update DataShifter ({DT.coin_name}).\n'
        #         f'Exception: {e}\n'
        #         f'Re-creating DataShifter.\n\n')
        #     DT.update(
        #         y=DR.y.copy(),
        #         X=DR.X.copy(),
        #         debug=debug,
        #         **{
        #             'update_data': True,
        #             'rewrite_data': True,
        #             'update_expectations': False,
        #             'update_ignore_features': False,
        #             'refit_transformers': False,
        #             'validate_data': True,
        #             'save': True
        #         }
        #     )


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


@timing
def data_processing_job(
    client: BinanceClient,
    intervals: str,
    data_params: dict,
    full_coin_list: list,

    update_client: bool,
    update_correlations: bool,
    update_lc_ids: bool,
    overwrite: bool,

    collective_data_update_params: dict = None,
    data_extractor_update_params: dict = None,
    data_cleaner_update_params: dict = None,
    data_shifter_update_params: dict = None,
    data_refiner_update_params: dict = None,
    selected_features_update_params: dict = None,
    data_transformer_update_params: dict = None,    

    debug: bool = False
):
    # Validate inputs
    if collective_data_update_params is None:
        collective_data_update_params = {}
    if data_extractor_update_params is None:
        data_extractor_update_params = {}
    if data_cleaner_update_params is None:
        data_cleaner_update_params = {}
    if data_shifter_update_params is None:
        data_shifter_update_params = {}
    if data_refiner_update_params is None:
        data_refiner_update_params = {}
    if selected_features_update_params is None:
        selected_features_update_params = {}
    if data_transformer_update_params is None:
        data_transformer_update_params = {}

    # Update BinanceClient attrs
    if update_client:
        # Update precision
        client.update_quantity_precision()

        # Update available_coins
        client.update_available_pairs()

    # Update correlations
    if update_correlations:
        # Re-calculate correlations
        update_coin_correlations(
            client=client,
            intervals_list=[intervals],
            periods=data_params.get('periods'),
            debug=True
        )

        # Set up Params.other_coins_json
        Params.other_coins_json = load_from_s3(
            path=f"{Params.bucket}/utils/correlations/crypto_correlations.json"
        )
    
    # Update lc_ids
    if update_lc_ids:
        LOGGER.info('Updating lc_ids:\n')
        
        url = "https://lunarcrush.com/api4/public/coins/list/v2"
        headers = {'Authorization': 'Bearer imhsiympkj8e55rufld8oq1ecl9yupntkk4m3ep'}

        # Get response from lunarcrush API
        response = requests.request("GET", url, headers=headers).json()['data']
        data: pd.DataFrame = pd.DataFrame(response).filter(items=['id', 'symbol'])

        # Create lc_ids dictionaty
        lc_ids: dict = data.set_index('symbol').to_dict()['id']

        # Correct "GAL" key
        # lc_ids["GAL"] = lc_ids["GAL3"]

        # Filter results
        keep_keys = Params.fixed_params.get("full_coin_list") + ['BTC']
        lc_ids = {
            k: v for k, v in lc_ids.items() if k in keep_keys
        }

        # Write lc_ids
        write_to_s3(
            asset=lc_ids,
            path=f"{Params.bucket}/utils/lc_ids/lc_ids.json"
        )

        Params.lc_ids = lc_ids.copy()

        # Show lc_ids
        LOGGER.info('lc_ids:\n%s\n\n', pformat(lc_ids))

    # Update Collective Data
    if (
        collective_data_update_params.get('update_data', False) 
        or collective_data_update_params.get('validate_data', False) 

        or data_extractor_update_params.get('update_data', False)
        or data_extractor_update_params.get('rewrite_data', False)
    ):
        LOGGER.info('Updating collective_data.')

        # Update Collective Data Expectations
        if collective_data_update_params.get('update_expectations'):
            LOGGER.info('Updating collective_data expectations:')
            update_collective_data_expectations()

        # Extract other_coins
        other_coins = Params.other_coins_json[intervals][:data_params['other_coins_n']]
        
        # Update Collective Data
        get_collective_data(
            client=client,
            loaded_collective_data=None,
            accelerated=False,
            category_features=None,
            other_coins=other_coins,
            intervals=intervals,
            periods=data_params.get('periods'),
            yfinance_params=data_params.get('yfinance_params'),
            parallel=True,
            skip_check=False,
            validate=True,
            save=True,
            overwrite=overwrite,
            save_mock=False,
            ignore_last_period_check=False,
            debug=debug
        )

    # Update DataExtractor, DataCleaner, DataShifter & DataRefiner Classes
    if (
        data_extractor_update_params.get('update_data', False)
        or data_extractor_update_params.get('validate_data', False)

        or data_cleaner_update_params.get('update_data', False)
        or data_cleaner_update_params.get('rewrite_data', False)
        or data_cleaner_update_params.get('validate_data', False)

        or data_shifter_update_params.get('update_data', False)
        or data_shifter_update_params.get('rewrite_data', False)
        or data_shifter_update_params.get('validate_data', False)

        or data_refiner_update_params.get('update_data', False)
        or data_refiner_update_params.get('rewrite_data', False)
        or data_refiner_update_params.get('validate_data', False)
    ):
        LOGGER.info('Updating DataExtractor, DataCleaner, DataShifter & DataRefiner datasets.')

        for coin_name in tqdm(full_coin_list):
            DE, DC, DS, DR, _, _ = load_data_classes(
                coin_name=coin_name,
                client=client,
                intervals=intervals,
                overwrite=overwrite,
                data_params=data_params,
                data_extractor_update_params=data_extractor_update_params,
                data_cleaner_update_params=data_cleaner_update_params,
                data_shifter_update_params=data_shifter_update_params,
                data_refiner_update_params=data_refiner_update_params
            )
            
            # Update DataExtractor
            update_data_extractor(
                DE=DE,
                data_extractor_update_params=data_extractor_update_params,
                debug=debug
            )

            # Update DataCleaner
            update_data_cleaner(
                DC=DC,
                DE=DE,
                data_cleaner_update_params=data_cleaner_update_params,
                debug=debug
            )
            
            # Delete DE
            del DE

            # Update DataShifter
            update_data_shifter(
                DS=DS,
                DC=DC,
                data_shifter_update_params=data_shifter_update_params,
                debug=debug
            )

            # Delete DC
            del DC

            # Update DataRefiner
            update_data_refiner(
                DR=DR,
                DS=DS,
                data_refiner_update_params=data_refiner_update_params,
                debug=debug
            )

            # Delete DS & DR
            del DS
            del DR

    # Select Best Features
    if (
        selected_features_update_params.get('update_primary_filter', False)
        or selected_features_update_params.get('update_selected_features', False)
        or selected_features_update_params.get('check_selected_features', False)
        or selected_features_update_params.get('validate_selected_features', False)
    ):
        LOGGER.info('Updating FeatureSelector:')

        # Delete Data Classes from memory
        try:
            del DE
            del DC
            del DS
            del DR
        except:
            pass

        # Find combined datasets
        comb_y, comb_X = load_combined_refined_data(
            full_coin_list=Params.fixed_params.get('full_coin_list'),
            intervals=intervals,
            re_create_comb_datasets=selected_features_update_params.get('re_create_comb_datasets'),
            reduce_comb_datasets=selected_features_update_params.get('reduce_comb_datasets'),
            local=True # False
        )

        # Load FeatureSelector
        _, _, _, _, FS, _ = load_data_classes(
            coin_name=None,
            client=client,
            intervals=intervals,
            overwrite=overwrite,
            data_params=data_params,
            selected_features_update_params=selected_features_update_params
        )
    
        # Update FeatureSelector
        update_feature_selector(
            FS=FS,
            comb_y=comb_y,
            comb_X=comb_X,
            selected_features_update_params=selected_features_update_params,
            debug=debug
        )

        # import json
        # with open("selected_features.json", "w") as f:
        #     json.dump(FS.selected_features, f, indent=4)

        # Delete comb_y & comb_X from memory
        del comb_y
        del comb_X

    # Update DataTransformer
    if (
        data_transformer_update_params.get('update_data', False)
        or data_transformer_update_params.get('rewrite_data', False)
        or data_transformer_update_params.get('validate_data', False)
    ):
        LOGGER.info('Updating DataTransformer datasets:')

        # Delete Data Classes from memory
        try:
            del DE
            del DC
            del DS
        except:
            pass

        for coin_name in tqdm(full_coin_list):
            _, _, _, DR, FS, DT = load_data_classes(
                coin_name=coin_name,
                client=client,
                intervals=intervals,
                overwrite=overwrite,
                data_params=data_params,
                data_transformer_update_params=data_transformer_update_params
            )

            # Update DataTransformer
            update_data_transformer(
                DR=DR,
                FS=FS,
                DT=DT,
                data_transformer_update_params=data_transformer_update_params,
                debug=debug
            )