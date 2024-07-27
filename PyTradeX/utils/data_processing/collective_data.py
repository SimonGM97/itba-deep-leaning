from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair
)
from PyTradeX.utils.others.timing import timing
from PyTradeX.utils.data_processing.repair_helper import (
    add_missing_rows,
    drop_unexpected_columns,
    repair_coin_cols,
    repair_stock_and_on_chain_cols,
    drop_duplicates,
    find_dummy_collective_data,
    load_other_coin_data
)

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from bs4 import BeautifulSoup
from math import isnan
import yfinance
import requests
from tqdm import tqdm
import holidays
import time
from typing import List, Dict, Any, Tuple
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

# Extract MBP
MBP = Params.data_params.get("mbp")


def update_major_coins(
    max_coins: int = 20, 
    ignore_btc: bool = True, 
    minimum_var: float = None, 
    intervals: str = None
) -> None:
    # https://www.youtube.com/watch?v=thHCp3TL6QE
    major_coins_df = pd.DataFrame(columns=['name', 'market_cap'])

    url = 'https://coinmarketcap.com/'
    webpage = requests.get(url)
    soup = BeautifulSoup(webpage.text, 'html.parser')

    # while len(major_coins_df) < max_coins:
    pass


def update_coin_correlations(
    client: BinanceClient, 
    intervals_list: list = None, 
    periods: int = 10000,
    debug: bool = False
) -> None:
    def dummy_fun(coin_name: str, intervals: str) -> pd.DataFrame:
        if debug:
            print(f'Fetching {coin_name} data.\n')

        data: pd.DataFrame = client.get_data(
            coin_name=coin_name,
            intervals=intervals,
            periods=periods,
            forced_stable_coin='USDT',
            update_last_prices=False,
            ignore_last_period=True
        )
        if debug:
            print(f'Finished fetching {coin_name} data.\n')

        sleep_time = 0
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        while data.index[-1] < utc_now - pd.Timedelta(minutes=MBP * 2) and sleep_time < 5:
            LOGGER.warning(
                'Data from %s (%s) has not been updated quickly enough.\n'
                'data.index[-1]: %s (utc_now: %s).\n'
                'Sleep_time: %s\n',
                coin_name, intervals, data.index[-1], utc_now, sleep_time
            )
            time.sleep(1)
            sleep_time += 1
            data: pd.DataFrame = client.get_data(
                coin_name=coin_name,
                intervals=intervals,
                periods=periods,
                forced_stable_coin='USDT',
                update_last_prices=False,
                ignore_last_period=False
            )

            if debug:
                print(f'Finished re-fetching {coin_name} data.\n')

        if data.shape[0] < periods - 3:
            LOGGER.warning( 
                '%s new data has %s periods (expected: %s).\n'
                'Retrying to extract complete dataset.\n',
                coin_name, data.shape[0], periods
            )
            data: pd.DataFrame = load_other_coin_data(
                coin_name=coin_name,
                intervals=intervals,
                periods=periods,
                df_index=None,
                client=client,
                price_only=True,
                debug=debug
            ).rename(columns={f'other_coins_{coin_name}_price': 'price'})

        # Update global coins_data
        return data[['price']].rename(columns={'price': f'{coin_name}_price'})
        # global coins_data

        # coins_data = pd.concat([coins_data, data[['price']].rename(columns={'price': f'{coin_name}_price'})])
        # coins_data[coin_name + '_price'] = data['price']

    print('Updating Correlations:\n')

    if intervals_list is None:
        intervals_list = ['60min', '30min']

    # Load correlations
    s3_correlations_path = f"{Params.bucket}/utils/correlations/crypto_correlations.json"
    crypto_correlations = load_from_s3(
        path=s3_correlations_path
    )

    if debug:
        print('crypto_correlations:')
        pprint(crypto_correlations)
        print('\n\n')

    print('\nUpdating crypto_correlations.\n\n')

    for intervals in tqdm(intervals_list):
        # Define crypto_correlations "intervals" key
        crypto_correlations[intervals] = {}

        # Find looping params
        full_coin_list = Params.fixed_params.get("full_coin_list")
        max_workers = min([len(full_coin_list), Params.cpus, 4])

        # Build coins_data
        coins_data: pd.DataFrame = pd.DataFrame()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            concat_list = []
            for name in full_coin_list:
                concat_list.append(executor.submit(dummy_fun, name, intervals))
            for dataset in as_completed(concat_list):
                coins_data = pd.concat([coins_data, dataset._result], axis=1)

        # Clean coins_data
        # DC = DataCleaner(
        #     coin_name=None,
        #     intervals=intervals,
        #     overwrite=False,
        #     **Params.data_params
        # )

        # DC.z_threshold = Params.data_params['z_threshold']

        # coins_data = DC.cleaner_pipeline(
        #     df=coins_data,
        #     unused_data=None,
        #     remove_unexpected_neg_values=True,
        #     non_neg_cols=coins_data.columns.tolist(),
        #     remove_inconsistent_prices=True,
        #     handle_rows_and_columns=True,
        #     expected_cols=coins_data.columns.tolist(),
        #     new_data=None,
        #     remove_outliers=True,
        #     update_outliers_dict=True,
        #     z_threshold=DC.z_threshold,
        #     impute_nulls=True,
        #     update_imputers=True,
        #     validate_data=False
        # )

        # Sleep for 60 min not to overload the API
        print(f'sleeping 60 sec not to overload API.\n')
        for i in tqdm(range(600)):
            time.sleep(0.1)

        # Second run to guarantee all coins were searched
        for name in full_coin_list:
            if f'{name}_price' not in coins_data.columns:
                print(f"{name} was not found in coins_data.\n"
                      f"Will re-try a second time.\n\n")
                coins_data: pd.DataFrame = pd.concat([
                    coins_data,
                    dummy_fun(coin_name=name, intervals=intervals)
                ], axis=1)

        for column in coins_data.columns:
            coin = column.split('_')[0]
            coins_data[f'{coin}_return'] = coins_data[f'{coin}_price'].pct_change()
            coins_data[f'{coin}_acceleration'] = coins_data[f'{coin}_return'].diff()
            coins_data[f'{coin}_jerk'] = coins_data[f'{coin}_acceleration'].diff()

        coins_data.drop(columns=[c for c in coins_data.columns if 'price' in c], inplace=True)

        for column in coins_data.columns:
            coins_data[f'{column}_shifted'] = coins_data[column].shift(1)

        # print(coins_data.tail())
        
        cm = pd.DataFrame(coins_data.corr().abs())

        # if debug:
        #   print(f'cm: \n{cm}\n\n')
        
        for coin in full_coin_list:
            # try:
            dummy_cm = cm.sort_values(by=[f'{coin}_return'], ascending=False)
            cm_columns = [c for c in dummy_cm.columns if (coin in c and 'shifted' in c)]
            cm_index = [c for c in dummy_cm.columns if (coin not in c and 'shifted' not in c)]
            dummy_cm = dummy_cm.loc[cm_index][cm_columns]

            correlation_score = dummy_cm.mean().mean()
            crypto_correlations[intervals][coin] = correlation_score

            if debug:
                print(f'{coin} correlation_score: {correlation_score}\n')
                #      f'dummy_cm: \n{dummy_cm}\n\n')
            # except Exception as e:
            #     print(f'[WARNING] Exception with {coin} data.\n'
            #           f'Exception: {e}\n\n')

        if debug:
            print(f'crypto_correlations[{intervals}]:')
            pprint(crypto_correlations[intervals])
            print('\n\n')

        crypto_correlations[intervals] = [
            k for k, _ in sorted(crypto_correlations[intervals].items(), key=lambda item: item[1], reverse=True)
        ]

        if debug:
            print(f'new crypto_correlations[{intervals}]:')
            pprint(crypto_correlations[intervals])
            print(f'\nlen(crypto_correlations[{intervals}]): {len(crypto_correlations[intervals])}\n\n')

    # File System
    # correlations_path = os.path.join(
    #     Params.base_cwd, Params.bucket, "utils", "correlations", "crypto_correlations.json"
    # )
    # with open(correlations_path, "w") as f:
    #     json.dump(crypto_correlations, f, indent=4)

    # S3
    write_to_s3(
        asset=crypto_correlations,
        path=s3_correlations_path
    )

    """
    with open(os.path.join(tmp_dir, "parameters.json"), "w") as f:
                    json.dump(parameters, f, indent=4)
    """

    """
    print('\nUpdating stock_correlations.\n\n')
    
    stock_correlations = {}
    
    for intervals in tqdm(['1d', '60min', '30min']):
        stock_correlations[intervals] = {}
    
        yfinance_stocks_data = yfinance.download(
            tickers=['^GSPC', '^DJI', '^IXIC', '^NYA', '^XAX', '^VIX', '^NDX', '^RUT'],
            period=Params.data_params.get("yfinance_params")[intervals]['period'],
            interval=Params.data_params.get("yfinance_params")[intervals]['interval'],
            progress=False,
            # verify=False
        )[['Close']]
    
        dummy_cols = [coin[1].split('-')[0][1:] + '_price' for coin in yfinance_stocks_data.columns]
        yfinance_stocks_data.columns = dummy_cols
        yfinance_stocks_data.index = pd.to_datetime(yfinance_stocks_data.index)
    
        if intervals != '1d':
            yfinance_stocks_data.index = yfinance_stocks_data.index.tz_convert(None)
    
        if intervals == '60min':
            yfinance_stocks_data = pd.DataFrame(
                yfinance_stocks_data.groupby(pd.Grouper(freq=intervals)).first()).ffill()
        else:
            yfinance_stocks_data = pd.DataFrame(
                yfinance_stocks_data.groupby(pd.Grouper(freq=intervals)).last())  # .ffill()
    
        for column in yfinance_stocks_data.columns:
            stock = column.split('_')[0]
            yfinance_stocks_data[f'{stock}_price_return'] = yfinance_stocks_data[f'{stock}_price'].pct_change()
            yfinance_stocks_data[f'{stock}_price_acceleration'] = yfinance_stocks_data[f'{stock}_price_return'].diff()
            yfinance_stocks_data[f'{stock}_price_jerk'] = yfinance_stocks_data[f'{stock}_price_acceleration'].diff()
    
        yfinance_stocks_data.drop(columns=dummy_cols, inplace=True)
    
        for coin in major_coins_list:
            try:
                data = client.get_data(
                    name=coin,
                    intervals=intervals,
                    periods=periods
                )[['price']]
    
                data[f'{coin}_price_return'] = data['price'].pct_change()
                data[f'{coin}_price_acceleration'] = data[f'{coin}_price_return'].diff()
                data[f'{coin}_price_jerk'] = data[f'{coin}_price_acceleration'].diff()
    
                cm = pd.DataFrame(
                    pd.concat([data, yfinance_stocks_data.shift(1)],
                              axis=1).corr().applymap(lambda x: np.abs(x))
                ).sort_values(by=[f'{coin}_price_acceleration'], ascending=False)
    
                correlation_list = [i.split('_')[0] for i in cm.index]
                correlation_indexes = np.unique(correlation_list, return_index=True)[1]
                stock_correlations[intervals][coin] = [correlation_list[index] for index in sorted(correlation_indexes)
                                                       if correlation_list[index] != coin]
    
                # print(f'coin: {coin}\n'
                #       f'stock_correlations: {[correlation_list[index] for index in sorted(correlation_indexes) if correlation_list[index] != coin]}\n\n'
                #       f'cm: \n{cm[f"{coin}_price_acceleration"].head(15)}\n\n')
            except Exception as e:
                print(f'[WARNING] Exception with {coin} data.\n'
                      f'Exception: {e}\n\n')
    
    base_cwd = os.getcwd()
    with open(f"{base_cwd}\\data_warehouse\\utils\\correlations\\stock_correlations.json", "w") as f:
        json.dump(stock_correlations, f, indent=4)
    """


def load_collective_data(
    intervals: str = Params.general_params.get("intervals"),
    load_reduced_dataset: bool = False,
    save_mock: bool = False
) -> pd.DataFrame:
    # Define load path
    load_path = f"{Params.bucket}/utils/collective_data/{intervals}/collective_data.parquet"

    i = 1
    loaded_data = None
    while loaded_data is None and i < 3:
        try:
            loaded_data: pd.DataFrame = load_from_s3(
                path=load_path,
                load_reduced_dataset=load_reduced_dataset
            )
        except Exception as e:
            LOGGER.error(
                'Unable to load Other Coins Data (%s).\n'
                'Retrying for the %sth time.\n'
                'Exception: %s\n',
                intervals, i+1, e
            )
        time.sleep(1)
        i += 1

    if save_mock:
        save_mock_asset(
            asset=loaded_data, 
            asset_name='loaded_collective_data', 
            intervals=intervals
        )

    return loaded_data


def save_collective_data(
    df: pd.DataFrame,
    intervals: str = Params.general_params.get("intervals"),
    overwrite: bool = True,
    **update_expectations
) -> None:
    # Validate periods
    expected_periods = Params.data_params.get('periods')
    if df.shape[0] < expected_periods and overwrite:
        LOGGER.warning(
            'Skipping collective_data overwrite save, as periods were too small.\n'
            'Expected periods: %s.\n'
            'Seen periods: %s.\n',
            expected_periods, df.shape[0]
        )
    else:
        # Find Diagnostics Dict
        diagnostics_dict = diagnose_collective_data(
            collective_data=df.copy(),
            intervals=intervals,
            debug=False,
            **update_expectations
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                '"collective_data" needs repair, thus it will not be saved.\n'
                'diagnostics_dict:\n%s\n',
                pformat(diagnostics_dict)
            )
        else:
            LOGGER.info(
                'Saving collective_data %s (last obs: %s, utc_now: %s).',
                df.shape, df.index[-1], datetime.now(timezone.utc).replace(tzinfo=None)
            )
            write_to_s3(
                asset=df,
                path=f"{Params.bucket}/utils/collective_data/{intervals}/collective_data.parquet",
                overwrite=overwrite
            )


def find_collective_data_columns(
    other_coins: list,
    save_mock: bool = False,
    debug: bool = False
) -> Tuple[List[str], List[str]]:
    full_coin_cols = []
    for name in other_coins:
        full_coin_cols.extend([
            f'other_coins_{name}_open', 
            f'other_coins_{name}_high', 
            f'other_coins_{name}_low',
            f'other_coins_{name}_price', 
            f'other_coins_{name}_volume'
        ])

    full_stock_cols = ['stock_' + stock + '_price' for stock in Params.fixed_params.get("full_stock_list")]

    if debug:
        print(f'full_coin_cols:\n')
        pprint(full_coin_cols)
        print('\n\n')
        print(f'full_stock_cols:\n')
        pprint(full_stock_cols)
        print('\n\n')

    if save_mock:
        save_mock_asset(
            asset=(full_coin_cols, full_stock_cols), 
            asset_name='data_columns', 
            intervals=None
        )

    return full_coin_cols, full_stock_cols


def find_other_coins_data(
    client: BinanceClient,
    intervals: str = Params.general_params.get('intervals'),
    periods: int = Params.data_params.get('periods'),
    parallel: bool = True,
    other_coins: List[str] = None,
    full_coin_cols: List[str] = None,
    accelerated: bool = False,
    category_features: Dict[str, List[str]] = None,
    save_mock: bool = False,
    ignore_last_period_check: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    def find_coin_data(coin_name: str) -> pd.DataFrame:
        if debug:
            print(f'Fetching {coin_name} data.\n')
        
        # try:
        data: pd.DataFrame = client.get_data(
            coin_name=coin_name,
            intervals=intervals,
            periods=periods + 1,
            forced_stable_coin='USDT', # None, 'USDT'
            update_last_prices=False,
            ignore_last_period=True
        )
        if debug:
            print(f'Finished fetching {coin_name} data.\n')
        
        if not ignore_last_period_check:
            delay = 0
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            while data.index[-1] < utc_now - pd.Timedelta(minutes=MBP * 2) and delay < 5:
                LOGGER.warning(
                    'Data from %s (%s) has not been updated quickly enough.\n'
                    'data.index[-1]: %s (utc_now: %s).\n',
                    coin_name, intervals, data.index[-1], utc_now
                )
                time.sleep(1)
                delay += 1
                data: pd.DataFrame = client.get_data(
                    coin_name=coin_name,
                    intervals=intervals,
                    periods=periods + 1,
                    forced_stable_coin='USDT', # None, 'USDT'
                    update_last_prices=False,
                    ignore_last_period=False
                )

                if debug:
                    print(f'Finished re-fetching {coin_name} data.\n')
        
        if save_mock:
            save_mock_asset(
                asset=data,
                asset_name='client_data',
                intervals=intervals,
                coin_name=coin_name
            )

        data = data.rename(columns={
            'open': f'other_coins_{coin_name}_open',
            'high': f'other_coins_{coin_name}_high',
            'low': f'other_coins_{coin_name}_low',
            'price': f'other_coins_{coin_name}_price',
            'volume': f'other_coins_{coin_name}_volume'
        })

        if data.shape[0] < periods:
            print(f'[WARNING] {coin_name} new data has {data.shape[0]} periods (expected: {periods}).\n'
                  f'Retrying to extract complete dataset.\n\n')
            data: pd.DataFrame = load_other_coin_data(
                coin_name=coin_name,
                intervals=intervals,
                periods=periods,
                df_index=None,
                client=client,
                price_only=False,
                debug=debug
            )
        
        # Update global coins_data
        # global coins_data
        # coins_data.drop(columns=list(data.columns), errors='ignore', inplace=True)
        # coins_data[list(data.columns)] = data
        return data
        # except Exception as e:
        #     print(f"[WARNING] Unable to extract {name_} data for collective_data.\n"
        #           f"Exception: {e}.\n\n")
    
    # Re-define other_coins
    if accelerated:
        # Extract other_coins_cols
        other_coins_cols = category_features['other_coins']

        # Define new other_coins
        new_other_coins = list(set([c.split('_')[2] for c in other_coins_cols]))

        # Re-define ordered other_coins
        other_coins = [c for c in other_coins if c in new_other_coins]

    # try:
    if parallel:
        # Find max_workers
        max_workers = min([len(other_coins), Params.cpus])

        # Build coins_data
        coins_data: pd.DataFrame = pd.DataFrame()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            concat_list = []
            for name in other_coins:
                concat_list.append(executor.submit(find_coin_data, name))
            for dataset in as_completed(concat_list):
                coins_data = pd.concat([coins_data, dataset._result], axis=1)
    else:
        concat_list = []
        for name in other_coins:
            concat_list.append(find_coin_data(coin_name=name))

        # Build coins_data
        coins_data: pd.DataFrame = pd.concat(concat_list, axis=1)

    for name in other_coins:
        if f'other_coins_{name}_price' not in coins_data.columns:
            print(f'Re-fetching {name} data to fill collective data.\n')
            coins_data: pd.DataFrame = pd.concat([
                coins_data,
                find_coin_data(coin_name=name)
            ], axis=1)
    
    # Check there are no null coins
    nulls = coins_data.isna().sum()
    null_cols: List[str] = nulls.loc[nulls > int(periods * 0.02)].index
    null_coins = list(set([col.split('_')[2] for col in null_cols]))
    for coin in null_coins:
        print_cols = [c for c in coins_data.columns if coin in c]
        null_values = coins_data[print_cols].isna().sum().sum()
        print(f'Coin {coin} had {null_values} null values.\n')
        find_coin_data(coin_name=coin)
        # print(f"New data from {coin}:\n{coins_data[print_cols]}\n\n")

    # Keep expected cols
    keep_cols = [c for c in coins_data.columns if c in full_coin_cols]
    coins_data = coins_data[keep_cols]

    # Complete coins_data
    if accelerated:
        # Find columns to add
        add_cols = [col for col in full_coin_cols if col not in coins_data.columns]

        if len(add_cols) > 0:
            # Create add_df
            add_df = pd.DataFrame(
                np.zeros(shape=(len(coins_data.index), len(add_cols))),
                columns=add_cols,
                index=coins_data.index
            )

            # Concatenate add_df
            coins_data = pd.concat([coins_data, add_df], axis=1)

    if debug:
        nulls = coins_data.isna().sum()
        null_cols = nulls.loc[nulls > 0].index
        null_coins = list(set([col.split('_')[2] for col in null_cols]))
        print(f'null_cols: {null_cols}\n'
              f'null_coins: {null_coins}\n'
              f'coins_data.columns: {coins_data.columns}\n'
              f'coins_data.shape: {coins_data.shape}\n'
              f'coins_data.tail(10): \n{coins_data.tail(10)}\n\n')
    # except Exception as e_:
    #     print(f'\n[WARNING] "coins_data" could not be found ({intervals}).\n'
    #           f'Exception: {e_}\n\n')
    #     coins_data = None

    return coins_data


def yfinance_download(
    tickers: List[str],
    yfinance_params: Dict[str, str],
    debug: bool = False
) -> pd.DataFrame:
    # Extract stocks_data
    stocks_data: pd.DataFrame = yfinance.download(
        tickers=tickers,
        period=yfinance_params['period'],
        interval=yfinance_params['interval'],
        ignore_tz=False,
        progress=False,
        # verify=False
    )[['Close']]
    
    i = 1
    while len(stocks_data.columns) < len(tickers) and i < 5:
        LOGGER.warning(
            'Unable to retrieve complete stock_data.\n'
            'Retrying for the %sth time.\n', i
        )
        time.sleep(1)
        stocks_data: pd.DataFrame = yfinance.download(
            tickers=tickers,
            period=yfinance_params['period'],
            interval=yfinance_params['interval'],
            ignore_tz=False,
            progress=False,
            # verify=False
        )[['Close']]
        i += 1

    if debug:
        print(f'Raw stocks_data:\n{stocks_data.tail()}\n\n')
    
    # Define stocks_data columns
    stocks_data.columns = ['stock_' + stock[1].split('-')[0][1:] + '_price' for stock in stocks_data.columns]

    # Set datetime index
    stocks_data.index = pd.to_datetime(stocks_data.index, utc=True)

    return stocks_data


def find_stock_data(
    intervals: str,
    full_stock_cols: List[str],
    yfinance_params: dict,
    accelerated: bool = False,
    category_features: Dict[str, List[str]] = None,
    expected_idx: pd.DatetimeIndex = None,
    save_mock: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    # Extract stocks
    stocks = [stock.split('_')[1] for stock in full_stock_cols]

    # Re-define stocks
    if accelerated:
        # Extract stock_cols
        stock_cols = category_features['stock']

        # Define new stocks
        new_stocks = list(set([c.split('_')[1] for c in stock_cols]))

        # Re-define ordered stock_cols
        stocks = [c for c in stocks if c in new_stocks]

    if debug:
        print(f'stocks: {stocks}\n')

    # try:
    # Find tickers
    tickers = [f"^{stock}" for stock in stocks]
    if debug:
        print(f'tickers: {tickers}\n')

    if len(tickers) > 0:
        # Extract yfinance stocks_data
        stocks_data: pd.DataFrame = yfinance_download(
            tickers=tickers,
            yfinance_params=yfinance_params,
            debug=debug
        )

        if save_mock:
            save_mock_asset(
                asset=stocks_data,
                asset_name='stocks_data',
                intervals=intervals
            )
        
        # if intervals != '1d':
        #     print(stocks_data.index)
        #     stocks_data.index = stocks_data.index.tz_convert(None)

        if intervals == '60min':
            stocks_data: pd.DataFrame = pd.DataFrame(
                stocks_data.groupby(pd.Grouper(freq=intervals)).first()
            )

        # Filter columns to keep
        keep_cols = [c for c in stocks_data.columns if c in full_stock_cols]
        stocks_data: pd.DataFrame = stocks_data[keep_cols]
        
        # De-localize index
        stocks_data.index = stocks_data.index.tz_convert('UTC').tz_localize(None)
    else:
        stocks_data = pd.DataFrame(
            columns=[],
            index=expected_idx
        )

    # Complete stocks_data
    if accelerated:
        # Find columns to add
        add_cols = [col for col in full_stock_cols if col not in stocks_data.columns]

        if len(add_cols) > 0:
            # Create add_df
            add_df = pd.DataFrame(
                np.zeros(shape=(len(stocks_data.index), len(add_cols))),
                columns=add_cols,
                index=stocks_data.index
            )

            # Concatenate add_df
            stocks_data = pd.concat([stocks_data, add_df], axis=1)

    if debug:
        print(f'datetime.now(timezone.utc): {datetime.now(timezone.utc)}\n'
              f'stocks_data.shape: {stocks_data.shape}\n'
              f'stocks_data.tail(): \n{stocks_data.tail()}\n\n')
    # except Exception as e:
    #     print(f'[WARNING] "stocks_data" could not be found ({intervals}).\n'
    #             f'Exception: {e}\n\n')
    #     stocks_data = None

    return stocks_data


def correct_stock_data(
    stock_data: pd.DataFrame
) -> pd.DataFrame:
    def correct_shift(group_data: pd.DataFrame):
        obs_idx = group_data.loc[group_data['stock_NDX_price'].notna()].index
        if len(obs_idx) > 0 and obs_idx[-1].hour < 19 and group_data.index[-1].hour > 19:
            print(f'Correcting:\n{group_data}\n\n')
            shift_hours = 19 - obs_idx[-1].hour
            shift_mins = 30 - obs_idx[-1].minute
            shift_periods = shift_hours * 2 + int(shift_mins/30)

            print(f'obs_idx[-1].hour: {obs_idx[-1].hour}\n'
                  f'shift_hours: {shift_hours}\n'
                  f'shift_mins: {shift_mins}\n'
                  f'shift_periods: {shift_periods}\n\n')

            group_data = group_data.shift(shift_periods)
        return group_data

    new_df = (
        stock_data
        .groupby(
            pd.Grouper(freq='D'), 
            group_keys=False
        )
        .apply(correct_shift)
    )
    if isinstance(new_df.index, pd.MultiIndex):
        new_df.index = new_df.index.droplevel(0)

    return new_df


def find_btc_fgi_data(
    intervals: str = Params.general_params.get('intervals'),
    periods: int = Params.data_params.get('periods'),
    accelerated: bool = False,
    category_features: Dict[str, List[str]] = None,
    save_mock: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    # Find wether or not to find info
    find_data = True

    if accelerated:
        # Find BTC Fear & Greed Index columns
        sentiment_btc_fgi_cols = category_features['sentiment_btc_fgi']

        if len(sentiment_btc_fgi_cols) == 0:
            find_data = False

    # BTC Fear & Greed Index (daily)
    if intervals == '1d' and find_data:
        data = pd.DataFrame(requests.get(f"https://api.alternative.me/fng/?limit={periods}").json()['data'])

        data['timestamp'] = data['timestamp'].apply(
            lambda t: datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d %H:%M')
        )
        data = (
            data
            .set_index('timestamp')
            .drop(columns=['time_until_update'])
            .rename(columns={
                'value': 'btc_fgi',
                'value_classification': 'btc_fgi_class'
            })
        )
        data.index = pd.to_datetime(data.index.to_series())

        data['btc_fgi'] = data['btc_fgi'].apply(lambda x: int(x))
        data = data.iloc[::-1]

        if debug:
            print(f'btc_fgi shape: {data.shape}\n'
                    f'btc_last tail: {data.tail(20)}\n')

        data.columns = 'sentiment_btc_fgi_' + data.columns
    else:
        data = None

    if save_mock:
        save_mock_asset(
            asset=data,
            asset_name='btc_fgi_data',
            intervals=intervals
        )

    return data


def find_economic_data(
    intervals: str = Params.general_params.get('intervals'),
    periods: int = Params.data_params.get('periods'),
    accelerated: bool = False,
    category_features: Dict[str, List[str]] = None,
    save_mock: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    # Find wether or not to find info
    find_data = True

    if accelerated:
        # Find economic columns
        economic_cols = category_features['economic']

        if len(economic_cols) == 0:
            find_data = False

    if intervals == '1d' and find_data:
        av_api_key = 'OW4T22MDSOAV06IT'

        # Treasury Yield
        url = rf'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={av_api_key}'
        req_ob = requests.get(url, verify=False)
        treasury_yield: pd.DataFrame = (
            pd.DataFrame(dict(req_ob.json())['data'])[::-1]
            .set_index('date')
            .rename(columns={'value': 'treasury_yield'})
            .applymap(lambda x: float(x) if x != '.' else 0)
        )

        # Federal Funds (Interest) Rate
        url = rf'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey={av_api_key}'
        req_ob = requests.get(url, verify=False)
        interest_rate: pd.DataFrame = (
            pd.DataFrame(dict(req_ob.json())['data'])[::-1]
            .set_index('date')
            .rename(columns={'value': 'interest_rate'})
            .applymap(lambda x: float(x) if x != '.' else 0)
        )

        treasury_yield.index = pd.to_datetime(treasury_yield.index)
        interest_rate.index = pd.to_datetime(interest_rate.index)
        economic_data = pd.concat([treasury_yield, interest_rate], axis=1).iloc[-periods:]

        economic_data.columns = 'economic_' + economic_data.columns

        if debug:
            print(f'economic_data: \n{economic_data.tail(20)}\n\n')
    else:
        economic_data = None

    if save_mock:
        save_mock_asset(
            asset=economic_data,
            asset_name='economic_data',
            intervals=intervals
        )

    return economic_data


def find_on_chain_data(
    intervals: str = Params.general_params.get('intervals'),
    periods: int = Params.data_params.get('periods'),
    accelerated: bool = False,
    category_features: Dict[str, List[str]] = None,
    expected_idx: pd.DatetimeIndex = None,
    save_mock: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    # Find wether or not to find info
    find_data = True

    if accelerated:
        # Find on_chain columns
        on_chain_cols = category_features['on_chain']

        if len(on_chain_cols) == 0:
            find_data = False

    try:
        # CRYPTOQUANT --> Esta tremendo, pero la versión gratis no te sirve

        # BLOCKCHAIN.COM
        if find_data:
            if intervals == '1d':
                # Total Number of Transactions
                url = f'https://api.blockchain.info/charts/n-transactions-total?timespan=3years&format=json'
                df1 = pd.DataFrame(
                    dict(requests.get(url, verify=False).json())['values']
                ).rename(columns={'y': 'total_transactions'}).set_index('x')

                # Miners Revenue (USD)
                url = 'https://api.blockchain.info/charts/miners-revenue?timespan=3years&format=json'
                df2 = pd.DataFrame(
                    dict(requests.get(url, verify=False).json())['values']
                ).rename(columns={'y': 'miners_revenue_usd'}).set_index('x')

                # Cost Per Transaction
                url = 'https://api.blockchain.info/charts/cost-per-transaction?timespan=3years&format=json'
                df3 = pd.DataFrame(
                    dict(requests.get(url, verify=False).json())['values']
                ).rename(columns={'y': 'cost_per_transaction'}).set_index('x')

                # Estimated Transaction Value
                url = 'https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=3years&format=json'
                df4 = pd.DataFrame(
                    dict(requests.get(url, verify=False).json())['values']
                ).rename(columns={'y': 'transaction_value'}).set_index('x')

                on_chain_data: pd.DataFrame = pd.concat([df1, df2, df3, df4], axis=1)
            else:
                # Transaction Rate Per Second
                url = 'https://api.blockchain.info/charts/transactions-per-second?format=json'
                on_chain_data: pd.DataFrame = pd.DataFrame(
                    dict(requests.get(url, verify=False).json())['values']
                ).rename(columns={'y': 'transaction_rate_per_second'}).set_index('x')

            on_chain_data.index = pd.to_datetime(on_chain_data.index.to_series(), unit='s')
            on_chain_data.columns = 'on_chain_' + on_chain_data.columns

            if debug:
                print(f'on_chain_data shape: {on_chain_data.shape}\n'
                      f'utc_now: {datetime.now(timezone.utc)}\n'
                      f'last index: {on_chain_data.index[-1]}\n'
                      f'on_chain_data: \n{on_chain_data.tail()}\n\n')

            if intervals == '60min':
                on_chain_data = on_chain_data.groupby(pd.Grouper(freq='60min')).first()
            elif intervals == '30min':
                on_chain_data = on_chain_data.groupby(pd.Grouper(freq='30min')).first()

            on_chain_data = on_chain_data.iloc[-periods:]

            if debug and intervals in ['60min', '30min']:
                print(f'on_chain_data after Grouper: \n{on_chain_data.tail()}\n\n')
        else:
            on_chain_data: pd.DataFrame = pd.DataFrame(
                np.zeros(shape=(len(expected_idx), 1)),
                columns=['on_chain_transaction_rate_per_second'],
                index=expected_idx
            )
    except Exception as e:
        LOGGER.warning(
            'Unble to retrieve On-Chain Data: (%s).\n'
            'on_chain_data was skipped.\n'
            'Exception: %s\n\n',
            intervals, e
        )
        on_chain_data: pd.DataFrame = None

    if save_mock:
        save_mock_asset(
            asset=on_chain_data,
            asset_name='on_chain_data',
            intervals=intervals
        )

    return on_chain_data
        

def save_mock_asset(
    asset: Any,
    asset_name: str,
    intervals: str,
    coin_name: str = None
) -> None:
    # Define base_path
    base_path: str = f"{Params.bucket}/mock/utils/collective_data"

    # Save asset
    if asset_name == 'loaded_collective_data':
        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/loaded_collective_data.parquet",
            overwrite=True
        )
    elif asset_name == 'data_columns':
        write_to_s3(
            asset=asset,
            path=f"{base_path}/data_columns.pickle",
            overwrite=True
        )
    elif asset_name == 'client_data':
        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/other_coins_data/{coin_name}/client_data.parquet",
            overwrite=True
        )
    # elif asset_name == 'other_coins_data':
    #     write_to_s3(
    #         asset=asset,
    #         path=f"{base_path}/{intervals}/other_coins_data.parquet",
    #         overwrite=True
    #     )
    elif asset_name == 'stocks_data':
        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/stocks_data.parquet",
            overwrite=True
        )
    elif asset_name == 'btc_fgi_data':
        if asset is None:
            asset = pd.DataFrame()

        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/btc_fgi_data.parquet",
            overwrite=True
        )
    elif asset_name == 'economic_data':
        if asset is None:
            asset = pd.DataFrame()

        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/economic_data.parquet",
            overwrite=True
        )
    elif asset_name == 'on_chain_data':
        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/on_chain_data.parquet",
            overwrite=True
        )
    elif asset_name == 'collective_data':
        write_to_s3(
            asset=asset,
            path=f"{base_path}/{intervals}/collective_data.parquet",
            overwrite=True
        )
    else:
        raise Exception(f'Invalid "asset_name" received: {asset_name}.\n')


def load_mock_asset(
    asset_name: str,
    intervals: str,
    coin_name: str = None
) -> pd.DataFrame | List[str]:
    # Define base_path
    base_path: str = f"{Params.bucket}/mock/utils/collective_data"

    # Load asset
    if asset_name == 'loaded_collective_data':
        return load_from_s3(
            path=f"{base_path}/{intervals}/loaded_collective_data.parquet",
            ignore_checks=True
        )
    elif asset_name == 'data_columns':
        return load_from_s3(
            path=f"{base_path}/data_columns.pickle",
            ignore_checks=True
        )
    elif asset_name == 'client_data':
        return load_from_s3(
            path=f"{base_path}/{intervals}/other_coins_data/{coin_name}/client_data.parquet",
            ignore_checks=True
        )
    # elif asset_name == 'other_coins_data':
    #     return load_from_s3(
    #         path=f"{base_path}/{intervals}/other_coins_data.parquet",
    #         ignore_checks=True
    #     )
    elif asset_name == 'stocks_data':
        return load_from_s3(
            path=f"{base_path}/{intervals}/stocks_data.parquet",
            ignore_checks=True
        )
    elif asset_name == 'btc_fgi_data':
        try:
            return load_from_s3(
                path=f"{base_path}/{intervals}/btc_fgi_data.parquet",
                ignore_checks=True
            )
        except:
            return None
    elif asset_name == 'economic_data':
        try:
            return load_from_s3(
                path=f"{base_path}/{intervals}/economic_data.parquet",
                ignore_checks=True
            )
        except:
            return None
    elif asset_name == 'on_chain_data':
        return load_from_s3(
            path=f"{base_path}/{intervals}/on_chain_data.parquet",
            ignore_checks=True
        )
    elif asset_name == 'collective_data':
        return load_from_s3(
            path=f"{base_path}/{intervals}/collective_data.parquet",
            ignore_checks=True
        )
    else:
        raise Exception(f'Invalid "asset_name" received: {asset_name}.\n')


def is_updated(
    loaded_data: pd.DataFrame,
    full_coin_cols: List[str],
    full_stock_cols: List[str],
    periods: int,
    debug: bool = False
) -> bool:
    loaded_coin_stock_cols = [c for c in loaded_data.columns if c in full_coin_cols + full_stock_cols]
    if len(loaded_coin_stock_cols) == len(full_coin_cols + full_stock_cols):
        last_update = loaded_data.index[-1]
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        if not (last_update < utc_now - pd.Timedelta(minutes=MBP * 2)):
            null_last_obs = [c for c in full_coin_cols if isnan(loaded_data[c].at[last_update])]

            if last_update.weekday() < 5:
                real_last_obs_time = (last_update + pd.Timedelta(value=MBP, unit='minutes')).time()
                start_time = datetime.strptime("13:30:00", "%H:%M:%S").time()
                end_time = datetime.strptime("19:30:00", "%H:%M:%S").time()

                if start_time <= real_last_obs_time <= end_time:
                    # Check if today is a holiday in the US
                    us_holidays = holidays.US()
                    today = date.today()

                    if today not in us_holidays:
                        null_last_obs.extend([c for c in full_stock_cols if isnan(loaded_data[c].at[last_update])])

            if debug:
                print(f'loaded_data.index[-1]: {loaded_data.index[-1]}\n'
                        f'last_update: {last_update}\n'
                        f'null_last_obs: {null_last_obs}\n'
                        f'{loaded_data[null_last_obs].tail()}\n\n')

            if len(null_last_obs) == 0: # and loaded_data.shape[0] >= periods:
                return True
            
            LOGGER.warning('Collective Data had to be re-updated.')
            if len(null_last_obs) != 0:
                LOGGER.warning(
                    'len(null_last_obs) == 0: %s --> '
                    'len(null_last_obs): %s\n'
                    'null_last_obs: \n%s\n'
                    'loaded_data[null_last_obs].tail(): \n%s\n',
                    len(null_last_obs) == 0, len(null_last_obs), null_last_obs, 
                    loaded_data[null_last_obs].tail().to_string()
                )
            if loaded_data.shape[0] < periods:
                LOGGER.warning(
                    'loaded_data.shape[0] >= periods: %s --> loaded_data.shape[0]: %s, periods: %s\n',
                    loaded_data.shape[0] >= periods, loaded_data.shape[0], periods
                )
            return False
        else:
            # print(f'loaded_data is not updated.\n'
            #       # f'loaded_coin_stock_cols ({loaded_coin_stock_cols})\n'
            #       f'last_update: {last_update}, datetime.now(timezone.utc) ({datetime.now(timezone.utc)}).\n\n')
            return False
    else:
        LOGGER.warning(
            'len(loaded_coin_stock_cols) (%s) != len(full_coin_cols + full_stock_cols) (%s).\n'
            'Will update collective_data.\n',
            len(loaded_coin_stock_cols), len(full_coin_cols + full_stock_cols)
        )
        return False

        
@timing
def get_collective_data(
    client: BinanceClient, 
    loaded_collective_data: pd.DataFrame = None,
    accelerated: bool = False,
    category_features: Dict[str, List[str]] = None,
    other_coins: list = None, 
    intervals: str = '30min', 
    periods: int = 10000,
    yfinance_params: dict = None,
    parallel: bool = False, 
    skip_check: bool = False,
    validate: bool = True,
    save: bool = True,
    overwrite: bool = True,
    save_mock: bool = False,
    ignore_last_period_check: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    # Define wether or not to load a reduced dataset
    if overwrite:
        load_reduced_dataset = False
    else:
        load_reduced_dataset = True
    
    # Load Data
    if loaded_collective_data is None:
        loaded_data: pd.DataFrame = load_collective_data(
            intervals=intervals,
            load_reduced_dataset=load_reduced_dataset,
            save_mock=save_mock
        )
    else:
        loaded_data: pd.DataFrame = loaded_collective_data

    # Update periods, if a reduced dataset was loaded
    if load_reduced_dataset:
        periods = loaded_data.shape[0] # np.min([loaded_data.shape[0], periods])

    # Collective Data Columns
    full_coin_cols, full_stock_cols = find_collective_data_columns(
        other_coins=other_coins,
        save_mock=save_mock,
        debug=debug
    )
    
    if loaded_data is not None:
        if not skip_check and is_updated(
            loaded_data=loaded_data,
            full_coin_cols=full_coin_cols,
            full_stock_cols=full_stock_cols,
            periods=periods,
            debug=debug
        ):
            return loaded_data.iloc[-periods - 1:]

    LOGGER.info('Updating collective_data (periods: %s)...', periods)

    # Load other coins data
    other_coins_data: pd.DataFrame = find_other_coins_data(
        client=client,
        intervals=intervals,
        periods=periods,
        parallel=parallel,
        other_coins=other_coins,
        full_coin_cols=full_coin_cols,
        accelerated=accelerated,
        category_features=category_features,
        save_mock=save_mock,
        ignore_last_period_check=ignore_last_period_check,
        debug=debug
    )

    if other_coins_data is None:
        print(f'Re-trying to fetch other_coins_data, with "parallel" turned off.\n'
              f'Initially tried with parallel = {parallel}.\n\n')
        other_coins_data: pd.DataFrame = find_other_coins_data(
            client=client,
            intervals=intervals,
            periods=periods,
            parallel=False,
            other_coins=other_coins,
            full_coin_cols=full_coin_cols,
            accelerated=accelerated,
            category_features=category_features,
            save_mock=save_mock,
            debug=debug
        )

    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    if other_coins_data.index[-1] > utc_now - pd.Timedelta(minutes=MBP):
        LOGGER.info("Removing other_coins_data last obs.")
        other_coins_data = other_coins_data.iloc[:-1]

    # Load Stocks Data
    stocks_data: pd.DataFrame = find_stock_data(
        intervals=intervals,
        full_stock_cols=full_stock_cols,
        yfinance_params=yfinance_params,
        accelerated=accelerated,
        category_features=category_features,
        expected_idx=other_coins_data.index,
        save_mock=save_mock,
        debug=debug
    )
    
    if stocks_data is not None and other_coins_data is not None:
        stocks_data = stocks_data.loc[stocks_data.index.isin(other_coins_data.index)]

    stocks_data: pd.DataFrame = correct_stock_data(stocks_data)
    
    # BTC Fear & Greed Index
    btc_fgi_data: pd.DataFrame = find_btc_fgi_data(
        intervals=intervals,
        periods=periods,
        accelerated=accelerated,
        category_features=category_features,
        save_mock=save_mock,
        debug=debug
    )
    if btc_fgi_data is not None and other_coins_data is not None:
        btc_fgi_data = btc_fgi_data.loc[btc_fgi_data.index.isin(other_coins_data.index)]

    # Load Economic Data
    economic_data: pd.DataFrame = find_economic_data(
        intervals=intervals,
        periods=periods,
        accelerated=accelerated,
        category_features=category_features,
        save_mock=save_mock,
        debug=debug
    )
    if economic_data is not None and other_coins_data is not None:
        economic_data = economic_data.loc[economic_data.index.isin(other_coins_data.index)]

    # Load On-Chain Data
    on_chain_data: pd.DataFrame = find_on_chain_data(
        intervals=intervals,
        periods=periods,
        accelerated=accelerated,
        category_features=category_features,
        expected_idx=other_coins_data.index,
        save_mock=save_mock,
        debug=debug
    )
    if on_chain_data is not None and other_coins_data is not None:
        on_chain_data = on_chain_data.loc[on_chain_data.index.isin(other_coins_data.index)]

    # Concat Data
    if not (
        other_coins_data is None 
        and stocks_data is None 
        and btc_fgi_data is None
        and economic_data is None 
        and on_chain_data is None
    ):
        # print(f'Checking concat data.\n'
        #       f'other_coins_data: {other_coins_data.tail()}\n'
        #       f'stocks_data: {stocks_data.tail()}\n'
        #       f'btc_fgi_data: {btc_fgi_data}\n'
        #       f'economic_data: {economic_data}\n'
        #       f'on_chain_data: {on_chain_data.tail()}\n\n')
        new_data: pd.DataFrame = pd.concat([other_coins_data, stocks_data, btc_fgi_data, economic_data, on_chain_data], axis=1)
        new_data[on_chain_data.columns] = new_data[on_chain_data.columns].shift(1).bfill()
        if debug:
            print(f'new_data.tail(10): \n{new_data.tail(10)}\n\n')
    else:
        LOGGER.warning(
            'Unable to concat new full_data.\n'
            'other_coins_data: %s\n'
            'stocks_data: %s\n'
            'btc_fgi_data: %s\n'
            'economic_data: %s\n'
            'on_chain_data: %s\n',
            other_coins_data, stocks_data, btc_fgi_data, economic_data, on_chain_data
        )
        return None

    # Combine new data with loaded data
    if loaded_data is not None:
        intersection_cols = loaded_data.columns.intersection(new_data.columns)
        collective_data = (
            loaded_data[intersection_cols].iloc[:-20]
            .combine_first(new_data[intersection_cols])
            .combine_first(loaded_data[intersection_cols])
            .sort_index(ascending=True)
        )
        collective_data: pd.DataFrame = collective_data.loc[~collective_data.index.duplicated(keep='last')]
    else:
        collective_data: pd.DataFrame = new_data.copy()

    # Validate Data
    if validate:
        collective_data: pd.DataFrame = validate_collective_data(
            collective_data=collective_data,
            intervals=intervals,
            repair=True,
            debug=debug,
            **{'expected_periods': periods}
        )
    
    # Save Collective Data
    if save and not accelerated:
        save_collective_data(
            df=collective_data,
            intervals=intervals,
            overwrite=overwrite,
            **{'expected_periods': periods}
        )

    if debug:
        print(f'collective_data.tail(): \n{collective_data.tail(10)}\n\n')
        # msno.matrix(collective_data.iloc[-periods:])
        # plt.show()

    # LOGGER.info('Done.')

    # Delete unnecessary datasets
    del other_coins_data
    del stocks_data
    del btc_fgi_data
    del economic_data
    del on_chain_data

    collective_data: pd.DataFrame = collective_data.iloc[-periods - 1:]
    
    if save_mock:
        save_mock_asset(
            asset=collective_data,
            asset_name='collective_data',
            intervals=intervals
        )

    return collective_data


def update_collective_data_expectations():
    # Define asset_path
    s3_asset_path = f"{Params.bucket}/utils/collective_data/{Params.general_params.get('intervals')}/collective_data.parquet"

    # Find Other Coins
    other_coins = Params.other_coins_json[Params.general_params.get("intervals")][:Params.data_params['other_coins_n']]

    # Find Expected Columns
    coin_cols, stock_cols = find_collective_data_columns(other_coins)
    expected_columns: List[str] = coin_cols + stock_cols + ['on_chain_transaction_rate_per_second']

    # Define expected schema
    expected_schema = {
        col: 'float' for col in expected_columns
    }

    # Define max values allowed
    max_values_allowed = None

    # Define min values allowed
    min_values_allowed = {
        col: 0.0 for col in expected_columns
        if not(col.endswith('return') or col.endswith('acceleration') or col.endswith('jerk'))
    }

    # Unique values allowed
    unique_values_allowed = None

    # Null values percentage allowed
    def find_null_perc_allowed(col: str):
        if (
            col.startswith('target') 
            or col.startswith('coin')
            or col.startswith('other_coins')
            or col.startswith('on_chain')
        ):
            return 0.05
        if col.startswith('stock'):
            return 0.85
        return 0.0
        
    null_perc_allowed: Dict[str, float] = {
        col: find_null_perc_allowed(col) for col in expected_columns
    }

    # Define duplicate rows subset
    duplicate_rows_subset = expected_columns.copy()

    # Find Expected Periods
    expected_periods = Params.data_params.get("periods")

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
        "check_extreme_values": False,
        "check_excess_features": False,
        "check_short_length": True,

        "expected_cols": expected_columns,
        "expected_schema": expected_schema,
        "max_values_allowed": max_values_allowed,
        "min_values_allowed": min_values_allowed,
        "unique_values_allowed": unique_values_allowed,
        "null_perc_allowed": null_perc_allowed,

        "duplicate_rows_subset": duplicate_rows_subset,
        "outliers_dict": None,
        "max_features_perc": None,
        "other_coins": other_coins,
        "expected_periods": expected_periods
    }

    LOGGER.debug('Collective data expectations:\n%s\n', pformat(s3_expectations))

    # Save expectations
    write_to_s3(
        asset=s3_expectations,
        path=f"{Params.bucket}/utils/expectations/{Params.general_params.get('intervals')}/collective_data_expectations.json"
    )


def diagnose_collective_data(
    collective_data: pd.DataFrame = None,
    intervals: str = Params.general_params.get("intervals"),
    debug: bool = False,
    **update_expectations
) -> Dict[str, bool]:
    # Load Collective Data
    if collective_data is None:
        collective_data = load_collective_data(intervals=intervals)

    # Find Diagnostics Dict
    diagnostics_dict = find_data_diagnosis_dict(
        df_name="collective_data",
        intervals=intervals,
        coin_name=None,
        df=collective_data,
        debug=debug,
        **update_expectations
    )

    if debug:
        print(f'collective_data diagnostics_dict:')
        pprint(diagnostics_dict)
        print('\n\n')

    return diagnostics_dict


def validate_collective_data(
    collective_data: pd.DataFrame = None,
    intervals: str = Params.general_params.get("intervals"),
    repair: bool = True,
    debug: bool = False,
    **update_expectations
) -> pd.DataFrame:
    # Load Collective Data
    if collective_data is None:
        collective_data = load_collective_data(intervals=intervals)

    # Find Diagnostics Dict
    diagnostics_dict = diagnose_collective_data(
        collective_data=collective_data.copy(),
        intervals=intervals,
        debug=debug,
        **update_expectations
    )

    if needs_repair(diagnostics_dict):
        LOGGER.warning(
            '"collective_data" needs repair.\n'
            'diagnostics_dict:\n%s\n',
            pformat(diagnostics_dict)
        )

        if repair:
            """
            diagnostics_dict:
                - has_missing_columns
                - has_missing_rows
                - has_null_values
                - has_duplicated_idx
                - has_duplicated_columns
                - has_short_length
            """
            # Load Expectations
            # File System
            # expectations: dict = json.load(open(os.path.join(
            #     Params.base_cwd, Params.bucket, "utils", "expectations", intervals, 
            #     "collective_data_expectations.json"
            # )))

            # S3
            expectations: dict = load_from_s3(
                path=f"{Params.bucket}/utils/expectations/{intervals}/collective_data_expectations.json"
            )

            # Add Missing Rows
            collective_data = add_missing_rows(
                df=collective_data.copy(),
                intervals=intervals
            )

            # Drop Unexpected Columns
            collective_data = drop_unexpected_columns(
                df=collective_data.copy(),
                expected_columns=expectations['expected_cols'],
                coin_name=None,
                df_name='collective_data'
            )

            # Repair Other Coins
            collective_data = repair_coin_cols(
                df=collective_data.copy(),
                other_coins=expectations['other_coins'],
                intervals=intervals,
                price_only=False
            )

            # Repair Stock & On-Chain cols
            collective_data = repair_stock_and_on_chain_cols(
                df=collective_data.copy(),
                intervals=intervals
            )

            # Drop duplicated IDX & Cols
            collective_data = drop_duplicates(
                df=collective_data.copy()
            )

            # Find Dummy Collective Data
            other_coins = Params.other_coins_json[intervals][:Params.data_params.get("other_coins_n")]

            dummy_df = find_dummy_collective_data(
                other_coins=other_coins,
                intervals=intervals,
                collective_data_idx=collective_data.index
            )
            
            # Complement Collective Data
            collective_data = collective_data.combine_first(dummy_df)
        else:
            LOGGER.warning(
                '"collective_data" needed repair, but "repair" parameter was set to False.\n'
                'Thus, collective_data will NOT be repaired.\n'
            ) 
            
    return collective_data
