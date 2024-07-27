from PyTradeX.config.params import Params
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.s3_helper import (
    write_to_s3, 
    load_from_s3,
    get_secrets
)
from binance import Client
import requests
import certifi
from concurrent.futures import ThreadPoolExecutor
import time
import math
import pandas as pd
import numpy as np
import requests
import hashlib
import hmac
import warnings
from datetime import datetime, timezone
from pprint import pformat
from typing import List, Dict

# Set the CA certificate bundle for requests
requests.packages.urllib3.util.ssl_.DEFAULT_CA_BUNDLE_PATH = certifi.where()

# Suppress warnings
requests.packages.urllib3.disable_warnings()

# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.log_params.get('level'),
    txt_fmt="%(name)s: %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(blue)s%(filename)s:%(lineno)s%(reset)s | [%(log_color)s%(account_id)s%(reset)s] %(log_color)s%(message)s%(reset)s",
    json_fmt="%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(account_id)s %(message)s",
    filter_lvls=Params.log_params.get('filter_lvls'),
    log_file=Params.log_params.get('log_file'),
    backup_count=Params.log_params.get('backup_count')
)

# Load binance_keys
BINANCE_KEYS = get_secrets(secret_name='binance_keys')

# TODO: APPLY FOR A REBATE! 
#   - https://www.binance.com/en/support/faq/how-to-apply-for-binance-link-program-api-rebate-2ae5b076d3834e1480f78c19898b213f
#   - https://www.binance.com/en/support/announcement/binance-launches-binance-link-benefit-system-0993b73a57b24659bdc6199ac072f965
#   - Check other ways for rebates: https://www.binance.com/en/fee/futureFee
#   - Check binance convert


class BinanceClient:

    full_coin_list: List[str] = Params.fixed_params.get("full_coin_list")
    stable_coins: List[str] = Params.fixed_params.get("full_stable_coin_list")

    def __init__(
        self,
        account_id: int = 0,
        stable_coin: str = Params.general_params.get('stable_coin') 
    ) -> None:
        def load_client(api_key: str, api_secret: str):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return Client(
                    api_key=api_key,
                    api_secret=api_secret,
                    requests_params={
                        "verify": False, 
                        # "use_server_time": True,
                        "timeout": 30 # 20
                    }
                )
        
        # Assign self.account_id
        self.account_id: int = account_id

        # Assign extra arguments
        self.extra: Dict[str, int] = {"account_id": f"account_id: {self.account_id}"}

        # Find account API keys
        api_key = BINANCE_KEYS[f"BINANCE_API_KEY_{self.account_id}"]
        api_secret = BINANCE_KEYS[f"BINANCE_API_SECRET_{self.account_id}"]
        
        i = 0
        self.client = None
        while i < 5 and self.client is None:
            try:
                self.client = load_client(
                    api_key=api_key,
                    api_secret=api_secret
                )
            except Exception as e:
                LOGGER.warning(
                    'Unable to create self.client\n'
                    'Exception: {e}\n', 
                    e, extra=self.extra
                )
                
                time.sleep(1)
                i += 1

        # Set self.stable_coin
        self.stable_coin: str = stable_coin

        # Load available pairs
        self.available_pairs: dict = self.load_available_pairs()

        if len(self.available_pairs) == 0:
            LOGGER.warning('Forcing BinanceClient.update_available_pairs().', extra=self.extra)
            self.update_available_pairs()

        # Load precision
        self.quantity_precision = self.load_precision()

        if len(self.quantity_precision) == 0:
            LOGGER.warning('Forcing BinanceClient.update_quantity_precision().', extra=self.extra)
            self.update_quantity_precision()

        # Assert that precision was loaded for all coins
        assert len([c for c in self.full_coin_list if c not in self.quantity_precision]) == 0

        # Define s3_last_prices_path
        self.s3_last_prices_path = f"{Params.bucket}/utils/last_prices/{Params.general_params.get('intervals')}/last_prices.pickle"
        
        # Define diff_dict
        # self.diff_dict: dict = {}

        # Define price_precision
        self.price_precision: Dict[str, float] = {}

    """
    General Methods
    """

    @property
    def last_prices(self) -> dict:
        try:
            return load_from_s3(
                path=self.s3_last_prices_path
            )
        except Exception as e:
            LOGGER.warning(
                'Unable to load last_prices.\n'
                'Exception: %s\n', 
                e, extra=self.extra
            )
            return {}

    def update_last_prices(
        self, 
        coin_list: list, 
        intervals: str = Params.general_params.get("intervals"),
        futures: bool = True
    ) -> dict:
        def dummy_fun(coin_name: str):
            df = self.get_data(
                coin_name=coin_name, 
                intervals=intervals, 
                periods=10,
                futures=futures,
                update_last_prices=False,
                ignore_last_period=False
            )
            if df.shape[0] == 0:
                df = self.get_data(
                    coin_name=coin_name, 
                    intervals=intervals, 
                    periods=10,
                    futures=futures,
                    update_last_prices=False,
                    ignore_last_period=False
                )

            last_idx = df.index[-1]
            last_prices[coin_name] = (df.at[last_idx, 'open'], df.at[last_idx, 'price'])
        
        # Define last_prices
        coin_list = [c for c in coin_list if c not in self.stable_coins]

        last_prices = self.last_prices

        last_prices['USDT'] = (1, 1)
        last_prices['BUSD'] = (1, 1)
        last_prices['USDC'] = (1, 1)

        # Fill last_prices
        if len(coin_list) > 0:
            with ThreadPoolExecutor(max_workers=len(coin_list)) as executor:
                for name in coin_list:
                    time.sleep(0.01)
                    executor.submit(dummy_fun, name)

        # Re-fetch prices that were not fetched
        for coin_name in coin_list:
            if coin_name not in last_prices.keys():
                LOGGER.warning('%s had to be re-fetched to fill last_prices.', coin_name, extra=self.extra)
                dummy_fun(coin_name=coin_name)
        
        # Save Last Prices
        write_to_s3(
            asset=last_prices,
            path=self.s3_last_prices_path
        )

    def load_available_pairs(self) -> dict:
        try:
            return load_from_s3(
                path=f"{Params.bucket}/utils/available_pairs/available_pairs.json"
            )
        except Exception as e:
            LOGGER.warning(
                'Unable to load available_pairs.\n'
                'Exception: %s.\n', e, extra=self.extra
            )
            return {}

    def update_available_pairs(self) -> None:
        LOGGER.info('Updating client available_pairs.', extra=self.extra)

        # Extract futures_exchange_info from Binance client.
        i = 0
        info = None
        while info is None and i < 3:
            try:
                info = self.client.futures_exchange_info()
            except Exception as e:
                LOGGER.warning(
                    'Unable to load futures_exchange_info.\n'
                    'Exception: %s\n'
                    'Retrying.\n', e, extra=self.extra
                )
                
                time.sleep(3)
                i += 1

        # Populate self.available_pairs
        self.available_pairs = {
            stable_coin: [
                si['symbol'][:-4] for si in info['symbols'] 
                if si['symbol'][-4:] == stable_coin and si['symbol'][:-4] in self.full_coin_list
            ] for stable_coin in ['USDT', 'USDC']
        }

        LOGGER.info('available_pairs:\n%s\n', pformat(self.available_pairs), extra=self.extra)

        # Update available_pairs.json
        write_to_s3(
            asset=self.available_pairs,
            path=f"{Params.bucket}/utils/available_pairs/available_pairs.json"
        )

    def load_precision(self) -> dict:
        try:
            return load_from_s3(
                path=f"{Params.bucket}/utils/precision/precision.json"
            )
        except Exception as e:
            LOGGER.warning(
                'Unable to load precision.\n'
                'Exception: %s.\n', e, extra=self.extra
            )
            return {}

    def update_quantity_precision(self) -> None:
        LOGGER.info('Updating client precision.', extra=self.extra)

        # Extract futures_exchange_info from Binance client.
        i = 0
        info = None
        while info is None and i < 3:
            try:
                info = self.client.futures_exchange_info()
            except Exception as e:
                LOGGER.warning(
                    'Unable to load futures_exchange_info.\n'
                    'Exception: %s\n'
                    'Retrying.\n', 
                    e, extra=self.extra
                )
                time.sleep(3)
                i += 1
        
        # Define pairs to search on
        pairs = [c + 'USDT' for c in self.full_coin_list]

        # Find self.quantity_precision
        self.quantity_precision = {
            si['symbol'][:-4]: (int(10 ** si['quantityPrecision']), si['quantityPrecision']) 
            for si in info['symbols'] if si['symbol'] in pairs
        }

        # Assert that all pairs were found
        assert len(self.quantity_precision) == len(self.full_coin_list)

        LOGGER.info('precision:\n%s\n', pformat(self.quantity_precision), extra=self.extra)

        # Update precision.json
        write_to_s3(
            asset=self.quantity_precision,
            path=f"{Params.bucket}/utils/precision/precision.json"
        )
    
    def extract_stable_coin(
        self,
        coin_name: str,
        forced_stable_coin: str = None
    ) -> str:
        # Find stable_coin
        if forced_stable_coin is not None:
            return forced_stable_coin
        else:
            if coin_name in self.available_pairs[self.stable_coin]:
                return self.stable_coin
            else:
                return 'USDT'

    """
    Data Extraction Methods
    """

    def get_data(
        self, 
        coin_name: str, 
        intervals: str = Params.general_params.get("intervals"), 
        periods: int = Params.data_params.get('periods'),
        futures: bool = True,
        forced_stable_coin: str = None,
        update_last_prices: bool = False,
        ignore_last_period: bool = True
    ) -> pd.DataFrame:
        def extract_binance_datasets(
            coin_name: str,
            stable_coin: str,
            interval: str,
            start_str: str
        ) -> pd.DataFrame:
            # Extract datasets Binance client
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if futures:
                    # Extract futures datasets
                    data = pd.DataFrame(self.client.futures_historical_klines(
                        symbol=coin_name.upper() + stable_coin,
                        interval=interval,
                        start_str=start_str
                    )).iloc[:, :6]
                else:
                    # Extract spot datasets
                    data = pd.DataFrame(self.client.get_historical_klines(
                        symbol=coin_name.upper() + stable_coin,
                        interval=interval,
                        start_str=start_str
                    )).iloc[:, :6]
            
            return data

        def filter_inconsistent_values(
            data: pd.DataFrame
        ) -> pd.DataFrame:
            data['ret'] = data['price'].diff()
            data['accel'] = data['ret'].diff()
            data['jerk'] = data['accel'].diff()

            mask = (
                (data['ret'] == 0) &
                (data['accel'] == 0) &
                (data['jerk'] == 0)
            )

            return data.loc[~mask].drop(columns=['ret', 'accel', 'jerk'])
        
        def find_data(
            coin_name: str, 
            stable_coin: str,
            interval: str,
            start_str: str
        ) -> pd.DataFrame:
            # Extract datasets Binance client
            data = extract_binance_datasets(
                coin_name=coin_name,
                stable_coin=stable_coin,
                interval=interval,
                start_str=start_str
            )

            # Rename columns
            if data.shape[0] == 0:
                # Log warning
                LOGGER.warning(
                    "%s-%s (%s) raw datasets extracted from binance.Client is null.", 
                    coin_name, stable_coin, intervals, extra=self.extra
                )

                # Define empty DataFrame
                data = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'price', 'volume'])
            else:
                data = data.rename(columns={
                    0: 'time',
                    1: 'open',
                    2: 'high',
                    3: 'low',
                    4: 'price',
                    5: 'volume'
                }).set_index('time').astype(float)

            # Define datetime index
            data.index = pd.to_datetime(data.index, unit='ms')
            
            # Filter inconsistent prices
            data = filter_inconsistent_values(data)
            
            return data

        # Get historical futures klines from Binance
        interval_dict = {
            '1d': self.client.KLINE_INTERVAL_1DAY,
            '60min': self.client.KLINE_INTERVAL_1HOUR,
            '30min': self.client.KLINE_INTERVAL_30MINUTE,
            '15min': self.client.KLINE_INTERVAL_15MINUTE,
            '5min': self.client.KLINE_INTERVAL_5MINUTE
        }

        # Define start_str dict
        start_str_dict = {
            '1d': '1000 days ago UTC',
            '60min': f'{int(math.ceil(periods/24))} days ago UTC',
            '30min': f'{int(math.ceil(periods/48))} days ago UTC',
            '15min': f'{int(math.ceil(periods/96))} days ago UTC',
            '5min': f'{int(math.ceil(periods/192))} days ago UTC'
        }

        # Extract stable_coin
        stable_coin = self.extract_stable_coin(
            coin_name=coin_name,
            forced_stable_coin=forced_stable_coin
        )
        
        # Extract data
        try:
            df: pd.DataFrame = find_data(
                coin_name=coin_name,
                stable_coin=stable_coin,
                interval=interval_dict[intervals],
                start_str=start_str_dict[intervals]
            )
        except Exception as e:
            LOGGER.warning(
                'Unable to call the self.client.futures_historical_klines() method on %s.\n'
                'Exception: %s\n'
                'Rtrying after waiting 3 sec.\n', 
                coin_name.upper() + stable_coin, e, extra=self.extra
            )

            time.sleep(3)

            df: pd.DataFrame = find_data(
                coin_name=coin_name,
                stable_coin=stable_coin,
                interval=interval_dict[intervals],
                start_str=start_str_dict[intervals]
            )

        # Check periods
        if df.shape[0] < periods - 1:
            LOGGER.warning(
                'new %s data has less periods than required.\n'
                'df.shape[0]: %s\n'
                'periods: %s\n'
                'Re-fetching df with USDT.\n', 
                coin_name, 
                df.shape[0],
                periods,
                extra=self.extra
            )

            # Re-fetch data with USDT stable_coin
            df = find_data(
                coin_name=coin_name,
                stable_coin='USDT',
                interval=interval_dict[intervals],
                start_str=start_str_dict[intervals]
            )

        # Update last_opens
        if update_last_prices:
            last_prices = self.last_prices
            
            last_idx = df.index[-1]
            last_prices[coin_name] = (df.at[last_idx, 'open'], df.at[last_idx, 'price'])

            # Update last_prices.json
            write_to_s3(
                asset=last_prices,
                path=self.s3_last_prices_path
            )

        # Return datasets
        if ignore_last_period:
            return df.iloc[-periods-1:-1]
        return df.iloc[-periods-1:]

    def get_long_short_data(
        self, 
        coin_name: str, 
        from_: str = None, 
        intervals: str = Params.general_params.get("intervals"),
        periods: int = Params.data_params.get('periods')
    ) -> pd.DataFrame:
        def extract_longshort_ratio(from_: str) -> pd.DataFrame:
            if from_ == 'global':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return pd.DataFrame(self.client.futures_global_longshort_ratio(
                        symbol=coin_name.upper() + stable_coin,
                        period=period_dict[intervals],
                        limit=min([periods, 500])
                    ))
            elif from_ == 'top_traders':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return pd.DataFrame(self.client.futures_top_longshort_account_ratio(
                        symbol=coin_name.upper() + stable_coin,
                        period=period_dict[intervals],
                        limit=min([periods, 500])
                    ))
            raise Exception(f'Invalid "from_" parameter was received: {from_}.')

        # Validate from_ parameter
        if from_ not in [None, 'global', 'top_traders']:
            raise Exception(f'Invalid "from_" parameter.\n')
        if from_ is None:
            from_ = 'global'

        # Define period_dict to pass to period parameter
        period_dict = {
            '1d': '1d',
            '60min': '1h',
            '30min': '30m',
            '15min': '15m'
        }

        # Set stable_coin
        stable_coin = 'USDT'
            
        # Extract longshort_ratio
        try:
            long_short_df = extract_longshort_ratio(from_=from_)
        except Exception as e:
            LOGGER.warning(
                'Unable to extract long_short_ratio Dataset.\n'
                'Exception: %s\n'
                'Rtrying after waiting 5 sec.\n', 
                e, extra=self.extra
            )                
            time.sleep(5)

            long_short_df = extract_longshort_ratio(from_=from_)
        
        # Define long_short_df
        long_short_df: pd.DataFrame = (
            long_short_df
            .drop(columns=['symbol', 'shortAccount'])
            .rename(columns={
                'longAccount': f'{from_}_long_perc',
                'longShortRatio': f'{from_}_long_short_ratio'
            })
            .set_index('timestamp')
            .astype(float)
        )

        # Transform index into pd.DatetimeIndex
        long_short_df.index = pd.to_datetime(long_short_df.index, unit='ms')

        return long_short_df
    
    """
    Account information Methods
    """

    def get_earn_balance(self):
        """
        https://binance-docs.github.io/apidocs/spot/en/#get-flexible-subscription-record-user_data
        
        client = BinanceClient(account_id=0)

        # Current timestamp (milliseconds)
        timestamp = int(time.time() * 1000)

        # Request parameters
        params = {
            'timestamp': timestamp
        }

        # Create a query string from the parameters
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])

        # Extract api_key & api_secret
        api_key = BINANCE_KEYS[f"BINANCE_API_KEY_{client.account_id}"]
        api_secret = BINANCE_KEYS[f"BINANCE_API_SECRET_{client.account_id}"]

        # Create a signature using HMAC-SHA256
        signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256) # .hexdigest()

        # Add the signature and API key to the parameters
        params['signature'] = signature
        params['apiKey'] = api_key

        # Define url
        url = "https://api.binance.com/sapi/v1/simple-earn/locked/history/subscriptionRecord"

        # Define headers
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
            'X-MBX-APIKEY': api_key
        }

        # Get response from binance API
        response = requests.request("GET", url, headers=headers, params=params).json()
        print(pformat(response))
        """
        """
        https://python-binance.readthedocs.io/en/latest/_modules/binance/client.html#AsyncClient.get_account_api_permissions

        def get_account(self, **params):
            return self._get('account', True, data=params)

        def _get(self, path, signed=False, version=BaseClient.PUBLIC_API_VERSION, **kwargs):
            return self._request_api('get', path, signed, version, **kwargs)

        def _request_api(
            self, method, path: str, signed: bool = False, version=BaseClient.PUBLIC_API_VERSION, **kwargs
        ):
            uri = self._create_api_uri(path, signed, version)
            return self._request(method, uri, signed, **kwargs)

        def _create_api_uri(self, path: str, signed: bool = True, version: str = PUBLIC_API_VERSION) -> str:
            # API_URL = 'https://api{}.binance.{}/api'
            url = self.API_URL
            if self.testnet:
                url = self.API_TESTNET_URL
            v = self.PRIVATE_API_VERSION if signed else version
            return url + '/' + v + '/' + path

        def _request(self, method, uri: str, signed: bool, force_params: bool = False, **kwargs):
            kwargs = self._get_request_kwargs(method, signed, force_params, **kwargs)

            self.response = getattr(self.session, method)(uri, **kwargs)
            return self._handle_response(self.response)
        """
        pass

    def get_spot_balance(
        self, 
        include_usd_valuation: bool = False,
        intervals: str = Params.general_params.get("intervals"),
        debug: bool = False
    ) -> pd.Series:
        # Define column_types & rename_columns
        column_types = {
            'asset': str,
            'free': float,
            'locked': float
        }

        rename_columns = {
            'asset': 'coin_name',
            'free': 'available_balance',
            'locked': 'locked_balance'
        }

        # Extract spot balances
        balance_df = (
            pd.DataFrame(self.client.get_account()['balances'])
            .filter(items=column_types.keys())
            .astype(column_types)
            .rename(columns=rename_columns)
            .set_index('coin_name')
        )

        # Keep positive balances
        balance_df = balance_df.loc[
            (balance_df['available_balance'] > 0) |
            (balance_df['locked_balance'] > 0)
        ]

        # Define total balance
        balance_df['total_balance'] = (
            balance_df['available_balance'] 
            + balance_df['locked_balance']
        )

        # Correct BETH
        balance_df.at['ETH', 'total_balance'] = (
            balance_df.at['ETH', 'total_balance'] 
            + balance_df.at['BETH', 'total_balance']
        )

        # Drop unrequired assets
        balance_df.drop(['BETH', 'ETHW', 'LDUSDT'], inplace=True)

        if include_usd_valuation:
            # Extract balance coins
            coin_list = balance_df.index.to_series().unique().tolist()

            # Assert that intervals is not None
            assert intervals is not None, "Intervals is None!"

            # Update last_prices & extract output
            self.update_last_prices(
                coin_list=coin_list,
                intervals=intervals,
                futures=False
            )

            last_prices = self.last_prices

            if debug:
                formatted_last_prices = pformat({
                    k: v for k, v in last_prices.items() if k in coin_list
                })
                LOGGER.debug('last_prices:\n%s\n', formatted_last_prices, extra=self.extra)
            
            # Create usd_price
            balance_df['usd_price'] =  balance_df.index.to_series().apply(lambda c: last_prices[c][1])

            # Calculate total_balance_usd_amount
            balance_df['total_balance_usd_amount'] = np.abs(balance_df['total_balance']) * balance_df['usd_price']

            # Drop low balances
            balance_df = balance_df.loc[balance_df['total_balance_usd_amount'] > 10]

        # Sort balance_df
        balance_df.sort_values(
            by='total_balance_usd_amount', 
            ascending=False, 
            inplace=True
        )

        if debug:
            LOGGER.debug('Spot balance_df:\n%s\n', balance_df.to_string(), extra=self.extra)

        return balance_df['total_balance_usd_amount']

    def get_futures_balance(
        self,
        include_usd_valuation: bool = False,
        intervals: str = Params.general_params.get("intervals"),
        debug: bool = False
    ) -> pd.DataFrame:
        # Define column_types & rename_columns
        column_types = {
            'asset': str,
            'balance': float,
            'availableBalance': float
        }

        rename_columns = {
            'asset': 'coin_name',
            'availableBalance': 'available_balance'
        }

        # Extract futures balances
        balance_df = (
            pd.DataFrame(self.client.futures_account_balance())
            .filter(items=column_types.keys())
            .astype(column_types)
            .rename(columns=rename_columns)
        )

        # Keep positive balances
        balance_df = balance_df.loc[balance_df['balance'] > 0]

        # Insert period & account_id
        balance_df['period'] = datetime.today()
        balance_df['account_id'] = self.account_id

        # Set period as index
        balance_df.set_index('period', inplace=True)

        # Define empty agg_balance_df
        agg_balance_df: pd.DataFrame = None

        for coin_name in balance_df['coin_name'].unique():
            # Filter balance_df
            filtered_balance_df: pd.DataFrame = balance_df.loc[balance_df['coin_name'] == coin_name]

            # Group indexes on 30min intervals
            new_agg_balance_df = (
                filtered_balance_df                
                .groupby(pd.Grouper(freq='30min'))
                .last()
                .reset_index(drop=False)
            )

            # Append new_agg_balance_df
            agg_balance_df = pd.concat(
                [agg_balance_df, new_agg_balance_df], axis=0, ignore_index=True
            )

        if include_usd_valuation:
            # Extract balance coins
            coin_list = agg_balance_df['coin_name'].unique().tolist()

            # Assert that intervals is not None
            assert intervals is not None, "Intervals is None!"

            # Update last_prices & extract output
            self.update_last_prices(
                coin_list=coin_list,
                intervals=intervals,
                futures=True
            )

            last_prices = self.last_prices

            if debug:
                formatted_last_prices = pformat({
                    k: v for k, v in last_prices.items() if k in coin_list
                })
                LOGGER.debug('last_prices:\n%s\n', formatted_last_prices, extra=self.extra)
            
            # Create usd_price
            agg_balance_df['usd_price'] =  agg_balance_df['coin_name'].apply(lambda c: last_prices[c][1])

            # Calculate balance_usd_amount & available_balance_usd_amount
            agg_balance_df['balance_usd_amount'] = np.abs(agg_balance_df['balance']) * agg_balance_df['usd_price']
            agg_balance_df['available_balance_usd_amount'] = np.abs(agg_balance_df['available_balance']) * agg_balance_df['usd_price']

        if debug:
            LOGGER.debug('Futures agg_balance_df:\n%s\n', agg_balance_df.to_string(), extra=self.extra)
        
        return agg_balance_df

    def get_futures_income_history(
        self,
        new_income_n: int = 15,
        debug: bool = False
    ) -> pd.DataFrame:
        # Define column_types & rename_columns
        column_types = {
            'time': float,
            'asset': str,
            'income': float,
            'incomeType': str
        }

        rename_columns = {
            'time': 'period',
            'asset': 'coin_name',
            'income': 'income_quantity',
            'incomeType': 'income_type'
        }

        # Extract income history
        income_history = (
            pd.DataFrame(self.client.futures_income_history(limit=new_income_n))
            .filter(items=column_types.keys())
            .astype(column_types)
            .rename(columns=rename_columns)
            .set_index('period')
        )

        # Turn time column into a datetime column
        income_history.index = pd.to_datetime(income_history.index.to_series(), unit='ms')

        if debug:
            LOGGER.debug('Raw income_history:\n%s\n', income_history.to_string(), extra=self.extra)

        # Define agg functions
        agg_funs = {
            'coin_name': 'first',
            'income_quantity': 'sum',
            'income_type': 'first'
        }

        # Define empty agg_income_history
        agg_income_history: pd.DataFrame = None

        for coin_name in income_history['coin_name'].unique():
            for income_type in income_history['income_type'].unique():
                # Filter income_history
                filtered_income_history: pd.DataFrame = income_history.loc[
                    (income_history['coin_name'] == coin_name) &
                    (income_history['income_type'] == income_type)
                ]

                # Group income_history by 30min intervals
                filtered_income_history = (
                    filtered_income_history
                    .groupby(pd.Grouper(freq='30min'))
                    .agg(agg_funs)
                    .reset_index(drop=False)
                )

                # Concatenate filtered_income_history
                agg_income_history: pd.DataFrame = pd.concat(
                    [agg_income_history, filtered_income_history], axis=0, ignore_index=False
                )

        # Add account_id column
        agg_income_history['account_id'] = self.account_id
        
        # Dropna
        agg_income_history.dropna(
            subset=['coin_name', 'income_type'],
            how='all',
            inplace=True
        )

        if debug:
            LOGGER.debug('Grouped agg_income_history:\n%s\n', agg_income_history.to_string(), extra=self.extra)

        return agg_income_history

    def get_futures_orders_history(
        self,
        new_orders_n: int = 15,
        debug: bool = False
    ) -> pd.DataFrame:
        # Define column_types & rename_columns
        column_types = {
            'time': float,
            'symbol': str,
            'status': str,
            'side': str, # BUY, SELL, etc
            'type': str, # LIMIT, MARKET, etc
            'avgPrice': float,
            'executedQty': float, # quantity of the asset traded
            'stopPrice': float
        }

        rename_columns = {
            'time': 'period',
            'status': 'order_status',
            'side': 'order_side',
            'type': 'order_type',            
            'avgPrice': 'order_avg_price',
            'executedQty': 'order_quantity',
            'stopPrice': 'order_stop_price'
        }

        # Create empty orders_history
        agg_orders_history: pd.DataFrame = None

        # Define traded_symbols
        traded_symbols = []
        for stable_coin, coin_names in self.available_pairs.items():
            traded_symbols.extend([
                coin_name + stable_coin for coin_name in coin_names
            ])
        
        for traded_symbol in traded_symbols:
            # Extract orders history
            new_orders_history = (
                pd.DataFrame(self.client.futures_get_all_orders(
                    symbol=traded_symbol,
                    limit=new_orders_n
                ))
            )

            if new_orders_history.shape[0] > 0:
                # Prepare dataframe
                new_orders_history = (
                    new_orders_history
                    .filter(items=column_types.keys())
                    .astype(column_types)
                    .rename(columns=rename_columns)
                    .set_index('period')
                )

                # Turn index into pd.DatetimeIndex
                new_orders_history.index = pd.to_datetime(new_orders_history.index.to_series(), unit='ms')

                # Add weighted avg price & weighted avg stop price
                new_orders_history['weighted_order_avg_price'] = (
                    new_orders_history['order_avg_price'] 
                    * new_orders_history['order_quantity']
                )
                new_orders_history['weighted_order_stop_price'] = (
                    new_orders_history['order_stop_price'] 
                    * new_orders_history['order_quantity']
                )

                if debug:
                    LOGGER.debug('Raw new_orders_history:\n%s\n', new_orders_history.to_string(), extra=self.extra)

                # Filter by filled orders
                new_orders_history = new_orders_history.loc[new_orders_history['order_status'] == 'FILLED']

                # Drop order_status column
                new_orders_history.drop(columns=['order_status'], inplace=True)

                # Define group_orders func
                def group_orders(gb_df: pd.DataFrame):
                    if gb_df.shape[0] > 0:
                        return pd.DataFrame({
                            'order_type': gb_df['order_type'].values[0],
                            'order_side': gb_df['order_side'].values[0],
                            'order_avg_price': gb_df['weighted_order_avg_price'].sum() / gb_df['order_quantity'].sum(),
                            'order_quantity': gb_df['order_quantity'].sum(),
                            'order_stop_price': gb_df['weighted_order_stop_price'].sum() / gb_df['order_quantity'].sum(),
                        }, index=[0])
                    else:
                        return pd.DataFrame(columns=[
                            'order_type', 'order_side', 'order_avg_price', 'order_quantity', 'order_stop_price'
                        ])                    

                # Group new_orders_history in 30min intervals
                for order_type in new_orders_history['order_type'].unique():
                    for order_side in new_orders_history['order_side'].unique():
                        # Filter new_orders_history, based on order_type & order_side
                        filtered_new_orders_history: pd.DataFrame = new_orders_history.loc[
                            (new_orders_history['order_type'] == order_type) &
                            (new_orders_history['order_side'] == order_side)
                        ]

                        if filtered_new_orders_history.shape[0] > 0:
                            # Group filtered_new_orders_history by 30min intervals
                            new_agg_orders_history = (
                                filtered_new_orders_history
                                .groupby(pd.Grouper(freq='30min'))
                                .apply(group_orders)
                                .reset_index(drop=False)
                            )

                            # Add coin_name, stable_coin & accoint_id columns
                            new_agg_orders_history['coin_name'] = traded_symbol[:-4]
                            new_agg_orders_history['stable_coin'] = traded_symbol[-4:]
                            new_agg_orders_history['account_id'] = self.account_id

                            if debug:
                                print(f'new_agg_orders_history:\n{new_agg_orders_history}\n')

                            # Concatenate new_agg_orders_history
                            agg_orders_history: pd.DataFrame = pd.concat(
                                [agg_orders_history, new_agg_orders_history], axis=0, ignore_index=True
                            )

        # Order columns
        ordered_cols = [
            'period', 'coin_name', 'stable_coin', 
            'order_side', 'order_type', 'account_id', 
            'order_avg_price', 'order_quantity', 'order_stop_price'
        ]
        agg_orders_history = agg_orders_history[ordered_cols]

        if debug:
            LOGGER.debug('Final agg_orders_history:\n%s\n', agg_orders_history.to_string(), extra=self.extra)

        return agg_orders_history

    def get_futures_positions(
        self,
        intervals: str = Params.general_params.get("intervals"),
        include_usd_valuation: bool = False,        
        debug: bool = False
    ) -> pd.DataFrame:
        # Define column_types & rename_columns
        column_types = {
            'symbol': str,
            'positionAmt': float, # can be positive or negative
            'entryPrice': float,
            'unRealizedProfit': float,
            'leverage': int
        }
        rename_columns = {
            'positionAmt': 'position_quantity',
            'entryPrice': 'entry_price',
            'unRealizedProfit': 'unrealized_pnl',
            'leverage': 'position_leverage'
        }

        # Extract Positions
        positions_df = (
            pd.DataFrame(self.client.futures_position_information())
            .filter(items=column_types.keys())
            .astype(column_types)
            .rename(columns=rename_columns)
            .set_index('symbol')
        )

        # Filter positions, based on position_quantity
        positions_df = positions_df.loc[np.abs(positions_df['position_quantity']) > 0]

        # Interpret position kind
        positions_df['position_kind'] = np.where(
            positions_df['position_quantity'] >= 0, 'long', 'short'
        )

        if include_usd_valuation:
            # Extract traded coins
            coin_list = [symbol[:-4] for symbol in positions_df.index.tolist()]

            # Update last_prices & extract output
            self.update_last_prices(
                coin_list=coin_list,
                intervals=intervals
            )

            last_prices = self.last_prices

            if debug:
                formatted_last_prices = pformat({
                    k: v for k, v in last_prices.items() if k in coin_list
                })
                LOGGER.debug('last_prices:\n%s\n', formatted_last_prices, extra=self.extra)
            
            # Create usd_price & usd_amount
            positions_df['usd_price'] = (
                positions_df.index
                .to_series()
                .apply(lambda symbol: last_prices[symbol[:-4]][1])
            )

            positions_df['position_usd_amount'] = np.abs(positions_df['position_quantity']) * positions_df['usd_price']

        if debug:
            LOGGER.debug('Futures positions_df:\n%s\n', positions_df.to_string(), extra=self.extra)
        
        return positions_df

    """
    Trading Methods
    """

    def place_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: float, 
        leverage: int,
        sl_price: float = None, 
        tp_price: float = None, 
        exit_side: str = None,
        order_type: str = 'LIMIT'
    ) -> dict:
        """
        Send in a new order: https://binance-docs.github.io/apidocs/futures/en/#new-order-trade
        """
        # Validate order_type
        available_types = [
            'LIMIT',
            'MARKET',
            'STOP',
            'TAKE_PROFIT',
            'STOP_MARKET',
            'TAKE_PROFIT_MARKET',
            'TRAILING_STOP_MARKET'
        ]
        assert order_type in available_types

        # Define timeInForce & re-define price (if necessary)
        if order_type == 'MARKET':
            price = None
            timeInForce = None
        else:
            timeInForce = 'GTC'

        LOGGER.info(
            'Placing order:\n'
            '   - symbol: %s\n'
            '   - side: %s\n'
            '   - order_type: %s\n'
            '   - quantity: %s\n'
            '   - price: %s\n'
            '   - timeInForce: %s\n', 
            symbol, 
            side, 
            order_type, 
            quantity, 
            price, 
            timeInForce, 
            extra=self.extra
        )
        
        # Send new order
        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,  # BUY/SELL same as LONG, SHORT?
            type=order_type,
            quantity=quantity,
            price=price,
            timeInForce=timeInForce
        )

        LOGGER.info('Order %s was successfully placed.', order['orderId'], extra=self.extra)

        # Change leverage
        LOGGER.info('Changing leverage to: %s', leverage, extra=self.extra)
        self.client.futures_change_leverage(
            symbol=symbol,
            leverage=leverage
        )

        # Send SL & TP orders
        sl_order, tp_order = self.place_sl_tp_order(
            symbol=symbol,
            quantity=quantity,
            sl_price=sl_price,
            tp_price=tp_price,
            exit_side=exit_side
        )

        return order, sl_order, tp_order

    def place_sl_tp_order(
        self, 
        symbol: str, 
        quantity: float,
        sl_price: float = None,
        tp_price: float = None, 
        exit_side: str = None
    ) -> None:
        # Define default orders
        sl_order, tp_order = None, None

        if not(tp_price is None or math.isnan(tp_price)):
            # Define tp_stopPrice
            coin_name = symbol[:-4]
            price_diff = float('0.' + ''.join(['0']*(self.price_precision[coin_name]-1)) + '1')
            # 10 ** -self.price_precision[coin_name]

            if exit_side == 'BUY':
                # If shorting: -> TP exit side is BUY -> price is decreasing -> tp_stopPrice > tp_price
                tp_stopPrice = tp_price + price_diff
            else:
                # If longing: -> TP exit side is SELL -> price is increasing -> tp_stopPrice < tp_price
                tp_stopPrice = tp_price - price_diff

            try:
                LOGGER.info(
                    'Placing TAKE_PROFIT order.\n'
                    '   - symbol: %s\n'
                    '   - exit_side: %s\n'
                    '   - quantity: %s\n'
                    '   - tp_stopPrice: %s\n'
                    '   - price_diff: %s\n'
                    '   - tp_price: %s\n', 
                    symbol, 
                    exit_side, 
                    quantity, 
                    tp_stopPrice, 
                    price_diff,
                    tp_price,
                    extra=self.extra
                )
                
                # Send TP order
                tp_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=exit_side,
                    type='TAKE_PROFIT',  # 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'
                    quantity=quantity,
                    price=tp_price,
                    stopPrice=tp_stopPrice,
                    # newClientOrderId=order['orderId'],
                    # selfTradePreventionMode='EXPIRE_TAKER',
                    timeInForce='GTC'
                )

                LOGGER.info('TAKE_PROFIT order (%s) successfully placed.\n', tp_order['orderId'], extra=self.extra)
            except Exception as e:
                LOGGER.error(
                    'Unable to set TAKE_PROFIT order.\n'
                    'Exception: %s\n',
                    e, extra=self.extra
                )

        if not(sl_price is None or math.isnan(sl_price)):
            # Activation price and actual price should be the same, to maximize the chances of triggering
            sl_stopPrice = sl_price

            try:
                LOGGER.info(
                    'Placing STOP order.\n'
                    '   - symbol: %s\n'
                    '   - exit_side: %s\n'
                    '   - quantity: %s\n'
                    '   - sl_stopPrice: %s\n'
                    '   - sl_price: %s\n', 
                    symbol, 
                    exit_side, 
                    quantity, 
                    sl_stopPrice, 
                    sl_price, 
                    extra=self.extra
                )
                
                # Send SL order
                sl_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=exit_side,
                    type='STOP',  # 'STOP_MARKET'
                    quantity=quantity,
                    price=sl_price,
                    stopPrice=sl_stopPrice,
                    # newClientOrderId=order['orderId'],
                    # selfTradePreventionMode='EXPIRE_TAKER',
                    timeInForce='GTC'
                )

                LOGGER.info('Stop loss order (%s) successfully placed.\n', sl_order['orderId'], extra=self.extra)
            except Exception as e:
                LOGGER.error(
                    'Unable to set STOP order.\n'
                    'Exception: %s\n', 
                    e, extra=self.extra
                )

        return sl_order, tp_order

    def find_position_price(
        self,
        coin_name: str,
        stable_coin: str,
        kind: str,
        open_: bool = True,
        limit_price: float = None,
        debug: bool = False
    ) -> float:
        # Define order_book_n
        if stable_coin == 'USDT':
            order_book_n = 1
        else:
            order_book_n = 0

        # Get the Order Book for the market
        order_book = self.client.futures_order_book(
            symbol=coin_name + stable_coin,
            limit=10
        )

        # Populate price precision
        first_ask: str = order_book['asks'][0][0]
        price_precision = first_ask[::-1].find('.')
        self.price_precision[coin_name] = price_precision

        # prices = pd.Series(
        #     [float(a[0]) for a in order_book['asks']] + [float(b[0]) for b in order_book['bids']]
        # )
        # self.diff_dict[coin_name + stable_coin] = prices.diff().abs().mean()

        if debug:
            LOGGER.debug(
                # 'diff: %s\n'
                'price_precision: %s\n'
                'order_book:\n%s\n',
                # self.diff_dict[coin_name + stable_coin], 
                price_precision,
                pformat(order_book), 
                extra=self.extra
            )

        if open_:
            # Find Open Price
            if kind == 'long':
                # long_open_price
                bid_price = float(order_book["bids"][order_book_n][0])
                if limit_price is not None:
                    return np.min([limit_price, bid_price])
                else:
                    return bid_price
            else:
                # short_open_price
                ask_price = float(order_book["asks"][order_book_n][0])
                if limit_price is not None:
                    return np.max([limit_price, ask_price])
                else:
                    return ask_price
        else:
            # Find Close Price
            if kind == 'long':
                # long_closing_price
                ask_price = float(order_book["asks"][order_book_n][0])
                if limit_price is not None:
                    return np.max([limit_price, ask_price])
                else:
                    return ask_price
            elif kind == 'short':
                # short_closing_price
                bid_price = float(order_book["bids"][order_book_n][0])
                if limit_price is not None:
                    return np.min([limit_price, bid_price])
                else:
                    return bid_price

    def open_position(
        self, 
        coin_name: str,
        kind: str, 
        stable_coin: str = None,
        limit_price: float = None, 
        usd_quantity: float = None, 
        leverage: int = 1,
        sl_price: float = None, 
        tp_price: float = None,
        debug: bool = False
    ) -> None:
        # Validate kind
        if kind not in ['long', 'short']:
            LOGGER.critical('Invalid "kind" parameter (%s).', kind, extra=self.extra)
            raise Exception(f'Invalid "kind" parameter ({kind}).\n\n')
        
        # Validate stable_coin
        if stable_coin is None:
            stable_coin = self.extract_stable_coin(
                coin_name=coin_name,
                forced_stable_coin=None
            )

        # Find Open Price
        open_price = self.find_position_price(
            coin_name=coin_name,
            stable_coin=stable_coin,
            kind=kind,
            open_=True,
            limit_price=limit_price,
            debug=debug
        )

        # Define side & exit_side
        if kind == 'long':
            side = 'BUY'
            exit_side = 'SELL'
        else:
            side = 'SELL'
            exit_side = 'BUY'

        # Calculate Crypto Quantity
        quantity = usd_quantity / open_price
        quantity = math.floor((quantity * (self.quantity_precision[coin_name][0]))*0.97) / (self.quantity_precision[coin_name][0])
        
        # Re-define sl_price & tp_price with the necessary precision
        if sl_price is not None:
            sl_price = round(sl_price, self.price_precision[coin_name])
        if tp_price is not None:
            tp_price = round(tp_price, self.price_precision[coin_name])

        LOGGER.info(
            'Opening %s %s position:\n'
            '   - open_price: %s\n'
            '   - sl_price: %s\n'
            '   - tp_price: %s\n',
            coin_name,
            kind,
            open_price,
            sl_price,
            tp_price,
            extra=self.extra
        )
        
        # Register initial time
        t0 = time.time()

        # Define symbol
        symbol = coin_name + stable_coin

        # Place order
        order, sl_order, tp_order = self.place_order(
            symbol=symbol,
            side=side,
            exit_side=exit_side,
            quantity=quantity,
            price=open_price,
            leverage=leverage,
            sl_price=sl_price,
            tp_price=tp_price,
            order_type='LIMIT'
        )

        if debug:
            LOGGER.debug("Waiting until %s's order is filled.", symbol, extra=self.extra)

        # Define initial fill_delay
        fill_delay = time.time() - t0

        # Wait 10 sec to give some time for the order to get filled
        time.sleep(10)

        if stable_coin == 'USDT':
            max_wait = 60 * 4
        else:
            max_wait = 60 * 7

        while fill_delay < max_wait:
            # Get information from sent order
            order_info = self.client.futures_get_order(
                symbol=symbol, 
                orderId=order['orderId']
            )
            
            if order_info['status'] == 'FILLED':
                LOGGER.info('Order was filled in %s segs.', time.time() - t0, extra=self.extra)
                return
            
            # Find new opening price
            updated_open_price = self.find_position_price(
                coin_name=coin_name,
                stable_coin=stable_coin,
                kind=kind,
                open_=True,
                limit_price=limit_price,
                debug=debug
            )

            LOGGER.info('updated_open_price: %s', updated_open_price, extra=self.extra)

            if updated_open_price != open_price:
                # Cancel orders
                self.cancel_orders(debug=False)

                # Check positions
                order_info = self.client.futures_get_order(
                    symbol=symbol, 
                    orderId=order['orderId']
                )

                if debug:
                    LOGGER.debug('Old order status: %s', order_info["status"], extra=self.extra)

                # Place new LIMIT order
                order, sl_order, tp_order = self.place_order(
                    symbol=symbol,
                    side=side,
                    exit_side=exit_side,
                    quantity=quantity,
                    price=updated_open_price,
                    leverage=leverage,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    order_type='LIMIT'
                )

                if debug:
                    LOGGER.debug('New order: %s', order["orderId"], extra=self.extra)

            # Sleep 10 sec
            time.sleep(10)

            # Register new fill_delay
            fill_delay = time.time() - t0
            if debug:
                LOGGER.debug('current delay: %s.', fill_delay, extra=self.extra)

        if self.client.futures_get_order(
            symbol=symbol, 
            orderId=order['orderId']
        )['status'] == 'FILLED':
            LOGGER.info('Order was filled in %s segs.', fill_delay, extra=self.extra)
            return
        
        LOGGER.warning(
            'Code was unable to open the position in time.\n'
            'Position will be opened as a MARKET order.\n', 
            extra=self.extra
        )

        # Cancel open orders
        self.cancel_orders(debug=debug)

        # Open MARKET order
        order, sl_order, tp_order = self.place_order(
            symbol=symbol,
            side=side,
            exit_side=exit_side,
            quantity=quantity,
            price=None,
            leverage=leverage,
            sl_price=sl_price,
            tp_price=tp_price,
            order_type='MARKET'
        )

    def close_positions(
        self, 
        positions_to_close: dict,
        debug: bool = False
    ) -> None:
        if debug:
            LOGGER.debug('positions_to_close:\n%s\n', pformat(positions_to_close), extra=self.extra)

        if positions_to_close is None:
            LOGGER.warning('Unable to close position: No open positions were found.', extra=self.extra)
            return

        for symbol in list(positions_to_close.keys()):
            # Find position params
            coin_name = symbol[:-4]
            stable_coin = symbol[-4:]
            kind = positions_to_close[symbol]['position_kind']
            quantity = np.abs(positions_to_close[symbol]['position_quantity'])
            leverage = int(positions_to_close[symbol]['position_leverage'])

            # Find Closing Price
            closing_price = self.find_position_price(
                coin_name=coin_name,
                stable_coin=stable_coin,
                kind=kind,
                open_=False,
                limit_price=None,
                debug=debug
            )

            # Define side & exit_side
            if positions_to_close[symbol]['position_kind'] == 'long':
                side = 'SELL'
                exit_side = None # 'BUY'
            else:
                side = 'BUY'
                exit_side = None # 'SELL'
            
            if debug:
                LOGGER.debug(
                    'Closing position:\n'
                    '   - symbol: %s\n'
                    '   - kind: %s\n'
                    '   - quantity: %s\n'
                    '   - leverage: %s\n'
                    '   - closing_price: %s\n'
                    '   - side: %s\n', 
                    symbol, 
                    kind, 
                    quantity, 
                    leverage, 
                    closing_price, 
                    side, 
                    extra=self.extra
                )

            # Register initial time
            t0 = time.time()

            # Send new limit order
            order, sl_order, tp_order = self.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=closing_price,
                leverage=leverage, # 1
                order_type='LIMIT'
            )

            if debug:
                LOGGER.debug("Waiting until %s's position in closed.", symbol, extra=self.extra)

            # Define initial fill_delay
            fill_delay = time.time() - t0

            # Wait 10 sec to give some time for the order to get filled
            time.sleep(10)

            while fill_delay < 60 * 5:
                # Get information from sent order
                order_info = self.client.futures_get_order(
                    symbol=symbol, 
                    orderId=order['orderId']
                )
                
                if order_info['status'] == 'FILLED':
                    LOGGER.info('Order was filled in %s segs.', time.time() - t0, extra=self.extra)

                    # Cancel orders
                    self.cancel_orders(debug=debug)
                    return

                # Find new closing price
                updated_closing_price = self.find_position_price(
                    coin_name=coin_name,
                    stable_coin=stable_coin,
                    kind=kind,
                    open_=False,
                    limit_price=None,
                    debug=debug
                )

                LOGGER.info('updated_closing_price: %s', updated_closing_price, extra=self.extra)

                if updated_closing_price != closing_price:
                    # Cancel orders
                    self.cancel_orders(debug=False)
                    
                    # Check positions
                    order_info = self.client.futures_get_order(
                        symbol=symbol, 
                        orderId=order['orderId']
                    )

                    if debug:
                        LOGGER.debug('Old order status: %s', order_info["status"], extra=self.extra)

                    # Place new LIMIT order
                    order, sl_order, tp_order = self.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=updated_closing_price,
                        leverage=leverage, # 1,
                        order_type='LIMIT'
                    )

                    if debug:
                        LOGGER.debug('New order: %s', order["orderId"], extra=self.extra)

                # Sleep 10 sec
                time.sleep(10)

                # Register new fill_delay
                fill_delay = time.time() - t0
                if debug:
                    LOGGER.debug('current delay: %s.', fill_delay, extra=self.extra)
                
            if self.client.futures_get_order(
                symbol=symbol, 
                orderId=order['orderId']
            )['status'] == 'FILLED':
                LOGGER.info('Order was filled in %s segs.', fill_delay, extra=self.extra)
                return
            
            LOGGER.warning(
                'Code was unable to close the %s position in time.\n'
                'Position will be closed as a MARKET order.\n', 
                symbol, extra=self.extra
            )

            # Cancel open orders
            self.cancel_orders(debug=debug)

            # Close position with MARKET order
            order, sl_order, tp_order = self.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=None,
                leverage=leverage,
                order_type='MARKET'
            )

    def modify_position(
        self,
        adj_symbol: str,
        exposure_adj_USD: float,
        intervals: str,
        current_positions: dict = None,
        adj_leverage: int = 1,
        tp_price: float = None,
        sl_price: float = None,
        debug: bool = False
    ) -> None:
        # Retrieve Current Positions
        if current_positions is None:
            # Extract current positions df
            current_positions_df = self.get_futures_positions(
                intervals=intervals,
                include_usd_valuation=True,
                debug=debug
            )

            # Transform to dict
            current_positions = (
                current_positions_df
                .sort_values(by='position_usd_amount', ascending=False)
                .T
                .to_dict()
            )
            
        if debug:
            LOGGER.debug('current_positions:\n%s\n', pformat(current_positions), extra=self.extra)

        if current_positions is None:
            LOGGER.warning('Unable to modify position: No open positions were found.', extra=self.extra)
            return
        
        elif adj_symbol not in current_positions.keys():
            LOGGER.error(
                'Unable to modify position:\n'
                '    - %s was not found in positions.\n'
                'current_positions:\n%s\n',
                adj_symbol, 
                pformat(current_positions), 
                extra=self.extra
            )
            return
        
        for symbol in list(current_positions.keys()):
            # Find position params
            if symbol == adj_symbol:
                # Find position params
                coin_name = symbol[:-4]
                stable_coin = symbol[-4:]
                kind = current_positions[symbol]['position_kind']
                # leverage = current_positions[symbol]['position_leverage']

                # Interpret modify kind
                if exposure_adj_USD >= 0:
                    modify_kind = 'long'
                else:
                    modify_kind = 'short'

                # Log modifying information
                LOGGER.info(
                    'Modfying position:\n'
                    '   - symbol: %s\n'
                    '   - exposure_adj_USD: %s\n'
                    '   - modify_kind: %s\n'
                    '   - adj_leverage: %s\n', 
                    symbol, 
                    exposure_adj_USD,
                    modify_kind,
                    adj_leverage, 
                    extra=self.extra
                )

                # Open New Position
                if np.abs(exposure_adj_USD) > 5:
                    self.open_position(
                        coin_name=coin_name,
                        kind=modify_kind,
                        stable_coin=stable_coin,
                        limit_price=None,
                        usd_quantity=np.abs(exposure_adj_USD),
                        leverage=adj_leverage,
                        sl_price=None, # SL will be updated afterwards
                        tp_price=None, # TP will be updated afterwards
                        debug=debug
                    )
                else:
                    LOGGER.warning(
                        'np.abs(exposure_adj_USD) (%s) is < 5, therefore current position will not be modified.', 
                        np.abs(exposure_adj_USD), extra=self.extra
                    )

                if (
                    sl_price is not None 
                    or tp_price is not None
                ):
                    # Find current_positions_df
                    current_positions_df = self.get_futures_positions(
                        intervals=intervals,
                        include_usd_valuation=True,
                        debug=debug
                    )

                    # Transform to dict
                    current_positions = (
                        current_positions_df
                        .sort_values(by='position_usd_amount', ascending=False)
                        .T
                        .to_dict()
                    )

                    if debug:
                        LOGGER.debug('new current_positions:\n%s\n', pformat(current_positions), extra=self.extra)

                    # Cancel open SL & TP orders
                    self.cancel_orders(debug=debug)

                    # Find open_price (required to update price precision)
                    open_price = self.find_position_price(
                        coin_name=coin_name,
                        stable_coin=stable_coin,
                        kind=kind,
                        open_=True,
                        limit_price=None,
                        debug=debug
                    )

                    # Define required exit_side
                    if kind == 'long':
                        exit_side = 'SELL'
                    else:
                        exit_side = 'BUY'

                    # Re-define sl_price & tp_price with the necessary precision
                    if sl_price is not None:
                        sl_price = round(sl_price, self.price_precision[coin_name])
                    if tp_price is not None:
                        tp_price = round(tp_price, self.price_precision[coin_name])

                    LOGGER.info(
                        'Modifying SL & TP orders.\n'
                        '   - sl_price: %s\n'
                        '   - tp_price: %s\n'
                        '   - exit_side: %s\n',
                        sl_price,
                        tp_price,
                        exit_side,
                        extra=self.extra
                    )
                    
                    # Place SL & TP order
                    self.place_sl_tp_order(
                        symbol=symbol,
                        quantity=np.abs(current_positions[symbol]['position_quantity']),
                        sl_price=sl_price,
                        tp_price=tp_price,
                        exit_side=exit_side
                    )

    def cancel_orders(
        self,
        debug: bool = False
    ) -> None:
        LOGGER.info('Cancelling all NEW orders.', extra=self.extra)

        # Get all futures account orders; active, canceled, or filled.
        orders = self.client.futures_get_all_orders(limit=10) # futures_get_open_orders

        if len(orders) > 0:
            orders_df = pd.DataFrame(orders)[['symbol', 'status', 'price', 'type', 'side', 'time']]
            orders_df = orders_df.loc[orders_df['status'] == 'NEW']
            
            if debug:
                LOGGER.debug('NEW orders_df:\n%s\n', orders_df.to_string(), extra=self.extra)

            for symbol in orders_df['symbol'].unique().tolist():
                LOGGER.info('Closing %s open orders.', symbol, extra=self.extra)

                # Cancel the symbol that is being traded
                response = self.client.futures_cancel_all_open_orders(symbol=symbol)

                assert response['msg'] == 'The operation of cancel all open order is done.'
        
        # # Check there's no new orders
        # orders = self.client.futures_get_all_orders(limit=10)
        # orders_df = pd.DataFrame(orders)[['symbol', 'status', 'price', 'type', 'side', 'time']]
        # new_orders = orders_df.loc[orders_df['status'] == 'NEW']

        # if len(new_orders) > 0:
        #     print(f'[WARNING] The cancel_orders method was called, but there are still open orders.\n'
        #           f'Open orders:\n {new_orders}\n\n')

    def other_functions(self):
        # Execute transfer between spot account and futures account
        # https://binance-docs.github.io/apidocs/futures/en/#new-future-account-transfer
        # client.futures_account_transfer()

        pass
