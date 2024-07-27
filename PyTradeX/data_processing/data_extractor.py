from PyTradeX.config.params import Params
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.utils.data_processing.repair_helper import (
    repair_missing_new_data,
    add_missing_rows,
    drop_unexpected_columns,
    drop_duplicates
)
from PyTradeX.utils.data_processing.collective_data import get_collective_data
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair
)
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date
from math import isnan
from matplotlib import pyplot as plt
import missingno as msno
import requests
import holidays
import time
import os
from typing import List, Dict, Any
from pprint import pformat
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


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


class DataExtractor(DataCleaner):

    needed_cols: List[str] = Params.fixed_params.get("raw_data_columns").copy()

    load_parquet = [
        'raw_data'
    ]
    load_pickle = [
        'unused_data',
        'null_last_obs_cols',
        'last_backup'
    ]

    def __init__(
        self, 
        coin_name: str, 
        intervals: str = Params.general_params.get("intervals"),
        client: BinanceClient = None,
        overwrite: bool = True,
        mock: bool = False,
        **data_params
    ) -> None:
        # General params
        self.coin_name: str = coin_name
        self.intervals: str = intervals
        self.client: BinanceClient = client

        # Load param
        self.overwrite: bool = overwrite
        self.mock: bool = mock
        
        # Data params
        self.features: bool = None
        self.periods: int = None
        self.other_coins_n: int = None

        self.new_n: int = None
        self.mbp: int = None
        self.yfinance_params: dict = None

        default_params = [
            'features',
            'periods',
            'other_coins_n',
            'new_n',
            'mbp',
            'yfinance_params'
        ]
        for param in default_params:
            if param not in data_params.keys():
                data_params[param] = getattr(Params, 'data_params')[param]

            setattr(self, param, data_params[param])

        # Update periods
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < self.periods:
            # LOGGER.warning(
            #     '%s DE has less periods than required.\n'
            #     'Expected periods: %s\n'
            #     'raw_data_periods: %s\n'
            #     'Thus, self.periods will be reverted to %s.\n',
            #     self.coin_name, self.periods, raw_data_periods, raw_data_periods
            # )
            self.periods: int = raw_data_periods
        
        self.other_coins: list = Params.other_coins_json[self.intervals][:self.other_coins_n]
        self.lc_ids: dict = Params.lc_ids.copy()
        self.i: int = 0

        # Load params
        self.raw_data: pd.DataFrame = None
        self.unused_data: pd.DataFrame = None
        self.null_last_obs_cols: list = None

        self.last_backup = None

        self.load(debug=False)
    
    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/data_processing/data_extractor/{self.intervals}/{self.coin_name}"
        else:
            return f"{Params.bucket}/data_processing/data_extractor/{self.intervals}/{self.coin_name}"
        
    @property
    def backup_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/backup/raw_data/{self.intervals}/{self.coin_name}"
        else:
            return f"{Params.bucket}/backup/raw_data/{self.intervals}/{self.coin_name}"

    def get_new_data(
        self,
        periods: int
    ) -> pd.DataFrame:
        if periods is None:
            periods = self.periods

        # Extract new_data
        data = self.client.get_data(
            coin_name=self.coin_name,
            intervals=self.intervals,
            periods=periods,
            forced_stable_coin=None,
            update_last_prices=True,
            ignore_last_period=True
        )

        sleep_time = 0

        # Re-rxtract new_data if it has not been updated
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        while data.index[-1] < utc_now - pd.Timedelta(minutes=self.mbp * 2) and sleep_time < 4:
            LOGGER.warning(
                'Data from %s (%s) has not been updated quickly enough.\n'
                'data.index[-1]: %s (utc_now: %s).\n',
                self.coin_name, 
                self.intervals, 
                data.index[-1], 
                utc_now
            )
            time.sleep(1)
            sleep_time += 1
            data = self.client.get_data(
                coin_name=self.coin_name,
                intervals=self.intervals,
                periods=periods,
                forced_stable_coin=None,
                update_last_prices=True,
                ignore_last_period=False
            )

        return data.rename(columns={
            'open': 'coin_open',
            'high': 'coin_high',
            'low': 'coin_low',
            'price': 'coin_price',
            'volume': 'ta_volume'
        })

    def get_long_short_data(
        self, 
        periods: int, 
        accelerated: bool = False,
        category_features: Dict[str, List[str]] = None,
        expected_idx: pd.DatetimeIndex = None,
        save_mock: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        # Find wether or not to find info
        find_data = True

        if accelerated:
            # Find long_short columns
            long_short_cols = category_features['long_short']

            if len(long_short_cols) == 0:
                find_data = False

        if find_data:
            # Find global long_short data
            global_long_short_data = self.client.get_long_short_data(
                coin_name=self.coin_name,
                from_='global',
                intervals=self.intervals,
                periods=periods
            )

            if save_mock:
                self.save_mock_asset(
                    asset=global_long_short_data,
                    asset_name='global_long_short_data'
                )

            # Find top traders long_short data
            top_traders_long_short_data = self.client.get_long_short_data(
                coin_name=self.coin_name,
                from_='top_traders',
                intervals=self.intervals,
                periods=periods
            )

            if save_mock:
                self.save_mock_asset(
                    asset=top_traders_long_short_data,
                    asset_name='top_traders_long_short_data'
                )

            long_short_data = pd.concat([global_long_short_data, top_traders_long_short_data], axis=1)
            long_short_data['top_global_ratio'] = long_short_data.apply(
                lambda row: row['top_traders_long_short_ratio'] / row['global_long_short_ratio'], axis=1
            )
            long_short_data.columns = 'long_short_' + long_short_data.columns

            if debug:
                LOGGER.debug(
                    'long_short_data.shape: %s\n'
                    'long_short_data.tail(10): \n%s\n',
                    long_short_data.shape, long_short_data.tail(10).to_string()
                )

            return long_short_data.tail(periods)
        else:
            expected_columns = [
                'long_short_global_long_perc',
                'long_short_global_long_short_ratio',
                'long_short_top_traders_long_perc',
                'long_short_top_traders_long_short_ratio',
                'long_short_top_global_ratio'
            ]

            return pd.DataFrame(
                np.zeros(shape=(len(expected_idx), len(expected_columns))),
                columns=expected_columns,
                index=expected_idx
            )

    def get_lc_data(
        self,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        V1:
            time: 1705629600, // Fri, 19 Jan 2024 02:00:00 GMT A unix timestamp (in seconds)
            open: 2,461.295, //Open price for the time period
            close: 2,451.049, //Close price for the time period
            high: 2,461.483, //Higest price fo rthe time period
            low: 2,449.781, //Lowest price for the time period
            volume_24h: 11,965,316,353.52, //Volume in USD for 24 hours up to this data point
            market_cap: 294,559,837,147.54,
            circulating_supply: 120,177,064.749,
            galaxy_score: 40, //A proprietary score based on technical indicators of price, average social sentiment, relative social activity, and a factor of how closely social indicators correlate with price and volume
            alt_rank: 22, //A proprietary score based on how an asset is performing relative to all other assets supported
            market_dominance: 18.111, //The percent of the total market cap that this asset represents

        V2:
            time: 1705622400, // Fri, 19 Jan 2024 00:00:00 GMT A unix timestamp (in seconds)
            open: 2,468.689, //Open price for the time period
            close: 2,465.628, //Close price for the time period
            high: 2,471.813, //Higest price fo rthe time period
            low: 2,460.902, //Lowest price for the time period
            volume_24h: 11,957,544,546.29, //Volume in USD for 24 hours up to this data point
            market_cap: 296,314,842,594.02,
            circulating_supply: 120,178,235.37,
            galaxy_score: 88, //A proprietary score based on technical indicators of price, average social sentiment, relative social activity, and a factor of how closely social indicators correlate with price and volume
            volatility: 0.01362,
            alt_rank: 31, //A proprietary score based on how an asset is performing relative to all other assets supported
            contributors_active: 7,010, //number of unique social accounts with posts that have interactions
            contributors_created: 676, //number of unique social accounts that created new posts
            posts_active: 11,997, //number of unique social posts with interactions
            posts_created: 796, //number of unique social posts created
            interactions: 967,680, //number of all publicly measurable interactions on a social post (views, likes, comments, thumbs up, upvote, share etc)
            social_dominance: 11.266, //The percent of the total social volume that this topic represents
        """
        # Send a request to the lunarcrush v4 REST api
        coin_id = self.lc_ids[self.coin_name]
        
        url = f"https://lunarcrush.com/api4/public/coins/{coin_id}/time-series/v2"
        os.environ["LUNAR_CRUSH_API_KEY"] = "imhsiympkj8e55rufld8oq1ecl9yupntkk4m3ep"
        headers = {'Authorization': f'Bearer {os.environ.get("LUNAR_CRUSH_API_KEY")}'}

        response = requests.request("GET", url, headers=headers).json()

        lc_columns = [
            'time',
            'circulating_supply', #
            'galaxy_score',
            'volatility', #
            'alt_rank',
            # 'market_dominance',
            'contributors_active', #
            'posts_active', #
            'interactions', #
            'social_dominance', #
        ]

        data = (
            pd.DataFrame(response['data'])
            .filter(items=lc_columns)
            .astype(float)
            .set_index('time')
        )

        data.index = pd.to_datetime(data.index.to_series(), unit='s')
        # data.dropna(axis=1, how='all', inplace=True)

        data.columns = 'sentiment_lc_' + data.columns

        if debug:
            LOGGER.debug(
                'Lunar Crush shape: %s\n'
                'utc_now: %s\n'
                'last index minute: %s\n'
                'lc_data: \n%s\n',
                data.shape, 
                datetime.now(timezone.utc), 
                data.index[-1].minute, 
                data.tail(10).to_string()
            )

        # if self.intervals == '30min':
        #     if debug:
        #         LOGGER.debug(f'hourly lunar_crush.tail(): \n{lc_data.tail()}\n')
        #     data = pd.DataFrame(
        #         data.groupby(pd.Grouper(freq=self.intervals)).first()
        #     )

        return data

    def get_sentiment_data(
        self, 
        periods: int,
        accelerated: bool = False,
        category_features: Dict[str, List[str]] = None,
        expected_idx: pd.DatetimeIndex = None,
        save_mock: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        INTRA-DAY FGI

        Santiment: Sanpy --> La versión gratis no te sirve (pero la versión paga está al palo)
            - Chequear los tutoriales que tenes guardados

        import san
        san.ApiConfig.api_key = '27ef5rl5gjdfkqwd_snwpnouivohv6rmi'
        """
        
        # Find wether or not to find info
        find_data = True

        if accelerated:
            # Find long_short columns
            long_short_cols = category_features['long_short']

            if len(long_short_cols) == 0:
                find_data = False

        if find_data:
            i = 1
            lc_data: pd.DataFrame = None
            while lc_data is None and i < 5 and self.coin_name != 'LUNA2':
                try:
                    lc_data = self.get_lc_data(debug=debug)

                    if save_mock:
                        self.save_mock_asset(
                            asset=lc_data,
                            asset_name='lc_data'
                        )
                except Exception as e:
                    LOGGER.error(
                        'Unable to retrieve lc_data (%s, %s).\n'
                        'Retrying for the %sth time.\n'
                        'Exception: %s\n',
                        self.coin_name,
                        self.intervals,
                        i+1, e
                    )
                    time.sleep(5)
                i += 1

            if lc_data is None:
                return None
            return lc_data.tail(periods)
        else:
            expected_columns = [
                'sentiment_lc_circulating_supply',
                'sentiment_lc_galaxy_score',
                'sentiment_lc_volatility',
                'sentiment_lc_alt_rank',
                'sentiment_lc_contributors_active',
                'sentiment_lc_posts_active',
                'sentiment_lc_interactions',
                'sentiment_lc_social_dominance'
            ]

            return pd.DataFrame(
                np.zeros(shape=(len(expected_idx), len(expected_columns))),
                columns=expected_columns,
                index=expected_idx
            )

    def check_null_last_obs(
        self, 
        df: pd.DataFrame, 
        debug: bool = False
    ) -> None:
        # Check Null Last Obs
        if self.raw_data is not None:
            last_obs = self.raw_data.index[-1]
        else:
            last_obs = df.index[-1]

        category_names = [
            'coin', 
            'long_short', 
            'other_coins', 
            'stock', 
            'economic',
            'sentiment_lc', 
            'sentiment_btc_fgi', 
            'on_chain'
        ]
        
        category_columns = {
            cat: [c for c in df.columns if c.startswith(cat)] for cat in category_names
        }

        self.null_last_obs_cols = []
        for cols in ['coin', 'long_short', 'other_coins']:  # 'economic']:
            null_last_obs = [c for c in category_columns[cols] if (last_obs in df.index and isnan(df[c].at[last_obs]))]
            if len(null_last_obs) > 0:
                self.null_last_obs_cols.extend(null_last_obs)

        self.null_last_obs_cols: List[str] = list(filter(lambda x: type(x) == str or not isnan(x), self.null_last_obs_cols))

        # Stock Data
        if last_obs.weekday() < 5 and last_obs in df.index:
            real_last_obs_time = last_obs.time() # (last_obs + pd.Timedelta(value=self.mbp, unit='minutes')).time()
            start_time = datetime.strptime("13:30:00", "%H:%M:%S").time()
            end_time = datetime.strptime("19:30:00", "%H:%M:%S").time()

            if debug:
                LOGGER.debug(
                    'last_obs_time: %s\n'
                    'real_last_obs_time: %s\n'
                    'start_time: %s\n'
                    'end_time: %s\n'
                    'last_obs.weekday(): %s\n'
                    'start_time <= real_last_obs_time <= end_time: %s\n',
                    last_obs.time(), 
                    real_last_obs_time, 
                    start_time, 
                    end_time, 
                    last_obs.weekday(), 
                    start_time <= real_last_obs_time <= end_time
                )

            if start_time <= real_last_obs_time <= end_time:
                # Check if today is a holiday in the US
                us_holidays = holidays.US()
                today = date.today()

                if today not in us_holidays:
                    # LOGGER.debug([df[c].at[last_obs] for c in category_columns['stock']])
                    # LOGGER.debug([isnan(df[c].at[last_obs]) for c in category_columns['stock']])
                    self.null_last_obs_cols.extend([c for c in category_columns['stock'] if isnan(df[c].at[last_obs])])

        # Lunar Crush Data + BTC Fear & Greed index
        # if last_obs.minute == 0 and last_obs in df.index:
        #     sentiment_cols = category_columns['sentiment_lc'] + category_columns['sentiment_btc_fgi']
        #     self.null_last_obs_cols.extend([c for c in sentiment_cols if
        #                                     (type(df[c].at[last_obs]) != str and isnan(df[c].at[last_obs]))])

        if debug:
            LOGGER.debug('self.null_last_obs_cols: %s\n', self.null_last_obs_cols)
            # for c in self.null_last_obs_cols:
            #     LOGGER.debug(df[c].tail())

    @timing
    def extractor_pipeline(
        self, 
        periods: int = None,
        loaded_collective_data: pd.DataFrame = None,
        skip_collective_data_check: bool = False,
        validate_collective_data_: bool = True,
        save_collective_data: bool = True,
        validate_data: bool = True,
        accelerated: bool = False,
        category_features: Dict[str, List[str]] = None,
        save_mock: bool = False,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        if periods is None:
            periods = self.periods + 5

        # Extract Other Coins & Stock Data
        collective_df: pd.DataFrame = get_collective_data(
            client=self.client,
            loaded_collective_data=loaded_collective_data,
            accelerated=accelerated,
            category_features=category_features,
            other_coins=self.other_coins,
            intervals=self.intervals,
            periods=periods,
            yfinance_params=self.yfinance_params,
            skip_check=skip_collective_data_check,
            validate=validate_collective_data_,
            parallel=True,
            save=save_collective_data,
            overwrite=self.overwrite,
            debug=debug
        )

        if save_mock:
            self.save_mock_asset(
                asset=collective_df,
                asset_name='collective_df'
            )
        
        # Extract New Coin Data
        df = self.get_new_data(periods=periods)

        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='client_data'
            )

        # if self.coin_name not in self.other_coins:
        #     df = self.get_new_data(periods=periods)
        # else:
        #     extract_cols = [c for c in collective_df.columns if self.coin_name in c]
        #     rename_dict = {col: col.replace(f'other_coins_{self.coin_name}', 'coin') for col in extract_cols}
        #     rename_dict[f'other_coins_{self.coin_name}_volume'] = 'ta_volume'

        #     df = (
        #         collective_df
        #         .filter(items=extract_cols)
        #         .rename(columns=rename_dict)
        #         .copy()
        #     )
        #     # collective_df.drop(columns=extract_cols, inplace=True)

        def drop_col(col):
            if 'high' in col or 'low' in col or 'open' in col or 'volume' in col:
                return True
            return False

        drop_cols = [c for c in collective_df.columns if drop_col(c)]
        collective_df.drop(columns=drop_cols, inplace=True)
        real_periods = len(df)

        # Calculate Derivarives
        df['coin_return'] = df['coin_price'].pct_change()
        df['coin_acceleration'] = df['coin_return'].diff()
        df['coin_jerk'] = df['coin_acceleration'].diff()

        if debug:
            LOGGER.debug(
                'data: \n%s\n'
                'collective_data: \n%s\n',
                df.tail(10).to_string(),
                collective_df.tail(10).to_string()
            )

        # Concatenate df & collective_data
        df: pd.DataFrame = pd.concat([df, collective_df.loc[collective_df.index.isin(df.index)]], axis=1)
        
        # Create target_data
        target_cols = [
            'target_price',
            'target_return',
            'target_acceleration',
            'target_jerk'
        ]
        mirror_cols = [
            'coin_price', 
            'coin_return', 
            'coin_acceleration', 
            'coin_jerk'
        ]
        df[target_cols] = df[mirror_cols].copy()

        # Define target index
        target_idx = df.index.copy()

        if debug:
            LOGGER.debug(
                'utcnow: %s\n'
                'df.index: %s\n'
                'coin, other coins & stock df (shape: %s): \n%s\n',
                datetime.now(timezone.utc),
                df.index,
                df.shape,
                df.tail(10).to_string()
            )

        # Define features
        if self.features:
            # Concat Long/Short Ratio
            long_short_df = self.get_long_short_data(
                periods=periods,
                accelerated=accelerated,
                category_features=category_features,
                expected_idx=df.index,
                save_mock=save_mock,
                debug=debug
            )

            if long_short_df is not None:
                df = pd.concat([df, long_short_df], axis=1)

            # Concat Sentiment Data
            sentiment_df = self.get_sentiment_data(
                periods=real_periods,
                accelerated=accelerated,
                category_features=category_features,
                expected_idx=df.index,
                save_mock=save_mock,
                debug=debug
            )

            if sentiment_df is not None:
                df = pd.concat([df, sentiment_df], axis=1)
            
            # Define unused_data
            self.unused_data: pd.DataFrame = df.loc[
                ~(df.index.isin(target_idx)) &
                (df.index > target_idx[-1])
            ]
            
            self.unused_data.index = pd.to_datetime(self.unused_data.index.to_series())

            # Remove unused_data
            df = (
                df
                .loc[(df.index.isin(target_idx)) &
                     (df['coin_price'].notnull())]
                .replace([np.inf, -np.inf], np.nan)
            )
            
            if debug:
                LOGGER.debug(
                    'self.unused_data: \n%s\n'
                    'df.shape: %s\n',
                    self.unused_data, df.shape
                )
        
        # Transform to datetime index
        df.index = pd.to_datetime(df.index.to_series())
        
        # Validate Data
        if validate_data:
            df = self.validate_data(
                df=df.copy(),
                repair=True,
                debug=debug,
                **update_expectations
            )
        
        # Correct DF Columns Order
        df = self.correct_columns_order(df=df.copy())

        # Check self.null_last_obs
        self.check_null_last_obs(
            df=df.copy(), 
            debug=debug
        )

        # Re-run extractor pipeline if there are unexpected null observations
        while len(self.null_last_obs_cols) > 0 and self.i >= 0 and self.i < 3:
            LOGGER.warning(
                'There are null last obs in %s new raw_data.\n'
                'Therefore, new data will be fetched again for the %sth time.\n'
                'self.null_last_obs_cols: %s\n',
                self.coin_name, self.i + 1, self.null_last_obs_cols
            )

            self.i += 1

            df = self.extractor_pipeline(
                periods=periods, 
                skip_collective_data_check=False,
                validate_collective_data_=True,
                save_collective_data=save_collective_data,
                validate_data=validate_data,
                debug=debug,
                **update_expectations
            )

        if self.i == 3:
            LOGGER.warning(
                '%s extractor_pipeline was run %s times (in the same run).\n'
                'self.i will be set to -1, and new iterations will be ignored.\n',
                self.coin_name, self.i
            )
            self.i = -1
        elif self.i != -1:
            self.i = 0

        if save_mock:
            self.save_mock_asset(
                asset=df,
                asset_name='raw_data'
            )
        
        return df
    
    # @timing
    def update(
        self,
        debug: bool = False,
        **update_params
    ) -> None:
        # Set Up Update Parameters
        complete_update_params = {
            'update_data': False,
            'force_update': False,
            'update_expectations': False,
            'validate_data': False,
            'save_backup': False,
            'save': False
        }
        for k, v in complete_update_params.items():
            if k not in update_params.keys():
                update_params[k] = v

        # Update Expectations
        if update_params['update_expectations']:
            self.update_expectations(debug=debug)

        # Find delta_mins
        self.i = 0
        delta_mins = pd.Timedelta(minutes=self.mbp * 2)

        # Update Raw Data (and Unused Data)
        if self.raw_data is None:
            LOGGER.warning(
                '%s (%s): self.raw_data is None, thus it will be re-written.\n'
                'self.raw_data: %s\n',
                self.coin_name, self.intervals, self.raw_data
            )

            # self.raw_data = self.extractor_pipeline(
            #     periods=self.periods,
            #     skip_collective_data_check=False,
            #     validate_collective_data_=True,
            #     save_collective_data=update_params['save'],
            #     debug=debug,
            #     **{
            #         'check_null_values': False
            #     }
            # )
            # print(f'self.raw_data.shape ({self.coin_name}): {self.raw_data.shape}\n')
            LOGGER.critical('%s raw_data is None!.', self.coin_name)
            raise Exception(f'{self.coin_name} raw_data is None!.\n\n')

        elif (
            (
                datetime.now(timezone.utc).replace(tzinfo=None) > self.raw_data.index[-1] + delta_mins 
                and update_params['update_data']
            )
            or len(self.null_last_obs_cols) > 0
            or update_params['force_update']
        ):
            # Define new periods
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_periods = np.max([int((utc_now - self.raw_data.index[-1]).seconds / (60 * self.mbp)), 0]) + self.new_n

            # Extract new_data
            new_data: pd.DataFrame = self.extractor_pipeline(
                periods=new_periods, 
                skip_collective_data_check=False,
                validate_collective_data_=True,
                save_collective_data=update_params['save'],
                debug=debug,
                **{
                    'check_null_values': False,
                    'expected_periods': new_periods
                }
            )
            col_diff = list(set(new_data.columns).symmetric_difference(set(self.raw_data.columns)))
            if len(col_diff) > 0:
                LOGGER.warning(
                    'self.raw_data and new_data have different columns (%s).\n'
                    'new_data.columns: %s\n'
                    'self.raw_data.columns: %s\n'
                    'col_diff: %s\n'
                    'Attempting to correct columns\n',
                    self.coin_name, new_data.columns, self.raw_data.columns, col_diff
                )

                self.raw_data = self.correct_columns(
                    df=self.raw_data.copy()
                )
            
            self.raw_data = (
                self.raw_data.iloc[:-24]
                .combine_first(new_data)
                .combine_first(self.raw_data)
                .sort_index(ascending=True)
                # .filter(items=new_data.columns.tolist())
            )

            self.raw_data = self.raw_data.loc[
                ~self.raw_data.index.duplicated(keep='last')
            ]

            if debug:
                LOGGER.debug(
                    'self.raw_data (after update): \n%s\n'
                    'self.raw_data.shape (after update): %s\n',
                    self.raw_data.tail().to_string(), self.raw_data.shape
                )

        # Validate Data
        if update_params['validate_data']:
            self.raw_data = self.validate_data(
                df=self.raw_data.copy(),
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )
        
        if debug:
            msno.matrix(self.raw_data.iloc[-self.periods:])
            plt.show()

        if self.periods > self.raw_data.shape[0]:
            LOGGER.warning(
                'self.raw_data.shape[0] < self.periods, so self.periods will be updated.\n'
                'self.raw_data.shape[0]: %s\n'
                'self.periods: %s\n',
                self.raw_data.shape[0], self.periods
            )
            self.periods = self.raw_data.shape[0]

        if len(self.null_last_obs_cols) > 0:
            show_raw_data = [c for c in self.null_last_obs_cols if c in self.raw_data.columns]
            if len(show_raw_data) > 0:
                LOGGER.warning(
                    'some last value features are null in %s.\n'
                    'self.null_last_obs_cols: %s\n'
                    'self.raw_data.iloc[-5:]: \n%s\n',
                    self.coin_name, 
                    self.null_last_obs_cols, 
                    self.raw_data[show_raw_data].iloc[-5:].to_string()
                )
                
        # Save Backup
        if update_params['save_backup']:
            self.save_backup(debug=debug)

        # Save Data Extractor
        if update_params['save']:
            self.save(debug=debug)

    def update_expectations(
        self,
        debug: bool = False
    ) -> None:
        # Define asset_path
        asset_path = f"{Params.bucket}/data_processing/data_extractor/{self.intervals}/{self.coin_name}/{self.coin_name}_raw_data.parquet"

        # Extract expected cols
        expected_cols: List[str] = Params.fixed_params.get("raw_data_columns").copy()
        for coin in self.other_coins:
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
        def find_null_perc_allowed(col: str):
            if (
                col.startswith('target') 
                or col.startswith('coin') 
                or col.startswith('ta')
            ):
                return 0.03
            if (
                col.startswith('on_chain')
                or col.startswith('other_coins')
            ):
                return 0.05
            if col.startswith('long_short'):
                if self.coin_name == 'BTC':
                    return 0.9
                return 0.45
            if col.startswith('sentiment_lc'):
                if self.intervals == '30min':
                    return 0.96
                return 0.92
            if col.startswith('stock'):
                return 0.85
            return 0.0
            
        null_perc_allowed: Dict[str, float] = {
            col: find_null_perc_allowed(col) for col in expected_cols
        }

        # Define duplicate rows subset
        duplicate_rows_subset = expected_cols.copy()

        # Expected periods
        expected_periods = Params.data_params.get("periods")
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < expected_periods:
            expected_periods = raw_data_periods

        # Define Expectations
        expectations = {
            "asset_path": asset_path,
            "check_new_missing_data": True,            
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
            "check_excess_features": True,
            "check_short_length": True,

            "expected_cols": expected_cols,
            "expected_schema": expected_schema,
            "max_values_allowed": max_values_allowed,
            "min_values_allowed": min_values_allowed,
            "unique_values_allowed": unique_values_allowed,
            "null_perc_allowed": null_perc_allowed,

            "duplicate_rows_subset": duplicate_rows_subset,
            "outliers_dict": None,
            "max_features_perc": Params.data_params.get("max_features"),
            "other_coins": self.other_coins,
            "expected_periods": expected_periods            
        }

        if debug:
            LOGGER.info('raw_data expectations:\n%s\n', pformat(expectations))

        # Save Expectations
        expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
        write_to_s3(
            asset=expectations,
            path=f"{expectations_base_path}/{self.coin_name}_raw_data_expectations.json"
        )

    def diagnose_data(
        self,
        df: pd.DataFrame = None,
        debug: bool = False,
        **update_expectations: dict
    ) -> Dict[str, bool]:
        if df is None:
            df = self.raw_data.copy()

        # Find Diagnostics Dict
        diagnostics_dict = find_data_diagnosis_dict(
            df_name="raw_data",
            intervals=self.intervals,
            coin_name=self.coin_name,
            df=df,
            debug=False, # debug
            **update_expectations
        )

        if debug:
            LOGGER.debug(
                '%s raw_data diagnostics_dict:\n%s\n', 
                self.coin_name, pformat(diagnostics_dict)
            )

        return diagnostics_dict

    def validate_data(
        self,
        df: pd.DataFrame = None,
        repair: bool = True,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        def extract_new_repair_data(df_: pd.DataFrame):
            # Complete collective_data (if needed)        
            # other_coins = Params.other_coins_json[self.intervals][:self.other_coins_n]
            # if self.coin_name in other_coins:
            #     collective_data = get_collective_data(
            #         client=self.client,
            #         other_coins=other_coins,
            #         intervals=self.intervals,
            #         periods=df_.shape[0],
            #         parallel=True,
            #         save=True,
            #         debug=False
            #     )

            #     correct_data = self.client.get_data(
            #         coin_name=self.coin_name,
            #         intervals=self.intervals,
            #         periods=df_.shape[0]
            #     )
            #     correct_data = correct_data.rename(columns={
            #         'open': f'other_coins_{self.coin_name}_open',
            #         'high': f'other_coins_{self.coin_name}_high',
            #         'low': f'other_coins_{self.coin_name}_low',
            #         'price': f'other_coins_{self.coin_name}_price',
            #         'volume': f'other_coins_{self.coin_name}_volume'
            #     })

            #     collective_data = correct_data.combine_first(collective_data)

            #     # Save collective_data
            #     save_collective_data(
            #         df=collective_data,
            #         intervals=self.intervals,
            #         overwrite=self.overwrite
            #     )

            # Extract new raw data
            freq = {
                '30min': '30min',
                '60min': '60min',
                '1d': '1D'
            }[self.intervals]

            full_idx = pd.date_range(df_.index.min(), df_.index.max(), freq=freq)
            
            new_raw_data = self.extractor_pipeline(
                periods=len(full_idx),
                skip_collective_data_check=True,
                validate_collective_data_=True,
                validate_data=False,
                **{'expected_periods': len(full_idx)}
            )

            # new_raw_data = new_raw_data.loc[new_raw_data.index.isin(self.raw_data.index)]

            return new_raw_data
        
        if df is None:
            df = self.raw_data.copy()
        
        # Find Diagnostics Dict
        diagnostics_dict = self.diagnose_data(
            df=df,
            debug=debug,
            **update_expectations
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                '%s raw_data needs repair.\n'
                'diagnostics_dict:\n%s\n',
                self.coin_name, pformat(diagnostics_dict)
            )

            if repair:
                LOGGER.info("Repairing %s raw_data...\n", self.coin_name)
                """
                Diagnostics Dict:
                    - has_missing_new_data
                    - has_missing_columns
                    - has_missing_rows
                    - has_null_values
                    - has_duplicated_idx
                    - has_duplicated_columns
                    - has_excess_features
                    - has_short_length
                }
                """
                # Load Expectations
                expectations_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
                expectations: dict = load_from_s3(
                    path=f"{expectations_path}/{self.coin_name}_raw_data_expectations.json"
                )
                
                # Repair Missing Data
                # if 'has_missing_new_data' in diagnostics_dict.keys() and diagnostics_dict['has_missing_new_data']:
                #     repair_missing_new_data(self.coin_name, self.intervals)

                # Add Missing Rows
                if diagnostics_dict['has_missing_rows']:
                    df = add_missing_rows(
                        df=df.copy(),
                        intervals=self.intervals
                    )
                
                # Drop Unexpected Columns
                if diagnostics_dict['has_unexpected_columns']:
                    df = drop_unexpected_columns(
                        df=df.copy(),
                        expected_columns=expectations['expected_cols'],
                        coin_name=None,
                        df_name='raw_data'
                    )
                
                # Drop duplicated IDX & Cols
                if diagnostics_dict['has_duplicated_idx'] or diagnostics_dict['has_duplicated_columns']:
                    df = drop_duplicates(df=df.copy())
                
                # Repair Other Coins, Stock & On-Chain Cols
                if diagnostics_dict['has_missing_columns']:
                    df = self.correct_columns(df=df.copy())
                
                if (
                    diagnostics_dict.get('has_missing_rows', False)
                    or diagnostics_dict.get('has_null_values', False)
                    or diagnostics_dict.get('has_short_length', False)
                ):
                    # Load new raw_data
                    new_raw_data = extract_new_repair_data(df_=df.copy())
                    new_raw_data = new_raw_data.filter(items=[c for c in new_raw_data.columns if c in df.columns])

                    # Combine with new_raw_data
                    df = df.combine_first(new_raw_data)

                    # Load backup_data
                    try:
                        backup_data: pd.DataFrame = load_from_s3(
                            path=f"{self.backup_path}/{self.coin_name}_raw_data_backup.parquet",
                            load_reduced_dataset=False
                        )
                        backup_data = backup_data.filter(items=[c for c in backup_data.columns if c in df.columns])

                        # Combine with backup_data
                        df = df.combine_first(backup_data)
                    except Exception as e:
                        LOGGER.error(
                            'Unable to load %s backup_data.\n'
                            'Exception: %s.\n',
                            self.coin_name, e
                        )
                
                # self.save()
            else:
                LOGGER.warning(
                    '%s raw_data needed repair, but "repair" parameter was set to False.\n'
                    'Thus, %s raw_data will NOT be repaired.\n',
                    self.coin_name, self.coin_name
                )
        
        return df
    
    def save_backup(
        self,
        debug: bool = False
    ) -> None:
        # Find if enough time has passed
        time_passed = 60*60*24*10
        if self.last_backup is not None:
            time_passed = (self.raw_data.index[-1] - self.last_backup).total_seconds()

        if time_passed > 60*60*48:
            LOGGER.info('Time since last backup: %s hs.', int(time_passed/(60*60)))

            diagnostics_dict = self.diagnose_data(
                df=self.raw_data.copy(),
                debug=debug,
                **{'expected_periods': self.periods}
            )

            if needs_repair(diagnostics_dict):
                LOGGER.info(
                    'Unable to save backup %s.\n'
                    'Attempting to repair raw_data.\n',
                    self.coin_name
                )

                self.raw_data = self.validate_data(
                    df=self.raw_data.copy(),
                    repair=True,
                    debug=debug,
                    **{'expected_periods': self.periods}
                )
                diagnostics_dict = self.diagnose_data(
                    df=self.raw_data.copy(),
                    debug=debug,
                    **{'expected_periods': self.periods}
                )

            if not needs_repair(diagnostics_dict):
                LOGGER.info('Saving %s raw_data to backup.', self.coin_name)
                # Write self.raw_data
                write_to_s3(
                    asset=self.raw_data,
                    path=f'{self.backup_path}/{self.coin_name}_raw_data_backup.parquet',
                    overwrite=self.overwrite
                )

                self.last_backup = self.raw_data.index[-1]
                self.save()
            else:
                LOGGER.info(
                    'Enough time has passed, but raw_data needs to be repaired.\n'
                    'Thus, backup save will be ignored.\n'
                )
        else:
            if debug:
                LOGGER.debug(
                    'Not enough time has passed for backup save.\n'
                    'time_passed: %s\n'
                    'self.last_backup: %s\n',
                    time_passed, self.last_backup
                )

    def revert_to_backup(self) -> None:
        LOGGER.warning(
            'self.raw_data will be reverted to last saved backup (%s, %s).',
            self.coin_name, self.intervals
        )
        
        time_passed = np.inf
        if self.last_backup is not None:
            time_passed = (self.raw_data.index[-1] - self.last_backup).total_seconds()

        if time_passed < 60*60*24*3:
            try:
                # Load self.raw_data
                self.raw_data = load_from_s3(
                    path=f'{self.backup_path}/{self.coin_name}_raw_data_backup.parquet'
                )
            except Exception as e:
                LOGGER.error(
                    'Unable to revert %s raw data to backup.'
                    'Exception: %s.\n',
                    self.coin_name, e
                )
        else:
            if not(np.isinf(time_passed)):
                hs_passed = int(time_passed / (60*60))
            else:
                hs_passed = np.inf

            LOGGER.warning(
                '%s raw_data was not reverted to backup.\n'
                'Time passed since last backup: %s hs.\n',
                self.coin_name, hs_passed
            )

        self.save(debug=True)

    def save_mock_asset(
        self,
        asset: Any,
        asset_name: str
    ) -> None:
        # print(f'Saving {asset_name} - [shape: {asset.shape}]')

        # Define base_paths
        base_path = f"{Params.bucket}/mock/data_processing/data_extractor/{self.intervals}/{self.coin_name}"

        # Define save_path
        if asset_name == 'collective_df':
            save_path = f"{base_path}/collective_df.parquet"
        elif asset_name == 'client_data':
            save_path = f"{base_path}/client_data.parquet"
        elif asset_name == 'global_long_short_data':
            save_path = f"{base_path}/global_long_short_data.parquet"
        elif asset_name == 'top_traders_long_short_data':
            save_path = f"{base_path}/top_traders_long_short_data.parquet"
        elif asset_name == 'lc_data':
            save_path = f"{base_path}/lc_data.parquet"
        elif asset_name == 'raw_data':
            save_path = f"{base_path}/raw_data.parquet"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Save asset to S3
        write_to_s3(asset=asset, path=save_path, overwrite=True)
    
    def load_mock_asset(
        self,
        asset_name: str
    ) -> pd.DataFrame:
        # Define base_path
        base_path = f"{Params.bucket}/mock/data_processing/data_extractor/{self.intervals}/{self.coin_name}"

        # Define load_path
        if asset_name == 'collective_df':
            load_path = f"{base_path}/collective_df.parquet"
        elif asset_name == 'client_data':
            load_path = f"{base_path}/client_data.parquet"
        elif asset_name == 'global_long_short_data':
            load_path = f"{base_path}/global_long_short_data.parquet"
        elif asset_name == 'top_traders_long_short_data':
            load_path = f"{base_path}/top_traders_long_short_data.parquet"
        elif asset_name == 'lc_data':
            load_path = f"{base_path}/lc_data.parquet"
        elif asset_name == 'raw_data':
            load_path = f"{base_path}/raw_data.parquet"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Load asset from S3
        asset = load_from_s3(path=load_path, ignore_checks=True)
        
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

        # Write pickled attrs
        write_to_s3(
            asset=pickle_attrs,
            path=f"{self.save_path}/{self.coin_name}_data_extractor_attr.pickle"
        )
        
        if debug:
            for attr_key, attr_value in pickle_attrs.items():
                LOGGER.debug('Saved pickled %s:\n%s\n', attr_key, pformat(attr_value))
        
        """
        Save .parquet files
        """
        for attr_name in self.load_parquet:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                # Write parquet df
                write_to_s3(
                    asset=getattr(self, attr_name),
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
            pickle_attrs: dict = load_from_s3(
                path=f"{self.save_path}/{self.coin_name}_data_extractor_attr.pickle"
            )

            for attr_key, attr_value in pickle_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

                    if debug:
                        LOGGER.debug('Loaded pickle %s:\n%s\n', attr_key, pformat(attr_value))
        except Exception as e:
            LOGGER.error(
                'Unable to load data_extractor (%s: %s).\n'
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
            self.raw_data: pd.DataFrame = load_from_s3(
                path=f"{self.save_path}/{self.coin_name}_raw_data.parquet",
                load_reduced_dataset=load_reduced_dataset
            )
        except Exception as e:
            LOGGER.error(
                'Unable to load %s_raw_data.\n'
                'Exception: %s.\n',
                self.coin_name, e
            )
        
        if self.raw_data is None:
            self.revert_to_backup()

        # Update periods if required
        if load_reduced_dataset:
            self.periods = self.raw_data.shape[0]
        
        # Check loaded data
        # diagnostics_dict = self.diagnose_data(
        #     **{
        #         'check_new_missing_data': False,
        #         'expected_periods': self.periods
        #     }
        # )
        # if needs_repair(diagnostics_dict):
        #     print(f'[WARNING] Loaded {self.coin_name} raw_data needs repair.\n'
        #           f'Thus, it will be reverted to backup.\n'
        #           f'diagnostics_dict:')
        #     pprint(diagnostics_dict)
        #     print('\n')

        #     self.revert_to_backup()
