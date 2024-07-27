from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing

from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
import numpy as np
import time
from numba import jit
from copy import deepcopy
from pprint import pprint, pformat
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


class TradingTable(pd.DataFrame):

    save_attr_list = [
        'model_id_',
        'coin_name_',
        'intervals_',

        'algorithm',
        'method',
        'pca',
        'trading_parameters',

        'table_name',

        'is_dummy',

        'resid_dist',
        'resid_params',
        'dist_name',

        'arg',
        'loc_',
        'scale',

        'pred_resid_correl',
        'resid_alpha',
        'resid_beta',
        'heteroskedasticity',
        'resid_auto_correlations',

        'est_monthly_ret',
        'est_stand_error',
        'est_sharpe_ratio',
        'ml_accuracy',
        'weighted_ml_accuracy',
        'trading_accuracy',
        'weighted_trading_accuracy',
        'n_trade_perc',
        'ret_pvalue',
        'cum_ret',
        'avg_real_trading_return',
        'wape',
        'lsr',
        'ssr',
        'long_rate',

        'base_path'
    ]
    main_cols = [
        'model_id', 'coin_name', 'intervals',
        'open', 'high', 'low', 'price',
        'return_forecast', 'real_return', 'return_residual',
        'increase_chance', 'decrease_chance', 'certainty', 'kind',
        'tp_price', 'sl_price', 'tp_triggered', 'sl_triggered',
        'trading_signal', 'passive_trading', 'flow', 
        'exposure', 'exposure_adj', 'leverage',
        'fees', 'trading_return', 'real_trading_return', 'total_cum_returns',
        'ml_accuracy', 'trading_accuracy', 'trade_class'
    ]

    def __init__(
        self, 
        *args,
        model_id: str = None,
        coin_name: str = None,
        intervals: str = None,

        algorithm: str = None,
        method: str = None,
        pca: bool = False,
        trading_parameters: dict = None,

        initialize: bool = False,
        table_name: str = None,
        mock: bool = False,
        load_table: bool = False,
        debug: bool = False,
        **kwargs
    ) -> None:
        super(TradingTable, self).__init__(*args, **kwargs)

        if initialize:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.correct_main_attrs = False

                # General Parameters
                self.model_id_ = model_id
                self.coin_name_ = coin_name
                self.intervals_ = intervals

                self.algorithm = algorithm
                self.method = method
                self.pca = pca
                self.trading_parameters = trading_parameters

                # Load available pairs
                available_pairs = load_from_s3(
                    path=f"{Params.bucket}/utils/available_pairs/available_pairs.json"
                )

                # Find stable_coin
                if coin_name in available_pairs[Params.general_params['stable_coin']]:
                    self.stable_coin_ = Params.general_params['stable_coin']
                else:
                    self.stable_coin_ = 'USDT'

                # Find trading_fees
                self.trading_fees = Params.trading_params.get('trading_fees_dict')[self.stable_coin_][Params.general_params['order_type']]

                # Naming convention:
                #   - <model_id>_<val/test/opt>_trading_df
                self.table_name = table_name

                # Mock & model_id
                self.mock: bool = mock

                if debug:
                    attr_names = [
                        'model_id_', 'coin_name_', 'intervals_', 'trading_fees', 
                        'algorithm', 'method', 'pca', 'trading_parameters'
                    ]
                    if debug:
                        print("Loaded attrs:\n"
                            "{")
                        for attr_name in attr_names:
                            print(f"    '{attr_name}': {getattr(self, attr_name)}")
                        print('}\n\n')

                self.is_dummy = False

                self.resid_dist = None
                self.resid_params = None
                self.dist_name = None

                self.arg = None
                self.loc_ = None
                self.scale = None

                self.pred_resid_correl = None
                self.resid_alpha = None
                self.resid_beta = None
                self.heteroskedasticity = None
                self.resid_auto_correlations = None

                self.est_monthly_ret = None
                self.est_stand_error = None
                self.est_sharpe_ratio = None
                self.ml_accuracy = None
                self.weighted_ml_accuracy = None
                self.trading_accuracy = None
                self.weighted_trading_accuracy = None
                self.n_trade_perc = None
                self.ret_pvalue = None
                self.cum_ret = None
                self.avg_real_trading_return = None
                self.wape = None
                self.lsr = None
                self.ssr = None
                self.long_rate = None

                # Define paths
                # self.s3_base_path = f"{Params.bucket}/trading/trading_table/{self.intervals_}/{model_id}"
                
                if load_table:
                    self.load(debug=debug)

    @property
    def _constructor(self):
        return TradingTable
    
    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/trading/trading_table/{self.intervals_}/{self.model_id_}"
        else:
            return f"{Params.bucket}/trading/trading_table/{self.intervals_}/{self.model_id_}"

    @property
    def general_attrs(self) -> dict:
        attr_names = [
            'table_name',
            'model_id_',
            'coin_name_',
            'stable_coin_',
            'intervals_',
            'trading_fees',

            'is_dummy',
            'algorithm',
            'method',
            'pca',
            'trading_parameters'
        ]
        return {attr_name: getattr(self, attr_name) for attr_name in attr_names}

    @property
    def residuals_attrs(self) -> dict:
        attr_names = [
            'resid_dist',
            'resid_params',
            'dist_name',

            'arg',
            'loc_',
            'scale',

            'pred_resid_correl',
            'resid_alpha',
            'resid_beta',
            'heteroskedasticity',
            'resid_auto_correlations'
        ]
        return {attr_name: getattr(self, attr_name) for attr_name in attr_names}

    @property
    def performance_attrs(self) -> dict:
        attr_names = [
            'est_monthly_ret',
            'est_stand_error',
            'est_sharpe_ratio',
            'ml_accuracy',
            'weighted_ml_accuracy',
            'trading_accuracy',
            'weighted_trading_accuracy',
            'n_trade_perc',
            'ret_pvalue',
            'cum_ret',
            'avg_real_trading_return',
            'wape',
            'lsr',
            'ssr',
            'long_rate'
        ]
        return {attr_name: getattr(self, attr_name) for attr_name in attr_names}
    
    def meets_minimum_criteria(self) -> bool:
        if (
            self.est_monthly_ret is None
            or self.ml_accuracy is None
            or self.weighted_ml_accuracy is None
            or self.trading_accuracy is None
            or self.weighted_trading_accuracy is None
            or self.cum_ret is None
            or self.ret_pvalue is None
        ):
            return False
        
        if (
            # Estimated monthly returns must be > 10%
            self.est_monthly_ret <= 0.12
            
            # ML Accuracy must be > 51%
            or self.ml_accuracy <= 0.51

            # Weighted ML Accuracy must be > 51%
            or self.weighted_ml_accuracy <= 0.51

            # Trading Accuracy must be > 51%
            or self.trading_accuracy <= 0.51

            # Weighted Trading Accuracy must be > 51%
            or self.weighted_trading_accuracy <= 0.51

            # Cumulative returns must be > 12%
            or self.cum_ret <= 0.12

            # Return p-value must be lower than 15%
            or self.ret_pvalue >= 0.15
        ):
            return False
        return True
    
    @property
    def tuning_metric(self) -> float:
        # Check if performances are above minimum criteris
        if self.meets_minimum_criteria():
            # Find Weighted ML Accuracy Score
            weighted_ml_accuracy_score = (
                (int(round(self.weighted_ml_accuracy * 100, 0)) - 50) 
                * self.n_trade_perc
                * (1 - self.ret_pvalue)
            )
            
            # Find Estimated Monthly Return Score
            est_monthly_ret_score = round(self.est_monthly_ret * 100, 3)

            # Find Estimated Standard Error Score
            est_stand_error_score = max([0, 1 - self.est_stand_error])

            # Risk-Reward Score
            risk_reward_score = round(est_monthly_ret_score * est_stand_error_score, 3)

            # Return tunning metric
            # if risk_reward_score >= 100:
            #     return float(f"{weighted_ml_accuracy_score}99.{str(risk_reward_score).replace('.', '')}")
            # return float(f"{weighted_ml_accuracy_score}{risk_reward_score}")
            return weighted_ml_accuracy_score + risk_reward_score / 9
        return 0

    @property
    def trading_metric(self) -> float:
        # Check if performances are above minimum criteris
        if self.meets_minimum_criteria():
            # Find Weighted ML Accuracy Score
            weighted_trading_accuracy_score = (
                (int(round(self.weighted_trading_accuracy * 100, 0)) - 50) 
                * self.n_trade_perc
                * (1 - self.ret_pvalue)
            )

            # Find Estimated Monthly Return Score
            est_monthly_ret_score = round(self.est_monthly_ret * 100, 3)

            # Find Estimated Standard Error Score
            est_stand_error_score = max([0, 1 - self.est_stand_error])

            # Risk-Reward Score
            risk_reward_score = min([100, round(est_monthly_ret_score * est_stand_error_score, 3)])
            
            # Return trading_metric
            return weighted_trading_accuracy_score + risk_reward_score / 7
        return 0

    def measure_trading_performance(
        self,
        smooth_returns: bool = False,
        return_weight: float = None,
        debug: bool = False
    ) -> None:
        """
        Find estimated Sharpe ratio
        """
        # Find y & X arrays to train on
        y = self['total_cum_returns'].values.reshape(-1, 1)
        X = np.arange(self.shape[0]).reshape(-1, 1)

        # Instanciate & fit Linear model
        linear_model = LinearRegression(n_jobs=-1)
        linear_model.fit(X, y)

        # Find fitted predictions
        est_cum_returns = linear_model.predict(X)

        # Calculate estimated monthly return
        periods_per_month = {
            '30min': 1440,
            '60min': 720
        }
        self.est_monthly_ret = (
            est_cum_returns[periods_per_month[self.intervals_]][0]
            - est_cum_returns[0][0]
        )

        # Calculate estimated Standard Error
        self.est_stand_error = np.sqrt(mean_squared_error(y, est_cum_returns)) / self.est_monthly_ret

        # Calculate estimated sharpe ratio
        self.est_sharpe_ratio = self.est_monthly_ret / self.est_stand_error

        """
        Find ML accuracy on certain predictions
        """
        # Find ml_df
        ml_df: pd.DataFrame = self.loc[
            self['certainty'] >= self.trading_parameters['certainty_threshold'],
            ['ml_accuracy', 'real_trading_return', 'certainty']
        ].copy()

        # Find ml_accuracy
        self.ml_accuracy = ml_df['ml_accuracy'].sum() / ml_df.shape[0]

        # Add ml_accuracy_weight column
        ml_df['ml_accuracy_weight'] = np.abs(ml_df['real_trading_return']) * np.abs(ml_df['certainty'])

        # Re-define ml_accuracy
        ml_df['ml_accuracy'] = np.where(
            ml_df['ml_accuracy'],
            ml_df['ml_accuracy_weight'],
            0
        )  

        self.weighted_ml_accuracy = ml_df['ml_accuracy'].sum() / ml_df['ml_accuracy_weight'].sum()

        """
        Find trading accuracy on actual trades
        """
        # Find trading_df
        trading_df: pd.DataFrame = self.loc[
            self['trading_signal'],
            ['trading_accuracy', 'certainty', 'real_trading_return', 'real_return', 
             'return_residual', 'total_cum_returns', 'trade_class', 'kind']
        ].copy()

        # Smooth Returns
        if smooth_returns:
            mean = trading_df['real_trading_return'].mean()
            std = trading_df['real_trading_return'].std()

            q_high = mean + 1.96 * std
            q_low = mean - 1.96 * std

            trading_df['real_trading_return'] = trading_df['real_trading_return'].apply(
                lambda x: q_high + (x - q_high) * 0.40 if x > q_high
                else q_low - (q_low - x) * 0.40 if x < q_low
                else x
            )

        # Add Returns Weight
        if return_weight is not None:
            """
            x1 * mult - x2 = 0
            x1        + x2 = 2

            A = [[1 * mult, -1]    |    X = [[x1]     |    B = [[0]
                 [1       ,  1]]   |         [x2]]    |         [2]]

            X = inverse(A).B
            """
            A = np.array([[1 * return_weight, -1],
                          [1, 1]])
            B = np.array([[0],
                          [2]])
            X = np.linalg.inv(A).dot(B)
            x1, x2 = float(X[0][0]), float(X[1][0])
            weight_diff = x2 - x1

            # Calculate weight
            trading_df['n'] = [i / trading_df.shape[0] for i in range(1, trading_df.shape[0] + 1)]
            trading_df['ret_weight'] = trading_df.apply(
                lambda row: (x1 + row.n * weight_diff), axis=1
            )

            # Re-define "real_trading_return" column
            trading_df['real_trading_return'] = trading_df['ret_weight'] * trading_df['real_trading_return']

        # Find trading_accuracy
        self.trading_accuracy = trading_df['trading_accuracy'].sum() / trading_df.shape[0]

        # Add trading_accuracy_weight column
        trading_df['trading_accuracy_weight'] = np.abs(trading_df['real_trading_return'])

        # Re-define trading_accuracy
        trading_df['trading_accuracy'] = np.where(
            trading_df['trading_accuracy'],
            trading_df['trading_accuracy_weight'],
            0
        )  

        self.weighted_trading_accuracy = trading_df['trading_accuracy'].sum() / trading_df['trading_accuracy_weight'].sum()

        """
        Find n_trade_perc, ret_pvalue, cum_ret, avg_real_trading_return, wape & secondary accuracy metrics
        """
        # Find Trade Percentage
        self.n_trade_perc = trading_df.shape[0] / self.shape[0]

        # Calculate return p-value
        t, p = st.ttest_1samp(trading_df['real_trading_return'], 0)
        if t > 0:
            self.ret_pvalue = p / 2
        else:
            self.ret_pvalue = 1.0

        # Cumulative Returns
        self.cum_ret = trading_df['total_cum_returns'].iat[-1]

        # Average real trading return
        self.avg_real_trading_return = trading_df['real_trading_return'].mean()

        # WAPE
        self.wape = 100 * trading_df['return_residual'].abs().sum() / trading_df['real_return'].abs().sum()

        # Accuracy secondary metrics        
        vals = ['long_point', 'short_point', 'long_fail', 'short_fail']
        counts = pd.DataFrame(trading_df['trade_class'].value_counts())
        for v in vals:
            if v not in counts.index:
                counts.loc[v] = 0

        self.lsr = float(counts.loc['long_point'] / (counts.loc['long_point'] + counts.loc['long_fail']))
        self.ssr = float(counts.loc['short_point'] / (counts.loc['short_point'] + counts.loc['short_fail']))

        self.long_rate = float(trading_df.loc[trading_df['kind'] == 'long'].shape[0] / trading_df.shape[0])

        if debug:
            print(f'self.est_monthly_ret: {self.est_monthly_ret}\n'
                  f'self.est_stand_error: {self.est_stand_error}\n'
                  f'self.est_sharpe_ratio: {self.est_sharpe_ratio}\n'
                  f'self.ml_accuracy: {self.ml_accuracy}\n'
                  f'self.weighted_ml_accuracy: {self.weighted_ml_accuracy}\n'
                  f'self.trading_accuracy: {self.trading_accuracy}\n'
                  f'self.weighted_trading_accuracy: {self.weighted_trading_accuracy}\n'
                  #  f'self.n_trade_perc: {self.n_trade_perc}\n'
                  f'self.ret_pvalue: {self.ret_pvalue}\n'
                  f'self.cum_ret: {self.cum_ret}\n'
                  f'self.avg_real_trading_return: {self.avg_real_trading_return}\n'
                  #   f'self.wape: {self.wape}\n'
                  #   f'self.lsr: {self.lsr}\n'
                  #   f'self.ssr: {self.ssr}\n'
                  #   f'self.long_rate: {self.long_rate}\n\n'
                  f'self.tuning_metric: {self.tuning_metric}\n'
                  f'self.trading_metric: {self.trading_metric}\n\n')

    def is_dummy_proof(
        self, 
        debug: bool = False
    ) -> bool:
        # Ignore Dummy Models
        predicted_0 = self.loc[self['return_forecast'] == 0].shape[0] / self.shape[0]
        if predicted_0 > 0.2:
            if debug:
                print(f'predicted_0: {predicted_0}\n')
            return False

        predicted_positive = self.loc[self['return_forecast'] > 0].shape[0] / self.shape[0]
        if predicted_positive > 0.67 or predicted_positive < 0.33:
            if debug:
                print(f'predicted_positive: {predicted_positive}\n')
            return False
        return True
    
    def complete_table(
        self,
        find_best_dist: bool = False,
        dummy_proof: bool = False
    ) -> None:
        if dummy_proof and not self.is_dummy_proof():
            self.is_dummy = True
            return
            
        # Add primary_columns
        primary_columns = ['model_id', 'coin_name', 'intervals']
        i = 0
        for column in primary_columns:
            if column not in self.columns:
                self.insert(loc=i, column=column, value=getattr(self, f'{column}_'))
                i += 1

        # Return Residuals
        self['return_residual'] = self['real_return'] - self['return_forecast']

        # Resid Distribution
        if find_best_dist:
            self.fit_distribution(
                remove_outliers=True
            )
        else:
            if self.resid_dist is None or self.resid_params is None:
                self.fit_distribution(
                    remove_outliers=True,
                    dist_name='norm'
                )
        
        # Decrease Chance
        self['decrease_chance'] = self.apply(
            lambda row: self.resid_dist.cdf(-row.return_forecast, *self.arg, loc=self.loc_, scale=self.scale), 
            axis=1
        )

        # Increase Chance
        self['increase_chance'] = 1 - self['decrease_chance']

        # Certainty
        self['certainty'] = np.maximum(self['increase_chance'].copy(), self['decrease_chance'].copy())

        # Kind
        self['kind'] = np.where(self['increase_chance'] >= self['decrease_chance'], 'long', 'short')

        # ML Accuracy
        # self['ml_accuracy'] = np.where(
        #     np.logical_or(
        #         np.logical_and(self['kind'] == 'long', self['real_return'] >= 0),
        #         np.logical_and(self['kind'] == 'short', self['real_return'] < 0)
        #     ),
        #     True,
        #     False
        # )

        # Re-create trading logic
        # @jit(nopython=True) # https://numba.pydata.org/numba-doc/latest/user/5minguide.html
        def accelerator(
            input_: np.ndarray,
            follow_flow: int,

            ltp: float,
            stp: float, 
            lsl: float,
            ssl: float,

            certainty_threshold: float,
            long_permission: float,
            short_permission: float,
            max_leverage: float,
            trading_fees: float
        ) -> np.ndarray: 
            """
            Define raw output
            """
            output = np.zeros((input_.shape[0], 17))

            """
            Define Columns
            """
            # Define Input Columns
            open = 0
            price = 1
            high = 2
            low = 3
            real_return = 4
            certainty = 5
            kind = 6

            # Define Output Columns
            flow = 0

            tp_price = 1
            sl_price = 2
            tp_triggered = 3
            sl_triggered = 4
            
            trading_signal = 5
            passive_trading = 6

            exposure = 7
            exposure_adj = 8
            leverage = 9
        
            fees = 10
            trading_return = 11
            real_trading_return = 12
            total_cum_returns = 13

            trading_accuracy = 14
            ml_accuracy = 15
            trade_class = 16

            """
            Flow:
            - Flow will be the same as previous flow until there is enough evidence to change flow
            - Enough evidence: an oposit 'kind', where certainty > threshold
            """
            # FLOW
            output[0, flow] = input_[0, kind]

            for i in range(1, output.shape[0]):
                if input_[i, certainty] > certainty_threshold and input_[i, kind] != output[i-1, flow]:
                    output[i, flow] = input_[i, kind]
                else:
                    output[i, flow] = output[i-1, flow]
            
            """
            TP Price, SL Price, TP Triggered, SL Triggered
            """
            # Take Proffit: tp = ltp, stp
            if ltp != None and stp != None:
                if follow_flow == 1:
                    # TP Price
                    output[:, tp_price] = np.where(
                        output[:, flow] == 1, # Flow: long
                        input_[:, open] * (1 + ltp),
                        input_[:, open] * (1 + stp)
                    )
                    # TP Triggered
                    output[:, tp_triggered] = np.where(
                        np.logical_or(
                            np.logical_and(output[:, flow] == 1, input_[:, high] > output[:, tp_price]), # Flow: long and high > tp_price
                            np.logical_and(output[:, flow] == 0, input_[:, low] < output[:, tp_price]) # Flow: short and low < tp_price
                        ),                    
                        1, # True
                        0 # False
                    )
                else:
                    # TP Price
                    output[:, tp_price] = np.where(
                        input_[:, kind] == 1, # kind: long
                        input_[:, open] * (1 + ltp),
                        input_[:, open] * (1 + stp)
                    )
                    # TP Triggered
                    output[:, tp_triggered] = np.where(
                        np.logical_or(
                            np.logical_and(input_[:, kind] == 1, input_[:, high] > output[:, tp_price]), # kind: long and high > tp_price
                            np.logical_and(input_[:, kind] == 0, input_[:, low] < output[:, tp_price]) # kind: short and low < tp_price
                        ),                    
                        1, # True
                        0 # False
                    )
            else:
                output[:, tp_price] = np.nan
                output[:, tp_triggered] = 0

            # Stop Loss: sl = lsl, ssl
            if lsl != None and ssl != None:
                if follow_flow == 1:
                    # SL Price
                    output[:, sl_price] = np.where(
                        output[:, flow] == 1, # Flow: long
                        input_[:, open] * (1 + lsl),
                        input_[:, open] * (1 + ssl)
                    )
                    # SL Triggered
                    output[:, sl_triggered] = np.where(
                        np.logical_or(
                            np.logical_and(output[:, flow] == 1, input_[:, low] < output[:, sl_price]), # Flow: long and low < sl_price
                            np.logical_and(output[:, flow] == 0, input_[:, high] > output[:, sl_price]) # Flow: short and high > sl_price
                        ),                    
                        1, # True
                        0 # False
                    )
                else:
                    # SL Price
                    output[:, sl_price] = np.where(
                        input_[:, kind] == 1, # Kind: long
                        input_[:, open] * (1 + lsl),
                        input_[:, open] * (1 + ssl)
                    )
                    # SL Triggered
                    output[:, sl_triggered] = np.where(
                        np.logical_or(
                            np.logical_and(input_[:, kind] == 1, input_[:, low] < output[:, sl_price]), # Kind: long and low < sl_price
                            np.logical_and(input_[:, kind] == 0, input_[:, high] > output[:, sl_price]) # Kind: short and high > sl_price
                        ),
                        1, # True
                        0 # False
                    )
            else:
                output[:, sl_price] = np.nan
                output[:, sl_triggered] = 0

            """
            Trading Signal:
                - Active position signal:
                    - Certainty is over the required threshold
                    - (or) Previous signal was active and was not forced to close
                        - This is considered as "passive" trading, or keeping the "flow"
                - Inactive trading signal:
                    - Certainty is less that required threshold
                    - Previous position was forced to close (i.e.: tp or sl was triggered)

            Passive Trading:
                - True if there is an active position open, although there is not enough certainty for it.
            """
            # TRADING SIGNAL
            if input_[0, certainty] > certainty_threshold:
                # Active
                output[0, trading_signal] = 1
            else:
                # Inactive
                output[0, trading_signal] = 0

            if follow_flow == 1:
                for i in range(1, output.shape[0]):
                    if (
                        # Certainty is over the required threshold
                        input_[i, certainty] > certainty_threshold
                        # (or) Previous position was active and was not forced to close (passive trading)
                        or (
                            output[i-1, trading_signal] == 1
                            and output[i-1, tp_triggered] == 0 
                            and output[i-1, sl_triggered] == 0
                        )
                    ):
                        # True
                        output[i, trading_signal] = 1
                    else:
                        # False
                        output[i, trading_signal] = 0

                # Correct long_permission & short_permission
                if long_permission == 0:
                    output[:, trading_signal] = np.where(output[:, flow] == 1, 0, output[:, trading_signal])
                if short_permission == 0:
                    output[:, trading_signal] = np.where(output[:, flow] == 0, 0, output[:, trading_signal])
            else:
                for i in range(1, output.shape[0]):
                    if input_[i, certainty] > certainty_threshold: # Certainty is over the required threshold
                        # True
                        output[i, trading_signal] = 1
                    else:
                        # False
                        output[i, trading_signal] = 0

                # Correct long_permission & short_permission
                if long_permission == 0:
                    output[:, trading_signal] = np.where(input_[:, kind] == 1, 0, output[:, trading_signal])
                if short_permission == 0:
                    output[:, trading_signal] = np.where(input_[:, kind] == 0, 0, output[:, trading_signal])

            # PASSIVE TRADING
            if follow_flow == 1:
                output[:, passive_trading] = np.where(
                    np.logical_and(
                        output[:, trading_signal] == 1, # There was a trading signal
                        input_[:, certainty] <= certainty_threshold # Not enough certainty
                    ),
                    1, # True
                    0 # False
                )
            else:
                output[:, passive_trading] = 0

            """
            Exposure:
            - Exposure will be 0 if the position is inactive
            - If not, exposure will depend on the certainty of the trade

            Leverage:
            - Leverage will only be implemented in cases where certainty is >= 90%
            - If not, it will be 1
            """
            # EXPOSURE
            max_exposures = np.ones(input_[:, certainty].shape, dtype=float)
            min_exposures = 0.5 * max_exposures
            raw_exposures = np.log(100 * input_[:, certainty] - 49) / np.log(90 - 49) + 0.1
            exposures = np.maximum(min_exposures, np.minimum(raw_exposures, max_exposures))

            output[:, exposure] = np.where(
                output[:, trading_signal] == 0, 
                # Position inactive -> Exposure is 0
                0.0, 
                # Position active
                exposures
            )
            
            # EXPOSURE ADJUSTMENT
            output[0, exposure_adj] = 0.0
            output[1:, exposure_adj] = output[1:, exposure] - output[:-1, exposure]

            # LEVERAGE
            output[:, leverage] = np.where(input_[:, certainty] > 0.9, max_leverage, 1.0)

            """
            Trading Fees
            - Depends on:
                - If a new position was opened 
                - If a previous position needed to be closed (change of flow)
                - If new position was forced to closed (i.e.: if a tp or sl was triggered)
            - Trading fees are adjusted by exposure and leverage
            """
            # Define staring value
            if output[0, trading_signal] == 1: 
                # Active position -> New position was opened
                output[0, fees] = trading_fees * output[0, exposure] * output[0, leverage]
                if output[0, tp_triggered] == 1 or output[0, sl_triggered] == 1:
                    # The new position that was opened, was forced to close
                    output[0, fees] *= 2
            else:
                # No new position was opened
                output[0, fees] = 0

            # Define the rest
            for i in range(1, output.shape[0]):
                if output[i, trading_signal] == 1: 
                    # There will be an open position in this period
                    if output[i-1, trading_signal] == 1:
                        # There was already an open position from last period
                        if follow_flow == 1:
                            if output[i, flow] == output[i-1, flow]:
                                # Flow is kept, therefore only need to adjust exposure
                                output[i, fees] = trading_fees * np.abs(output[i, exposure_adj]) * output[i, leverage]
                            else:
                                # Flow is not kept, therefore a previous position was closed and a new position was opened
                                output[i, fees] = trading_fees * output[i-1, exposure] * output[i-1, leverage] # Previous position closed
                                output[i, fees] += trading_fees * output[i, exposure] * output[i, leverage] # New position opened
                        else:
                            if input_[i, kind] == input_[i-1, kind]:
                                # Kind is kept, therefore only need to adjust exposure
                                output[i, fees] = trading_fees * np.abs(output[i, exposure_adj]) * output[i, leverage]
                            else:
                                # Kind is not kept, therefore a previous position was closed and a new position was opened
                                output[i, fees] = trading_fees * output[i-1, exposure] * output[i-1, leverage] # Previous position closed
                                output[i, fees] += trading_fees * output[i, exposure] * output[i, leverage] # New position opened
                    else:
                        # There was not an open position from last period (no position needed to be closed)
                        output[i, fees] = trading_fees * output[i, exposure] * output[i, leverage] # New position opened
                    # Forced to close
                    if output[i, tp_triggered] == 1 or output[i, sl_triggered] == 1:
                        output[i, fees] += trading_fees * output[i, exposure] * output[i, leverage] # New position was forced to close
                else:
                    # There will not be an active position in this period
                    if output[i-1, trading_signal] == 1:
                        # There was an active position before, which needs to be closed
                        output[i, fees] = trading_fees * output[i-1, exposure] * output[i-1, leverage]
                    else:
                        # There was no position to be closed
                        output[i, fees] = 0

            """
            Trading Return, Real Trading Return, Total Cumulative Returns
            """
            # TRADING RETURN
            for i in range(output.shape[0]):
                if output[i, trading_signal] == 1 and output[i, tp_triggered] == 0 and output[i, sl_triggered] == 0:
                    # Active position, no TP triggered, no SL triggered:
                    #   - trading_return = ((close - open) / open) * exposure * leverage
                    output[i, trading_return] = ((input_[i, price] - input_[i, open]) / input_[i, open]) * output[i, exposure] * output[i, leverage]

                elif output[i, trading_signal] == 1 and output[i, tp_triggered] == 1 and output[i, sl_triggered] == 0:
                    # Active position, TP triggered, no SL triggered:
                    #   - trading_return = ((tp_price - open) / open) * exposure * leverage
                    output[i, trading_return] = ((output[i, tp_price] - input_[i, open]) / input_[i, open]) * output[i, exposure] * output[i, leverage]

                elif output[i, trading_signal] == 1 and output[i, tp_triggered] == 0 and output[i, sl_triggered] == 1:
                    # Active position, no TP triggered, SL triggered:
                    #   - trading_return = ((sl_price - open) / open) * exposure * leverage
                    output[i, trading_return] = ((output[i, sl_price] - input_[i, open]) / input_[i, open]) * output[i, exposure] * output[i, leverage]

                else:
                    output[i, trading_return] = 0
            """
            output[:, trading_return] = np.where(
                # Active position, no TP triggered, no SL triggered:
                #   - trading_return = ((close - open) / open) * exposure * leverage
                np.logical_and(
                    output[:, trading_signal] == 1,
                    output[:, tp_triggered] == 0,
                    output[:, sl_triggered] == 0
                ),
                ((input_[:, price] - input_[:, open]) / input_[:, open]) * output[:, exposure] * output[:, leverage],
                np.where(
                    # Active position, TP triggered, no SL triggered:
                    #   - trading_return = ((tp_price - open) / open) * exposure * leverage
                    np.logical_and(
                        output[:, trading_signal] == 1,
                        output[:, tp_triggered] == 1,
                        output[:, sl_triggered] == 0
                    ),
                    ((output[:, tp_price] - input_[:, open]) / input_[:, open]) * output[:, exposure] * output[:, leverage],
                    np.where(
                        # Active position, no TP triggered, SL triggered:
                        #   - trading_return = ((sl_price - open) / open) * exposure * leverage
                        np.logical_and(
                            output[:, trading_signal] == 1,
                            output[:, tp_triggered] == 0,
                            output[:, sl_triggered] == 1
                        ),
                        ((output[:, sl_price] - input_[:, open]) / input_[:, open]) * output[:, exposure] * output[:, leverage],
                        0
                    )
                )
            )
            """
            # Reverse 'short' trading_returns
            if follow_flow == 1:
                output[:, trading_return] = np.where(
                    output[:, flow] == 1, # Flow: long
                    output[:, trading_return], # unreversed trading_return
                    -1 * output[:, trading_return] # reversed trading_return
                )
            else:
                output[:, trading_return] = np.where(
                    input_[:, kind] == 1, # Kind: long
                    output[:, trading_return], # unreversed trading_return
                    -1 * output[:, trading_return] # reversed trading_return
                )

            # REAL TRADING RETURN
            output[:, real_trading_return] = output[:, trading_return] + output[:, fees]

            # TOTAL CUMULATIVE RETURNS
            output[:, total_cum_returns] = np.cumprod(1 + output[:, real_trading_return]) - 1

            """
            Trading Accuracy, ML Accuracy & Trading Class
            """
            # TRADING ACCURACY
            output[:, trading_accuracy] = np.where(
                output[:, real_trading_return] >= 0, # Positive real trading returns
                1, # True
                0 # False
            )

            # ML ACCURACY
            output[:, ml_accuracy] = np.where(
                np.logical_or(
                    np.logical_and(
                        input_[:, kind] == 1, # Kind: long
                        input_[:, real_return] >= np.abs(output[:, fees]) # Coin returns are bigger than or equal to the fees
                    ),
                    np.logical_and(
                        input_[:, kind] == 0, # Kind: short 
                        input_[:, real_return] <= np.abs(output[:, fees]) # Coin returns are lower than or equal to the fees
                    )
                ),
                1,
                0
            )

            # TRADE CLASS
            """
            'trade_class': {
                0: 'long_point',
                1: 'long_fail',
                2: 'short_point',
                3: 'short_fail',
                4: 'long_missed',
                5: 'short_missed'
            }
            """
            output[:, trade_class] = np.where(
                output[:, trading_signal] == 1,
                # Active position
                np.where(
                    input_[:, kind] == 1,
                    # Kind: long
                    np.where(
                        # Positive return
                        input_[:, real_return] > 0,
                        0, # long_point
                        1 # long_fail
                    ),
                    # Kind: short
                    np.where(
                        # Negative return
                        input_[:, real_return] < 0,
                        2, # short_point
                        3 # short_fail
                    )
                ),
                # No position
                np.where(
                    input_[:, kind] == 1,
                    # Kind: long,
                    4, # long_missed
                    5 # short_missed
                )
            )

            return output

        input_cols = [
            'open',
            'price',
            'high',
            'low',
            'real_return',
            'certainty',
            'kind',
        ]
        replace_input = {
            'long': 1,
            'short': 0
        }
        
        output_cols = [
            'flow',

            'tp_price',
            'sl_price',
            'tp_triggered',
            'sl_triggered',
            
            'trading_signal',
            'passive_trading',            

            'exposure',
            'exposure_adj',
            'leverage',
            
            'fees',
            'trading_return',
            'real_trading_return',
            'total_cum_returns',

            'trading_accuracy',
            'ml_accuracy',
            'trade_class'
        ]
        replace_output = {
            'flow': {
                1: 'long',
                0: 'short'
            },
            'tp_triggered': {
                1: True,
                0: False
            },
            'sl_triggered': {
                1: True,
                0: False
            },
            'trading_signal': {
                1: True,
                0: False
            },
            'passive_trading': {
                1: True,
                0: False
            },            
            'trading_accuracy': {
                1: True,
                0: False
            },
            'ml_accuracy': {
                1: True,
                0: False
            },
            'trade_class': {
                0: 'long_point',
                1: 'long_fail',
                2: 'short_point',
                3: 'short_fail',
                4: 'long_missed',
                5: 'short_missed'
            }
        }

        # Apply accelerator
        follow_flow = 1 if self.trading_parameters['follow_flow'] else 0
        long_permission = 1 if self.trading_parameters['long_permission'] else 0
        short_permission = 1 if self.trading_parameters['short_permission'] else 0
        
        self[output_cols] = accelerator(
            self[input_cols].replace(replace_input).astype(float).values,
            follow_flow=follow_flow, # follow_flow
            
            ltp=self.trading_parameters['tp'][0],
            stp=self.trading_parameters['tp'][1],
            lsl=self.trading_parameters['sl'][0],
            ssl=self.trading_parameters['sl'][1],

            certainty_threshold=self.trading_parameters['certainty_threshold'],
            long_permission=long_permission, # long_permission,
            short_permission=short_permission, # short_permission,
            max_leverage=self.trading_parameters['max_leverage'],
            trading_fees=self.trading_fees
        )
        
        # Replace numpy values
        for col in replace_output.keys():
            self[col] = self[col].replace(replace_output[col])
        
        # self[input_cols + output_cols].to_excel('test.xlsx')

        cols_diff = list(set(self.columns).symmetric_difference(set(self.main_cols)))
        if len(cols_diff) > 0:
            # assert len(cols_diff) == 0, f"[WARNING] There are missing columns in {self.save_name}: {cols_diff}\n\n"
            drop_cols = [c for c in self.columns if c not in self.main_cols]
            LOGGER.warning('Droping the following columns:\n%s\n', pformat(drop_cols))

            self.drop(columns=drop_cols, inplace=True)

    def update(
        self,
        new_table_input: pd.DataFrame,
        update_performance: bool = True,
        smooth_returns: bool = False,
        return_weight: float = None,
        debug: bool = False
    ) -> None:
        # Save paramters
        general_attrs = deepcopy(self.general_attrs)
        residuals_attrs = deepcopy(self.residuals_attrs)

        # Instanciate new table
        table_input = pd.concat([
            self[new_table_input.columns.tolist()],
            new_table_input
        ], axis=0)

        new_table = TradingTable(
            table_input,
            model_id=self.model_id_,
            coin_name=self.coin_name_,
            intervals=self.intervals_,

            algorithm=self.algorithm,
            method=self.method,
            pca=self.pca,
            trading_parameters=self.trading_parameters,

            initialize=True,
            table_name=self.table_name,
            load_table=False,
            debug=debug
        )

        # Set residuals attrs
        for attr_name, attr_value in residuals_attrs.items():
            setattr(new_table, attr_name, attr_value)

        # Set general attrs
        for attr_name, attr_value in general_attrs.items():
            setattr(self, attr_name, attr_value)
        
        new_table.complete_table(
            find_best_dist=False,
            dummy_proof=False
        )

        new_table.measure_trading_performance(
            smooth_returns=smooth_returns,
            return_weight=return_weight,
            debug=debug
        )
        
        return new_table

        if debug:
            print(f'self (before adding new table):\n {self.tail(10)}\n\n'
                  f'new_table:\n {new_table.tail(10)}\n\n')

        self: TradingTable = self.combine_first(new_table)

        # Set general attrs
        for attr_name, attr_value in general_attrs.items():
            setattr(self, attr_name, attr_value)

        # Set residuals attrs
        for attr_name, attr_value in residuals_attrs.items():
            setattr(self, attr_name, attr_value)

        if debug:
            print(f'self (after adding new table):\n {self.tail(10)}\n\n')
        
        # Update performance
        if update_performance:
            self.measure_trading_performance(
                smooth_returns=smooth_returns,
                return_weight=return_weight,
                debug=debug
            )
        
        return self

    def fit_distribution(
        self,
        remove_outliers: bool = True, 
        dist_name: str = None,
        plot_=False
    ) -> None:
        def make_pdf(size=10000):
            """Generate distributions's Probability Distribution Function """
            # Get sane start and end points of distribution
            if self.arg:
                start = self.resid_dist.ppf(0.01, *self.arg, loc=self.loc_, scale=self.scale)
                end = self.resid_dist.ppf(0.99, *self.arg, loc=self.loc_, scale=self.scale)
            else:
                start = self.resid_dist.ppf(0.005, loc=self.loc_, scale=self.scale)
                end = self.resid_dist.ppf(0.995, loc=self.loc_, scale=self.scale)

            # Build PDF and turn into pandas Series
            x = np.linspace(start, end, size)
            y = self.resid_dist.pdf(x, loc=self.loc_, scale=self.scale, *self.arg)

            pdf = pd.Series(y, x)

            return pdf

        def display(pdf: pd.Series):
            fig = go.Figure(
                data=[
                    go.Histogram(
                        name='Model Residuals',
                        x=data,
                        opacity=0.5
                    ),
                    go.Scatter(
                        name=f'Fitted Distribution: {dist_name}',
                        x=pdf.index,
                        y=pdf,
                    )
                ],
                layout=go.Layout(
                    title='Residuals vs. Fitted Distribution',
                )
            )

            fig.show()
            """
            # Bins
            h = (2 * (data.quantile(0.75) - data.quantile(0.25)) / len(data.unique()) ** (1 / 3))
            bins = range(int(data.min()), int(data.max()) + 1, int(h) + 1)

            # Make PDF with best params
            pdf = make_pdf(best_dist, best_param)

            fig = plt.figure(figsize=(12, 8))
            ax = pdf.plot(lw=2, label=f'Simulated Data {best_dist_name}', legend=True)
            data.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Real Data', legend=True, ax=ax)

            param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
            param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_param)])
            dist_str = 'best fit distribution: {}({})'.format(best_dist_name, param_str)

            if title is not None:
                ax.set_title(f'{title}\n {dist_str}')
            plt.show()
            """

        """
        Fit multiple distributions to our data, and use the Kolmogorov-Smirnov (KS) test to determine the
        best fit.
        """
        data = self[['return_residual']].copy()
        data['return_residual'] = pd.to_numeric(data['return_residual'], errors='coerce')
        data = data[['return_residual']].dropna()        
        
        # Remove Outliers
        if remove_outliers:
            scaler = StandardScaler(with_mean=True, with_std=True)
            data['standard_obs'] = scaler.fit_transform(data)

            data = data.loc[
                (data['standard_obs'] > -2.1) &
                (data['standard_obs'] < 2.1)
            ]['return_residual']

        if len(data) >= 30:
            if dist_name is None:
                # Find best fit
                dist_names = ['gamma', 'norm', 'expon', 'lognorm', 'beta', 'uniform']
                results = []

                for name in dist_names:
                    dist = getattr(st, name)
                    param = dist.fit(data)

                    # Apply the KS test:
                    D, p = st.kstest(data, name, args=param)

                    results.append((dist, param, name, p))

                self.resid_dist, self.resid_params, self.dist_name, test_value = (
                    max(results, key=lambda item: item[-1])
                )
            else:
                self.resid_dist = getattr(st, dist_name)
                self.resid_params = self.resid_dist.fit(data)
                self.dist_name = dist_name

            self.arg, self.loc_, self.scale = self.resid_params[:-2], self.resid_params[-2], self.resid_params[-1]

            # Display
            if plot_:
                # Make PDF with best params
                pdf = make_pdf()
                display(pdf=pdf)
                
        else:
            self.resid_dist, self.resid_params, self.dist_name = None, None, None
            self.arg, self.loc_, self.scale = None, None, None 

    def evaluate_residuals(
        self,
        update_resid_dist: bool = False,
        dist_name: str = None,
        update_pred_resid_correl: bool = False,
        check_resid_mean: bool = False,
        check_homoscedasticity: bool = False, 
        features_df: pd.DataFrame = None,
        check_resid_autocorrel: bool = False,
        debug: bool = False
    ) -> None:
        """
        Residuals should be Stationary (Constant Mean, Constant Variance & No Auto-Correlations)
        """
        self.dropna(subset=['return_residual'], inplace=True)

        if update_resid_dist:
            if dist_name is None:
                self.fit_distribution(
                    remove_outliers=True
                )
            else:
                self.fit_distribution(
                    remove_outliers=True,
                    dist_name=dist_name
                )
        
        if update_pred_resid_correl:
            # Prediction - Residuals Correlation:
            self.pred_resid_correl = self['return_forecast'].corr(self['return_residual'])
            if debug:
                print(f'pred_resid_correl: {round(self.pred_resid_correl*100, 2)}')

        # p_value = adfuller(self.residuals_df['return_residual'])[1]
        # if p_value > 0.1: Residuals are NOT Stationary. --- else: Residuals are Stationary.

        if check_resid_mean:
            ttest_ind_p_value = st.ttest_ind(
                a=self[['return_residual']].iloc[:-int(len(self) * 0.5)],
                b=self[['return_residual']].iloc[-int(len(self) * 0.5):],
            )[1][0]
            if ttest_ind_p_value < 0.05:
                if debug:
                    print(f"ttest_ind p_value: {ttest_ind_p_value}\n"
                          f"Residuals don't have constant mean.\n\n")

                y = self['return_residual']
                x = [t for t in range(len(self))]
                x = sm.add_constant(x)
                results = sm.OLS(y, x).fit()

                self.resid_alpha = results.params[0]
                self.resid_beta = results.params[1]
            else:
                if debug:
                    print(f"ttest_ind p_value: {ttest_ind_p_value}\n"
                          f"Residuals have constant mean (or at least we can't disprove it!).\n\n")

        if check_homoscedasticity and features_df is not None:
            bp_data = features_df.loc[features_df.index.isin(self.index)]

            het_breuschpagan_p_value = het_breuschpagan(self[['return_residual']], bp_data)[-1]
            if het_breuschpagan_p_value < 0.05:
                self.heteroskedasticity = True
                if debug:
                    print(f'f_test het_breuschpagan_p_value: {het_breuschpagan_p_value}\n'
                          f'Model is Heteroskedastic (residual´s variance is not constant).\n\n')
            else:
                if debug:
                    print(f"f_test het_breuschpagan_p_value: {het_breuschpagan_p_value}\n"
                          f"Model is not Heteroskedastic (residual´s variance is constant) (or at least we can't disprove it!).\n\n")

                self.heteroskedasticity = False

        if check_resid_autocorrel:
            acorr_ljungbox_p_values = acorr_ljungbox(self['return_residual'], lags=10)['lb_pvalue'].values
            if len([pv for pv in acorr_ljungbox_p_values if pv < 0.05]) == 0:
                self.resid_auto_correlations = None
                if debug:
                    print(f'acorr_ljungbox p_values: {acorr_ljungbox_p_values}\n'
                          f'Residuals are not Correlated.\n\n')
            else:
                self.resid_auto_correlations = acf(self['return_residual'], nlags=10, alpha=0.05)[0]
                if debug:
                    print(f'acorr_ljungbox p_values: {acorr_ljungbox_p_values}\n'
                          f'resid_auto_correlations: {self.resid_auto_correlations}\n'
                          f'Residuals have Auto-Correlations.\n\n')
    
    def save_mock_asset(
        self,
        asset: pd.DataFrame | dict = None,
        asset_name: str = None
    ) -> None:
        # print(f'Saving {asset_name} - [shape: {asset.shape}]')

        # Define base_path
        base_path = f"{Params.bucket}/mock/trading/trading_table/{self.intervals_}"

        # Define save_path
        if asset_name == 'test_trading_df':
            save_path = f"{base_path}/test_trading_df.parquet"
        elif asset_name == 'test_trading_df_attr':
            save_path = f"{base_path}/test_trading_df_attr.pickle"
        elif asset_name == 'expected_TT_output':
            save_path = f"{base_path}/expected_TT_output.parquet"
        elif asset_name == 'performance_attrs':
            save_path = f"{base_path}/performance_attrs.pickle"
        elif asset_name == 'metrics':
            save_path = f"{base_path}/metrics.pickle"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Save asset to S3
        write_to_s3(asset=asset, path=save_path, overwrite=True)
    
    def load_mock_asset(
        self,
        asset_name: str,
        re_create: bool = False
    ) -> pd.DataFrame | dict:
        # Define re_create_base_path
        if re_create:
            prod_model_id: str = load_from_s3(
                f"{Params.bucket}/modeling/model_registry/{self.intervals_}/model_registry.json"
            )['production'][0][0]

            LOGGER.info("Re-creating mocked TradingTable from model_id: %s", prod_model_id)

            re_create_base_path = f"{Params.bucket}/trading/trading_table/{self.intervals_}/{prod_model_id}"
        else:
            prod_model_id = None
            re_create_base_path = None

        # Define base_path
        base_path = f"{Params.bucket}/mock/trading/trading_table/{self.intervals_}"

        # Define load_path
        if asset_name == 'test_trading_df':
            if re_create:
                load_path = f"{re_create_base_path}/{prod_model_id}_test_trading_df.parquet"
            else:
                load_path = f"{base_path}/test_trading_df.parquet"
        elif asset_name == 'test_trading_df_attr':
            if re_create:
                load_path = f"{re_create_base_path}/{prod_model_id}_test_trading_df_attr.pickle"
            else:
                load_path = f"{base_path}/test_trading_df_attr.pickle"
        elif asset_name == 'expected_TT_output':
            load_path = f"{base_path}/expected_TT_output.parquet"
        elif asset_name == 'performance_attrs':
            load_path = f"{base_path}/performance_attrs.pickle"
        elif asset_name == 'metrics':
            load_path = f"{base_path}/metrics.pickle"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Load asset from S3
        asset = load_from_s3(path=load_path, ignore_checks=True)

        if re_create:
            if asset_name == 'test_trading_df':
                # Filter required columns
                asset = asset[[
                    'return_forecast',
                    'real_return',
                    'open',
                    'high',
                    'low',
                    'price'
                ]]
            elif asset_name == 'test_trading_df_attr':
                # Rename keys
                asset['model_id'] = asset.pop('model_id_')
                asset['coin_name'] = asset.pop('coin_name_')
                asset['intervals'] = asset.pop('intervals_')

                # Drop unnecessary keys
                keep_keys = [
                    'model_id', 'coin_name', 'intervals',
                    'algorithm', 'method', 'pca', 'trading_parameters'
                ]
                asset: dict = {k: v for k, v in asset.items() if k in keep_keys}
        
        # print(f'Loaded {asset_name} - [shape: {asset.shape}]')

        return asset

    def save(
        self,
        debug: bool = False
    ) -> None:
        # Define base_path
        # self.s3_base_path = f"{Params.bucket}/trading/trading_table/{self.intervals_}/{self.model_id_}"
        
        # Save parquet files
        write_to_s3(
            asset=self,
            path=f"{self.save_path}/{self.table_name}.parquet"
        )

        # Define attributes to save
        trading_table_attr = {key: value for key, value in self.__dict__.items() if key in self.save_attr_list}

        # Save pickle attrs
        write_to_s3(
            asset=trading_table_attr,
            path=f"{self.save_path}/{self.table_name}_attr.pickle"
        )
        
        if debug:
            print(f'Saved Attributes: {[k for k in trading_table_attr.keys()]}\n')
            pprint(trading_table_attr)
            print(f'\n\nDF Head: \n{self.head()}\n\n')

    def load(
        self,
        debug: bool = False
    ) -> None:
        # Define base path
        # self.s3_base_path = f"{Params.bucket}/trading/trading_table/{self.intervals_}/{self.model_id_}"
        
        try:
            # Load parquet files
            loaded_parquet = load_from_s3(path=f"{self.save_path}/{self.table_name}.parquet")

            # Instanciate Tradingtable
            self.__init__(
                loaded_parquet,
                model_id=self.model_id_,
                coin_name=self.coin_name_,
                intervals=self.intervals_,

                algorithm=self.algorithm,
                method=self.method,
                pca=self.pca,
                trading_parameters=self.trading_parameters,

                initialize=True,
                table_name=self.table_name,
                mock=self.mock,
                load_table=False,
                debug=debug
            )

            if debug:
                print(f'\n\nDF Head: \n{self.head()}\n\n')
        except Exception as e:
            LOGGER.warning(
                "Unable to load Trading Table's parquet file (%s: %s).\n"
                "Exception: %s\n\n",
                self.table_name, self.intervals_, e
            )
        
        try:
            # Load pickle files
            trading_table_attr: dict = load_from_s3(path=f"{self.save_path}/{self.table_name}_attr.pickle")

            # Set pickled attrs
            for attr_key, attr_value in trading_table_attr.items():
                if attr_key in self.save_attr_list:
                    setattr(self, attr_key, attr_value)
                    if debug:
                        print(f'Setting {attr_key} as {attr_value}\n')
        except Exception as e:
            LOGGER.warning(
                "Unable to load Trading Table's attributes (%s: %s).\n"
                "Exception: %s\n\n",
                self.table_name, self.intervals_, e
            )
        
        if debug:
            print(f'loaded trading_metric ({self.table_name}): {self.trading_metric}')

    def show_attrs(
        self,
        general_attrs: bool = True,
        residuals_attrs: bool = True,
        performance_attrs: bool = True
    ) -> str:
        output = 'TradingTable:'

        if general_attrs:
            # Define general attributes
            gen_attrs = {
                'Table Name': self.table_name,
                'Model ID': self.model_id_,
                'Coin Name': self.coin_name_,
                'Stable Coin': self.stable_coin_,
                'Intervals': self.intervals_,
                'Trading Fees': self.trading_fees,

                'Is dummy': self.is_dummy,
                'Algorithm': self.algorithm,
                'Method': self.method,
                'PCA': self.pca,
                'Trading Parameters': self.trading_parameters
            }

            output += f'General Attributes:\n{pformat(gen_attrs)}\n\n'

        if residuals_attrs:
            # Define residuals attributes
            resid_attrs = {
                'Residuals Distribution': self.resid_dist,
                'Residuals Parameters': self.resid_params,
                'Distribution Name': self.dist_name,

                'Arg': self.arg,
                'Loc': self.loc_,
                'Scale': self.scale,

                'Predictions-Residuals Correlation': self.pred_resid_correl,
                'Residuals Alpha': self.resid_alpha,
                'Residuals Beta': self.resid_beta,
                'Heteroskedasticity': self.heteroskedasticity,
                'Residuals Auto-Correlations': self.resid_auto_correlations
            }

            output += f'Residuals Attributes:\n{pformat(resid_attrs)}\n\n'

        if performance_attrs:
            def prepare_metric(metric: float):
                if metric is None:
                    return metric
                return f"{round(metric * 100, 1)} %"
            
            # Define performance attrs
            perf_attrs = {
                "Trading Metric": self.trading_metric,
                "Tuning Metric": self.tuning_metric,
                "Estimated Monthly Return": prepare_metric(self.est_monthly_ret),
                "Estimated Standard Error": prepare_metric(self.est_stand_error),
                "Estimated Sharpe Ratio": prepare_metric(self.est_sharpe_ratio),
                "ML Accuracy": prepare_metric(self.ml_accuracy),
                "Weighted ML Accuracy": prepare_metric(self.weighted_ml_accuracy),
                "Trading Accuracy": prepare_metric(self.trading_accuracy),
                "Weighted Trading Accuracy": prepare_metric(self.weighted_trading_accuracy),
                "Return P-Value": prepare_metric(self.ret_pvalue),
                "Cumulative Return": prepare_metric(self.cum_ret),
                "N-Trade Percentage": prepare_metric(self.n_trade_perc),
                "WAPE": prepare_metric(self.wape),
                "Long Success Rate": prepare_metric(self.lsr),
                "Short Success Rate": prepare_metric(self.ssr),
                "Long Rate": prepare_metric(self.long_rate)
            }

            output += f'Performance Attributes:\n{pformat(perf_attrs)}\n\n'

        return output
