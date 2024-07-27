from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.data_warehousing.data_warehouse_manager import DataWarehouseManager
from PyTradeX.data_warehousing.trading_analyst import TradingAnalyst
from PyTradeX.modeling.model import Model
from PyTradeX.modeling.model_registry import ModelRegistry
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3

import scipy.stats as st
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
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


class Drift:

    load_parquet = [
        'simulated_returns'
    ]
    load_pickle = [
        'drift_analysis'
    ]

    def __init__(
        self,
        intervals: str,
        expected_monthly_avg_ret: float,
        expected_acc: float,
        avg_ret_certainty_threshold: float = 0.05,
        acc_certainty_threshold: float = 0.05,
        cum_ret_certainty_threshold: float = 0.05,
        ml_params: dict = Params.ml_params.copy(),
        TA_start_date: datetime.date = Params.data_params.get('TA_start_date'),
        debug: bool = False
    ) -> None:
        # Retrieve Performance Expectancies
        self.expected_monthly_avg_ret: float = expected_monthly_avg_ret
        self.expected_acc: float = expected_acc

        # Retrieve Certainty Thresholds
        self.avg_ret_certainty_threshold: float = avg_ret_certainty_threshold
        self.acc_certainty_threshold: float = acc_certainty_threshold

        if cum_ret_certainty_threshold == 0.05:
            self.cum_ret_certainty_threshold = 'Q-5%'
        elif cum_ret_certainty_threshold == 0.01:
            self.cum_ret_certainty_threshold = 'Q-1%'
        else:
            self.cum_ret_certainty_threshold = None

        # Define self.intervals & periods_per_month
        self.intervals = intervals
        periods_per_month = {
            '30min': 1460,
            '60min': 730
        }[self.intervals]
        
        # Calculate self.expected_avg_period_ret
        #   - (1 + self.expected_monthly_avg_ret) = (1 + self.expected_avg_period_ret) ** periods_per_month
        self.expected_avg_period_ret = ((1 + self.expected_monthly_avg_ret) ** (1/periods_per_month)) - 1

        if debug:
            print(f'self.expected_avg_period_ret: {self.expected_avg_period_ret}\n\n')

        # Instanciate ModelRegistry
        self.model_registry = ModelRegistry(
            n_candidates=ml_params.get('n_candidates'),
            intervals=self.intervals
        )

        # Find Champion
        self.champion = self.model_registry.load_prod_model(light=False)

        # Instanciate DataWarehouseManager & TradingAnalyst
        self.DWM = DataWarehouseManager(intervals=self.intervals)
        self.TA = TradingAnalyst(
            intervals=self.intervals,
            TA_start_date=TA_start_date
        )

        # Build trading_returns
        self.TA.build_ml_trading_returns(
            account_ids=None,
            model_ids=[self.champion.model_id],
            debug=debug
        )

        self.model_trading_returns = self.TA.ml_trading_returns.copy(deep=True)

        # Define self.drift_analysis & self.simulated_returns
        self.drift_analysis: dict = {}
        self.simulated_returns: pd.DataFrame = None
        
        # Define base_path
        self.s3_base_path = f"{Params.bucket}/trading/drift/{self.intervals}"

        self.load()

        if debug:
            def _print(attr_val):
                if isinstance(attr_val, pd.DataFrame):
                    return attr_val.tail()
                elif isinstance(attr_val, (
                    str, dict, Model, ModelRegistry, 
                    DataWarehouseManager, TradingAnalyst
                )):
                    return attr_val
                elif attr_val is None:
                    return None
                return f'{round(attr_val*100, 3)} %'
            print({
                k: _print(v) for k, v in self.__dict__.items()
            })

    def set_triggers(self):
        # Update trading round trigger
        s3_trading_round_trigger_path = f"{Params.bucket}/utils/triggers/{self.intervals}/trading_round_trigger.json"
        write_to_s3(
            asset={'run': False},
            path=s3_trading_round_trigger_path
        )

        # Update model updating trigger
        s3_model_updating_trigger_path = f"{Params.bucket}/utils/triggers/{self.intervals}/model_updating_trigger.json"
        write_to_s3(
            asset={'run': False},
            path=s3_model_updating_trigger_path
        )

        # Update model building trigger
        s3_model_building_trigger_path = f"{Params.bucket}/utils/triggers/{self.intervals}/model_building_trigger.json"
        write_to_s3(
            asset={'run': True},
            path=s3_model_building_trigger_path
        )

    def analyze_avg_ret_drift(
        self, 
        days_to_eval: int = None,
        debug: bool = False
    ):
        """
        Test de Hipótesis:
        H0: los retornos promedio del champion en producción (r) son >= que los retornos promedios esperados.
        H1: los retornos promedio del champion en producción (r) son < que los retornos promedio esperados.

        R ~ N(r, sigma2/n)
        T = ((R-mu) / (S/np.sqrt(n))) ~ T-Student(n-1)
            - r: promedio poblacional de los retornos del champion en producción
            - R: promedio muestral de los retornos del champion en producción
            - mu: promedio esperado de los retornos del champion en producción
            - S: desvío estándar (sigma) muestral de los retornos del champion en producción
            - n: cantidad de observaciones

        Para calcular p_value:
            - t_score = (R - mu) / (S/np.sqrt(n)))
                - mu = self.expected_avg_period_ret
                - R = np.mean(self.model_trading_returns['real_trading_return'])
                - S = np.std(self.model_trading_returns['real_trading_return'], ddof=1)
                - n = self.model_trading_returns.shape[0]
            - p_val = 1 - st.t.sf(t_score, n-1)

        Nota: IC[1-alpha] = R+-t_alpha/2*(S/np.sqrt(n))
        """
        # Find periods_to_analyze
        daily_periods = {
            '30min': 48,
            '60min': 24
        }[self.intervals]
        periods_to_analyze = daily_periods * days_to_eval if days_to_eval is not None else self.model_trading_returns.shape[0]

        # Filter self.model_trading_returns
        analyze_returns = self.model_trading_returns.iloc[-periods_to_analyze:]

        if analyze_returns.shape[0] >= daily_periods:
            # Find P-Value
            t_score = (
                (analyze_returns['real_trading_return'].mean() - self.expected_avg_period_ret) / 
                (np.std(analyze_returns['real_trading_return'], ddof=1) / np.sqrt(analyze_returns.shape[0]))
            )
            p_val = 1 - st.t.sf(t_score, analyze_returns.shape[0]-1)

            if debug:
                print(f'avg_ret_p_val: {p_val}\n\n')

            # Update drift_analysis
            self.drift_analysis['avg_ret_p_val'] = p_val
        else:
            self.drift_analysis['avg_ret_p_val'] = None

        if (
            self.drift_analysis['avg_ret_p_val'] is not None and 
            self.drift_analysis['avg_ret_p_val'] < self.avg_ret_certainty_threshold
        ):
            LOGGER.warning(
                'Avg Return Drift was detected in model %s.\n'
                'Trading Rounds will cease and Model Building will be triggered.\n'
                'drift_analysis:\n%s\n',
                self.champion.model_id, pformat(self.drift_analysis)
            )
            
            self.set_triggers()

    def analyze_acc_drift(
        self,
        days_to_eval: int = None,
        debug: bool = False
    ):
        """
        Test de Hipótesis:
        H0: el trading_accuracy del champion en producción (a) es >= que el trading_accuracy esperado.
        H1: el trading_accuracy del champion en producción (a) es < que el trading_accuracy esperado.

        ai ~ B(n, a)
        A ~ N(a, sigma2/n)
        T = ((A-mu) / (S/np.sqrt(n))) ~ T-Student(n-1)
            - a: trading_accuracy poblacional del champion en producción
            - A: trading_accuracy muestral del champion en producción
            - mu: trading_accuracy esperado del champion en producción
            - S: desvío estándar (sigma) muestral del trading_accuracy del champion en producción
            - n: cantidad de observaciones

        Para calcular p_value:
            - t_score = (A - mu) / (S/np.sqrt(n)))
                - mu = self.expected_acc
                - A = self.model_trading_returns['trading_accuracy'].mean()
                - S = np.std(self.model_trading_returns['trading_accuracy'], ddof=1)
                - n = self.model_trading_returns.shape[0]
            - p_val = 1 - st.t.sf(t_score, n-1)

        Nota: IC[1-alpha] = R+-t_alpha/2*(S/np.sqrt(n))
        """
        # Find Returns to Analyze
        daily_periods = {
            '30min': 48,
            '60min': 24
        }[self.intervals]
        periods_to_analyze = daily_periods * days_to_eval if days_to_eval is not None else self.model_trading_returns.shape[0]

        analyze_returns = self.model_trading_returns.iloc[-periods_to_analyze:]
        
        if analyze_returns.shape[0] >= daily_periods:
            # Find P-Value
            t_score = (
                (analyze_returns['trading_accuracy'].mean() - self.expected_acc) / 
                (np.std(analyze_returns['trading_accuracy'], ddof=1) / np.sqrt(analyze_returns.shape[0]))
            )
            p_val = 1 - st.t.sf(t_score, analyze_returns.shape[0]-1)

            if debug:
                print(f'Trading accuracy: {analyze_returns["trading_accuracy"].mean()}\n'
                      f'acc_p_val: {p_val}\n\n')

            # Update drift_analysis
            self.drift_analysis['acc_p_val'] = p_val
        else:
            self.drift_analysis['acc_p_val'] = None

        if (
            self.drift_analysis['acc_p_val'] is not None and
            self.drift_analysis['acc_p_val'] < self.acc_certainty_threshold
        ):
            LOGGER.warning(
                'Accuracy Drift was detected in model %s.\n'
                'Trading Rounds will cease and Model Building will be triggered.\n'
                'drift_analysis:\n%s\n',
                self.champion.model_id, pformat(self.drift_analysis)
            )
            
            self.set_triggers()

    def analyze_cum_ret_drift(
        self,
        recreate_sim: bool = False,
        debug: bool = False
    ):
        """
        Simulation:
            - Estimate expected_avg_interval_ret & expected ret std
            - Simulate 100k portfolios with that ret
            - Estimate cum_ret CI
            - Determine drift %
        """
        def simulate_run(mu, sigma, size):
            random_returns = np.random.normal(loc=mu, scale=sigma, size=size)
            random_cum_ret = np.cumprod(1 + random_returns) - 1
            return random_cum_ret

        if self.simulated_returns is None or self.model_trading_returns.shape[0] == 0 or recreate_sim:
            print(f'Re-creating simulated returns:\n')

            # Load cleaned_data
            # cleaned_data = pd.read_parquet(os.path.join(
            #     Params.base_cwd, Params.bucket, "data_processing", "data_cleaner", self.intervals,
            #     f'{self.champion.coin_name}_cleaned_data.parquet'
            # ))

            # Calculate sigma
            # sigma = cleaned_data['target_return'].std()
            sigma = self.champion.optimized_table['real_trading_return'].std()

            # Prepare simulation idx
            freq = {
                '30min': '30min',
                '60min': '60min',
                '1d': '1D'
            }[self.intervals]

            if self.model_trading_returns.shape[0] == 0:
                client = BinanceClient()

                start_date = client.get_data(
                    coin_name=self.champion.coin_name,
                    intervals=self.intervals,
                    periods=10,
                    ignore_last_period=True
                ).index[-1]
            else:
                start_date = self.model_trading_returns.index[0]

            final_date = start_date + pd.Timedelta(days=365)
            full_idx = pd.date_range(start_date, final_date, freq=freq)

            # Run raw_simulations
            raw_simulations = pd.DataFrame(index=full_idx)
            for run in tqdm(range(30000)):
                raw_simulations[run] = simulate_run(
                    mu=self.expected_avg_period_ret, 
                    sigma=sigma,
                    size=len(full_idx)
                )

            sim_cols = raw_simulations.columns.tolist()

            # Avg Returns & Quantiles
            raw_simulations['Q-1%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.01), axis=1)
            raw_simulations['Q-5%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.05), axis=1)
            raw_simulations['Q-10%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.1), axis=1)
            raw_simulations['Q-25%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.25), axis=1)

            raw_simulations['mean'] = raw_simulations.mean(axis=1)

            raw_simulations['Q-75%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.75), axis=1)
            raw_simulations['Q-90%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.90), axis=1)
            raw_simulations['Q-95%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.95), axis=1)
            raw_simulations['Q-99%'] = raw_simulations[sim_cols].apply(lambda x: np.quantile(x, 0.99), axis=1)

            # Extract simulated_returns
            self.simulated_returns = raw_simulations.filter(
                items=[c for c in raw_simulations.columns if c not in sim_cols]
            )

            print(f'self.simulated_returns.head():\n {self.simulated_returns.head()}\n\n'
                  f'self.simulated_returns.tail():\n {self.simulated_returns.tail()}\n\n')

        # Update drift_analysis
        if self.model_trading_returns.shape[0] > 1:
            compare_idx = self.model_trading_returns.index[-1]
            compare_vals: pd.Series = self.simulated_returns.loc[compare_idx]
            self.drift_analysis['cum_ret_quantiles'] = compare_vals.to_dict()

            if debug:
                print(f'last cum_ret: {self.model_trading_returns.at[compare_idx, "total_cum_returns"]}\n'
                      f'cum_ret_quantiles:')
                pprint(self.drift_analysis['cum_ret_quantiles'])
                print('\n\n')

            if self.cum_ret_certainty_threshold is None:
                LOGGER.warning(
                    'self.cum_ret_certainty_threshold is None.\n'
                    'Therefore, it will be set to "Q-5%".\n',
                )
                self.cum_ret_certainty_threshold = 'Q-5%'

            real_cum_ret = self.model_trading_returns.at[compare_idx, 'total_cum_returns']
            threshold_val = self.drift_analysis['cum_ret_quantiles'][self.cum_ret_certainty_threshold]

            if real_cum_ret < threshold_val:
                LOGGER.warning(
                    'Cum Ret Drift was detected in model %s.\n'
                    'Trading Rounds will cease and Model Building will be triggered.\n'
                    'drift_analysis:\n%s\n',
                    self.champion.model_id, pformat(self.drift_analysis)
                )
                
                self.set_triggers()
    
    @timing
    def drift_pipeline(
        self,
        days_to_eval: int = None,
        recreate_sim: bool = False,
        debug: bool = False
    ):
        # Analyze Average Return Drift
        self.analyze_avg_ret_drift(
            days_to_eval=days_to_eval,
            debug=debug
        )

        # Analyze Accuracy Drift
        self.analyze_acc_drift(
            days_to_eval=days_to_eval,
            debug=debug
        )

        # Analyze Cumulative Return Drift
        self.analyze_cum_ret_drift(
            recreate_sim=recreate_sim,
            debug=debug
        )

        self.save()
    
    def save(
        self,
        debug: bool = False
    ):
        # if not os.path.exists(self.base_path):
        #     os.makedirs(self.base_path)

        """
        Step 1) Save .pickle files
        """
        pickle_attrs = {key: value for key, value in self.__dict__.items() if key in self.load_pickle}
        
        # File System
        # with open(os.path.join(self.base_path, "drift.pickle"), 'wb') as handle:
        #     pickle.dump(pickle_attrs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # S3
        write_to_s3(
            asset=pickle_attrs,
            path=f"{self.s3_base_path}/drift.pickle"
        )

        if debug:
            for attr_key, attr_value in pickle_attrs.items():            
                print(f'Saved pickle {attr_key}:')
                pprint(attr_value)
                print('\n')
        
        """
        Step 3) Save .parquet files
        """
        for attr_name in self.load_parquet:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                # File System
                # df.to_parquet(os.path.join(self.base_path, f"{attr_name}.parquet"))

                # S3
                write_to_s3(
                    asset=df,
                    path=f"{self.s3_base_path}/{attr_name}.parquet"
                )

    def load(
        self,
        debug: bool = False
    ):
        """
        Load .pickle files
        """
        pickle_attrs = None
        try:
            # Load parquet attribute
            pickle_attrs: dict = load_from_s3(path=f"{self.s3_base_path}/drift.pickle")

            for attr_key, attr_value in pickle_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

                    if debug:
                        print(f'Loaded pickle {attr_key}:')
                        pprint(attr_value)
                        print('\n')
        except Exception as e:
            LOGGER.critical(
                'Unable to load drift pickle_attrs (%s).\n'
                'Exception: %s\n',
                self.intervals, e
            )

        """
        Load .parquet files
        """
        for attr_name in self.load_parquet:
            try:
                # Load parquet attribute
                setattr(self, attr_name, load_from_s3(
                    path=f"{self.s3_base_path}/{attr_name}.parquet"
                ))
            except Exception as e:
                LOGGER.critical('Unable to load %s.parquet.', attr_name)
