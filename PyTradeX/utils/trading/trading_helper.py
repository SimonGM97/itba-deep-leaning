from PyTradeX.config.params import Params
from PyTradeX.trading.trading_table import TradingTable
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Tuple
from tqdm import tqdm
from typing import List, Dict, Tuple
from pprint import pformat
from copy import deepcopy


def load_ltp_lsl_stp_ssl(
    coin_name: str = None,
    intervals: str = Params.general_params.get('intervals')
):
    # Load file
    ltp_lsl_stp_ssl = load_from_s3(
        path=f"{Params.bucket}/utils/ltp_lsl_stp_ssl/{intervals}/ltp_lsl_stp_ssl.json"
    )

    # Find full_coin_list
    if coin_name is None:
        return ltp_lsl_stp_ssl        
    else:
        return ltp_lsl_stp_ssl[coin_name]

def find_ltp_lsl_stp_ssl(
    df: pd.DataFrame,
    coin_name: str,
    intervals: str,
    trading_metric_type: str = 'avg_ret',
    plot_returns: bool = False,
    debug: bool = False
) -> Tuple[float, float, float, float]:
    def prepare_dummy_forecast_df() -> pd.DataFrame:
        # Extract normal dist parameters
        long_mean, long_std = long_df['target_return'].mean(), long_df['target_return'].std()
        short_mean, short_std = short_df['target_return'].mean(), short_df['target_return'].std()

        # Define random seed
        np.random.seed(23111997)

        # Prepare dummy_forecast_df
        dummy_forecast_df = pd.DataFrame({
            'accuracy': np.random.choice(a=[True, False], size=df.shape[0], p=[0.55, 0.45]),
            'real_return': df['target_return'].values,
            'long_sim_ret': np.abs(np.random.normal(loc=long_mean, scale=long_std, size=df.shape[0])),
            'short_sim_ret': -np.abs(np.random.normal(loc=short_mean, scale=short_std, size=df.shape[0])),
            'open': df['coin_open'].values,
            'high': df['coin_high'].values,
            'low': df['coin_low'].values,
            'price': df['coin_price'].values,
        }, index=df.index.tolist())

        # Simulate returns
        def extract_sim_ret(row: pd.Series):
            if row['accuracy']:
                if row['real_return'] >= 0:
                    return row['long_sim_ret']
                else:
                    return row['short_sim_ret']
            else:
                if row['real_return'] >= 0:
                    return row['short_sim_ret']
                else:
                    return row['long_sim_ret']

        dummy_forecast_df['return_forecast'] = dummy_forecast_df.apply(extract_sim_ret, axis=1)

        # if debug:
        #     print(f'raw dummy_forecast_df:\n{dummy_forecast_df}\n\n')

        # Filter required columns
        dummy_forecast_df = dummy_forecast_df.filter(items=[
            'return_forecast',
            'real_return',
            'open',
            'high',
            'low',
            'price'
        ])

        return dummy_forecast_df

    def prepare_table(trading_parameters: dict) -> TradingTable:
        table = TradingTable(
            dummy_forecast_df.copy(),
            **trading_table_input.copy(),
            trading_parameters=trading_parameters,
            initialize=True,
            table_name="dummy_trading_df",
            load_table=False,
            debug=False
        )

        # Update Residuals Attributes
        for attr_name, attr_value in residuals_attrs.items():
            setattr(table, attr_name, attr_value)
        
        # Complete table
        table.complete_table(
            find_best_dist=False,
            dummy_proof=False
        )

        # if debug:
        #     print(f'table:\n{table.tail(10)}\n\n')

        # Measure new performance
        table.measure_trading_performance(
            smooth_returns=Params.trading_params.get('smooth_returns'),
            return_weight=Params.trading_params.get('return_weight'),
            debug=False
        )

        # if debug:
        #     perf_attrs = table.show_attrs(
        #         general_attrs=False, 
        #         residuals_attrs=False, 
        #         performance_attrs=True
        #     )
        #     print(f'Performance attrs:\n{perf_attrs}\n\n')

        return table
    
    def find_trading_metric(table: TradingTable) -> float:
        if trading_metric_type == 'standard':
            # Find Weighted ML Accuracy Score
            weighted_trading_accuracy_score = (
                (int(round(table.weighted_trading_accuracy * 100, 0)) - 50) 
                * table.n_trade_perc
                * (1 - table.ret_pvalue)
            )

            # Find Estimated Monthly Return Score
            est_monthly_ret_score = round(table.est_monthly_ret * 100, 3)

            # Risk-Reward Score
            risk_reward_score = min([100, round(est_monthly_ret_score, 3)])
            
            # Return trading_metric
            return weighted_trading_accuracy_score + risk_reward_score / 7

        elif trading_metric_type == 'monthly_ret':
            return table.est_monthly_ret
        
        elif trading_metric_type == 'avg_ret':
            return table.avg_real_trading_return
        
        raise Exception(f'Invalid "trading_metric_type" was given: {trading_metric_type}.\n\n')

    # Prepare Datasets
    df['high_return'] = (df['coin_high'] - df['coin_open']) / df['coin_open']
    df['low_return'] = (df['coin_low'] - df['coin_open']) / df['coin_open']

    # Prepare long_df & short_df
    long_df = df.loc[df['target_return'] >= 0]
    short_df = df.loc[df['target_return'] < 0]

    # Prepare dummy forecasts
    dummy_forecast_df: pd.DataFrame = prepare_dummy_forecast_df()

    """
    Find dummy table
    """
    # Prepare dummy TradingTable
    trading_table_input = {
        'model_id': 'dummy_id',
        'coin_name': coin_name,
        'intervals': intervals,

        'algorithm': 'dummy_algo',
        'method': 'dummy_method',
        'pca': False,
    }

    trading_parameters = {
        'follow_flow': False,
        'certainty_threshold': 0.55,
        'max_leverage': 1,
        'tp': (None, None),
        'sl': (None, None),
        'long_permission': True,
        'short_permission': True
    }

    dummy_table = TradingTable(
        dummy_forecast_df.copy(),
        **trading_table_input.copy(),
        trading_parameters=trading_parameters,
        initialize=True,
        table_name="dummy_trading_df",
        load_table=False,
        debug=False
    )

    # Complete table
    dummy_table.complete_table(
        find_best_dist=False,
        dummy_proof=False
    )

    # Extract residuals_attrs
    residuals_attrs = deepcopy(dummy_table.residuals_attrs)

    # Measure benchmark performance
    dummy_table.measure_trading_performance(
        smooth_returns=Params.trading_params.get('smooth_returns'),
        return_weight=Params.trading_params.get('return_weight'),
        debug=False
    )

    if debug:
        print(f'Benchmark metric: {find_trading_metric(dummy_table)}\n'
              f'Performance attrs:\n{dummy_table.show_attrs(general_attrs=False, residuals_attrs=False)}\n\n')

    # Define simulation columns
    sim_columns = [
        'quantile', 'trading_metric', 'triggers_perc',
        'est_monthly_ret', 'est_stand_error', 
        'weighted_trading_accuracy'
    ]

    """
    Find Long Take Profits
    """
    # Define empty ltp_df
    ltp_df: pd.DataFrame = pd.DataFrame(columns=sim_columns)

    best_metric: float = None
    ltp_table: TradingTable = None

    print(f'\nFinding {coin_name} optimal Long Take Profits:')

    for q in tqdm(np.linspace(0.7, 0.99, 29)):
        # Long Take Profits
        ltp = np.quantile(long_df['high_return'], q)

        # if debug:
        #     print(f'ltp: {ltp}\n')

        # Prepare TradingTable
        trading_parameters = {
            'follow_flow': False,
            'certainty_threshold': 0.55,
            'max_leverage': 1,
            'tp': (ltp, -0.9),
            'sl': (None, None),
            'long_permission': True,
            'short_permission': True
        }

        # Find table
        table: TradingTable = prepare_table(trading_parameters)
        
        # Extract trading_metric
        trading_metric = find_trading_metric(table)

        # Calculate percentage of triggers
        triggers_perc = table.loc[table['tp_triggered']].shape[0] / table.shape[0]

        # Populate ltp_df
        ltp_df.loc[ltp] = [
            q, trading_metric, triggers_perc, table.est_monthly_ret,
            table.est_stand_error, table.weighted_trading_accuracy
        ]

        # Re-set ltp_table
        if best_metric is None or trading_metric > best_metric:
            best_metric = trading_metric
            ltp_table = table

    # Find optimal ltp
    optimal_ltp = ltp_df.loc[
        ltp_df['trading_metric'] == ltp_df['trading_metric'].max()
    ].index[0]

    if debug:
        print(f'optimal_ltp: {optimal_ltp} ({ltp_df["trading_metric"].max()})\n'
              f'ltp_df:\n{ltp_df.tail(10)}\n\n')
    
    """
    Find Short Take Profits
    """
    # Define empty stp_df
    stp_df: pd.DataFrame = pd.DataFrame(columns=sim_columns)

    best_metric: float = None
    stp_table: TradingTable = None

    print(f'\nFinding {coin_name} optimal Short Take Profits:')

    for q in tqdm(np.linspace(0.01, 0.3, 29)):
        # Short Take Profits
        stp = np.quantile(short_df['low_return'], q)

        # if debug:
        #     print(f'stp: {stp}\n')

        # Prepare TradingTable
        trading_parameters = {
            'follow_flow': False,
            'certainty_threshold': 0.55,
            'max_leverage': 1,
            'tp': (1.0, stp),
            'sl': (None, None),
            'long_permission': True,
            'short_permission': True
        }

        # Find table
        table: TradingTable = prepare_table(trading_parameters)
        
        # Extract trading_metric
        trading_metric = find_trading_metric(table)

        # Calculate percentage of triggers
        triggers_perc = table.loc[table['tp_triggered']].shape[0] / table.shape[0]
        
        # Populate stp_df
        stp_df.loc[stp] = [
            q, trading_metric, triggers_perc, table.est_monthly_ret,
            table.est_stand_error, table.weighted_trading_accuracy
        ]

        # Re-set stp_table
        if best_metric is None or trading_metric > best_metric:
            best_metric = trading_metric
            stp_table = table

    # Find optimal ltp
    optimal_stp = stp_df.loc[
        stp_df['trading_metric'] == stp_df['trading_metric'].max()
    ].index[0]

    if debug:
        print(f'optimal_stp: {optimal_stp} ({stp_df["trading_metric"].max()})\n'
              f'stp_df:\n{stp_df.head(10)}\n\n')
    
    """
    Find Long Stop Loss
    """
    # Define empty lsl_df
    lsl_df: pd.DataFrame = pd.DataFrame(columns=sim_columns)

    best_metric: float = None
    lsl_table: TradingTable = None

    print(f'\nFinding {coin_name} optimal Long Stop Loss:')

    for q in tqdm(np.linspace(0.01, 0.15, 14)):
        # Long Stop Loss
        lsl = np.quantile(short_df['low_return'], q)

        # if debug:
        #     print(f'lsl: {lsl}\n')

        # Prepare TradingTable
        trading_parameters = {
            'follow_flow': False,
            'certainty_threshold': 0.55,
            'max_leverage': 1,
            'tp': (None, None),
            'sl': (lsl, 1.0),
            'long_permission': True,
            'short_permission': True
        }

        # Find table
        table: TradingTable = prepare_table(trading_parameters)
        
        # Extract trading_metric
        trading_metric = find_trading_metric(table)

        # Calculate percentage of triggers
        triggers_perc = table.loc[table['sl_triggered']].shape[0] / table.shape[0]
        
        # Populate lsl_df
        lsl_df.loc[lsl] = [
            q, trading_metric, triggers_perc, table.est_monthly_ret,
            table.est_stand_error, table.weighted_trading_accuracy
        ]

        # Re-set lsl_table
        if best_metric is None or trading_metric > best_metric:
            best_metric = trading_metric
            lsl_table = table

    # Find optimal lsl
    optimal_lsl = lsl_df.loc[
        lsl_df['trading_metric'] == lsl_df['trading_metric'].max()
    ].index[0]

    if debug:
        print(f'optimal_lsl: {optimal_lsl} ({lsl_df["trading_metric"].max()})\n'
              f'lsl_df:\n{lsl_df.head(10)}\n\n')
    
    """
    Find Short Stop Loss
    """
    # Define empty ssl_df
    ssl_df: pd.DataFrame = pd.DataFrame(columns=sim_columns)

    best_metric: float = None
    ssl_table: TradingTable = None

    print(f'\nFinding {coin_name} optimal Short Stop Loss:')

    for q in tqdm(np.linspace(0.85, 0.99, 14)):
        # Short Stop Loss
        ssl = np.quantile(long_df['high_return'], q)

        # if debug:
        #     print(f'ssl: {ssl}\n')

        # Prepare TradingTable
        trading_parameters = {
            'follow_flow': False,
            'certainty_threshold': 0.55,
            'max_leverage': 1,
            'tp': (None, None),
            'sl': (-0.9, ssl),
            'long_permission': True,
            'short_permission': True
        }

        # Find table
        table: TradingTable = prepare_table(trading_parameters)
        
        # Extract trading_metric
        trading_metric = find_trading_metric(table)

        # Calculate percentage of triggers
        triggers_perc = table.loc[table['sl_triggered']].shape[0] / table.shape[0]
        
        # Populate ssl_df
        ssl_df.loc[ssl] = [
            q, trading_metric, triggers_perc, table.est_monthly_ret,
            table.est_stand_error, table.weighted_trading_accuracy
        ]

        # Re-set ssl_table
        if best_metric is None or trading_metric > best_metric:
            best_metric = trading_metric
            ssl_table = table

    # Find optimal lsl
    optimal_ssl = ssl_df.loc[
        ssl_df['trading_metric'] == ssl_df['trading_metric'].max()
    ].index[0]

    if debug:
        print(f'optimal_ssl: {optimal_ssl} ({ssl_df["trading_metric"].max()})\n'
              f'ssl_df:\n{ssl_df.tail(10)}\n\n')

    if plot_returns:
        # Prepare returns_df
        returns_df: pd.DataFrame = (
            dummy_table
            .filter(items=['total_cum_returns'])
            .rename(columns={'total_cum_returns': 'dummy_total_cum_returns'})
        )

        returns_df['ltp_total_cum_returns'] = ltp_table['total_cum_returns'].values
        returns_df['stp_total_cum_returns'] = stp_table['total_cum_returns'].values
        returns_df['lsl_total_cum_returns'] = lsl_table['total_cum_returns'].values
        returns_df['ssl_total_cum_returns'] = ssl_table['total_cum_returns'].values
        
        returns_df = returns_df.multiply(100)

        # Create plotly figure
        fig = go.Figure()

        # Add dummy cumulative returns
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df['dummy_total_cum_returns'], 
            mode='lines',
            line={
                'width': 2,
                'color': '#212422'
            },
            name='Dummy returns'
        ))

        # Add LTP cumulative returns
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df['ltp_total_cum_returns'], 
            mode='lines',
            line={
                'width': 2,
                'color': '#40FF00'
            },
            name='Optimal LTP returns'
        ))

        # Add STP cumulative returns
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df['stp_total_cum_returns'], 
            mode='lines',
            line={
                'width': 2,
                'color': '#008F02'
            },
            name='Optimal STP returns'
        ))

        # Add LSL cumulative returns
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df['lsl_total_cum_returns'], 
            mode='lines',
            line={
                'width': 2,
                'color': '#FF0000'
            },
            name='Optimal LSL returns'
        ))

        # Add SSL cumulative returns
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df['ssl_total_cum_returns'], 
            mode='lines',
            line={
                'width': 2,
                'color': '#8E0000'
            },
            name='Optimal SSL returns'
        ))
        
        # Update layout
        title = {
            'text': "Cumulative Returns [%]",
            'y':0.9, # new
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top' # new
        }

        fig.update_layout(
            title=title,
            width=1650,
            height=900,
            showlegend=True
        )

        # Show plot
        fig.show()

        # Save dataframes to examine
        # save_cols = [
        #     'return_forecast', 'real_return', 'open', 'high', 'low', 'price', 'decrease_chance', 
        #     'increase_chance', 'certainty', 'kind', 'tp_price', 'sl_price', 'tp_triggered', 
        #     'sl_triggered', 'trading_signal', 'exposure', 'exposure_adj', 'fees', 'trading_return', 
        #     'real_trading_return', 'total_cum_returns', 'trading_accuracy', 'ml_accuracy', 'trade_class'
        # ]
        # ltp_table[save_cols].tail(1000).to_excel('ltp_table.xlsx')
        # lsl_table[save_cols].tail(1000).to_excel('lsl_table.xlsx')
        # stp_table[save_cols].tail(1000).to_excel('stp_table.xlsx')
        # ssl_table[save_cols].tail(1000).to_excel('ssl_table.xlsx')
    
    # Delete dfs from memory
    del df
    del long_df    
    del short_df
    del ltp_df
    del ltp_table
    del lsl_df
    del lsl_table
    del stp_df
    del stp_table
    del ssl_df
    del ssl_table

    return optimal_ltp, optimal_lsl, optimal_stp, optimal_ssl

def update_ltp_lsl_stp_ssl(
    coin_name: str = None,
    intervals: str = Params.general_params.get('intervals'),
    trading_metric_type: str = 'avg_ret',
    debug: bool = False
):
    # Find full_coin_list
    if coin_name is None:
        full_coin_list: List[str] = Params.fixed_params.get('full_coin_list')
    else:
        full_coin_list: List[str] = [coin_name]

    # Load ltp_lsl_stp_ssl
    try:
        ltp_lsl_stp_ssl: Dict[str, Tuple] = load_ltp_lsl_stp_ssl(
            coin_name=None,
            intervals=intervals
        )
    except Exception as e:
        print(f'[WARNING] Unable to load ltp_lsl_stp_ssl.\n'
              f'Exception: {e}\n\n')
        ltp_lsl_stp_ssl: Dict[str, Tuple] = {}

    for coin_name in full_coin_list:
        # Load DataCleaner
        DC = DataCleaner(
            coin_name=coin_name,
            intervals=intervals,
            overwrite=True,
            **Params.data_params.copy()
        )

        # Extract LTP, LSL, STP & SSL
        ltp, lsl, stp, ssl = find_ltp_lsl_stp_ssl(
            df=DC.cleaned_data.copy(),
            coin_name=coin_name,
            intervals=intervals,
            trading_metric_type=trading_metric_type,
            debug=debug
        )

        # Populate ltp_lsl_stp_ssl
        ltp_lsl_stp_ssl[coin_name] = ltp, lsl, stp, ssl

    if debug:
        print(f'ltp_lsl_stp_ssl:\n{pformat(ltp_lsl_stp_ssl)}\n\n')

    # Save ltp_lsl_stp_ssl
    write_to_s3(
        asset=ltp_lsl_stp_ssl,
        path=f"{Params.bucket}/utils/ltp_lsl_stp_ssl/{intervals}/ltp_lsl_stp_ssl.json"
    )


# source deactivate
# conda deactivate
# source .pytradex_venv/bin/activate
# .pytradex_venv/bin/python PyTradeX/utils/trading/trading_helper.py
if __name__ == '__main__':
    coin_name = 'XRP'
    intervals = '60min'
    trading_metric_type = 'avg_ret'

    print(pformat(load_ltp_lsl_stp_ssl(coin_name='XRP')))

    # update_ltp_lsl_stp_ssl(
    #     coin_name=None,
    #     intervals=intervals,
    #     trading_metric_type=trading_metric_type,
    #     debug=True
    # )