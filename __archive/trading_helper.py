from PyTradeX.config.params import Params
from PyTradeX.trading.trading_table import TradingTable
import pandas as pd
import numpy as np
from typing import Tuple
from tqdm import tqdm
from pprint import pformat
from copy import deepcopy


def _find_ltp_lsl_stp_ssl(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    # Prepare Datasets
    df['high_return'] = (df['coin_high'] - df['coin_open']) / df['coin_open']
    df['low_return'] = (df['coin_low'] - df['coin_open']) / df['coin_open']

    long_df = df.loc[df['target_return'] > 0]
    short_df = df.loc[df['target_return'] < 0]

    full_len = df.shape[0]

    # Find Placeholders
    ltp_, lsl_, stp_, ssl_ =  (
        np.quantile(long_df['high_return'], q=0.95),
        np.quantile(long_df['low_return'], q=0.05),
        np.quantile(short_df['low_return'], q=0.05),
        np.quantile(short_df['high_return'], q=0.95)
    )

    """
    Find Long Take Profits
    """
    ltp_df: pd.DataFrame = pd.DataFrame(columns=['Quantile', 'Success Rate', 'Perc Triggered'])

    for q in np.linspace(0.01, 0.99, 200):
        # Long Take Profits
        ltp = np.quantile(long_df['target_return'], q)
        
        # Cases where the coin_high actually triggered the take profits
        triggered_df: pd.DataFrame = df.loc[df['high_return'] > ltp]
        triggered_len = triggered_df.shape[0]
        
        perc_triggered = 100 * triggered_len / full_len
        
        # Percentage of times where the high reached the price and the close was lower than this
        success_df: pd.DataFrame = triggered_df.loc[triggered_df['target_return'] < ltp]
        success_len = success_df.shape[0]
        
        success_rate = 100 * success_len / triggered_len
        
        ltp_df.loc[ltp] = [q, success_rate, perc_triggered]

    # ltp_df[['Perc Triggered', 'Success Rate']].plot()

    ltp_df = ltp_df.loc[
        (ltp_df['Perc Triggered'] > 3) &
        (ltp_df['Success Rate'] > 55)
    ]
    ltp_df = ltp_df.loc[
        ltp_df['Success Rate'] == ltp_df['Success Rate'].max()
    ]

    if ltp_df.shape[0] == 0:
        ltp = ltp_
    else:
        ltp = ltp_df.index[0]

    """
    Find Long Stop Loss
    """
    lsl_df: pd.DataFrame = pd.DataFrame(columns=['Quantile', 'Success Rate', 'Perc Triggered'])
    
    for q in np.linspace(0.01, 0.99, 200):
        # Long Stop Loss
        lsl = np.quantile(short_df['target_return'], q)
        
        # Cases where the coin_low actually triggered the stop loss (on positive periods)
        triggered_df = df.loc[df['low_return'] < lsl]
        triggered_len = triggered_df.shape[0]
        
        perc_triggered = 100 * triggered_len / full_len
        
        # Percentage of times where the low reached the price and the close was lower than this
        success_df = triggered_df.loc[triggered_df['target_return'] < lsl]
        success_len = success_df.shape[0]
        
        success_rate = 100 * success_len / triggered_len
        
        lsl_df.loc[lsl] = [q, success_rate, perc_triggered]

    # lsl_df[['Perc Triggered', 'Success Rate']].plot()

    lsl_df = lsl_df.loc[
        (lsl_df['Perc Triggered'] < 10) &
        (lsl_df['Success Rate'] > 45)
    ]
    lsl_df = lsl_df.loc[
        lsl_df['Success Rate'] == lsl_df['Success Rate'].max()
    ]

    if lsl_df.shape[0] == 0:
        lsl = lsl_
    else:
        lsl = lsl_df.index[0]

    """
    Find Short Take Profits
    """
    stp_df: pd.DataFrame = pd.DataFrame(columns=['Quantile', 'Success Rate', 'Perc Triggered'])

    for q in np.linspace(0.01, 0.99, 200):
        # Short Take Profits
        stp = np.quantile(short_df['target_return'], q)
        
        # Cases where the coin_low actually triggered the take profits
        triggered_df = df.loc[df['low_return'] < stp]
        triggered_len = triggered_df.shape[0]
        
        perc_triggered = 100 * triggered_len / full_len
        
        # Percentage of times where the high reached the price and the close was lower than this
        success_df = triggered_df.loc[triggered_df['target_return'] > stp]
        success_len = success_df.shape[0]
        
        success_rate = 100 * success_len / triggered_len
        
        stp_df.loc[stp] = [q, success_rate, perc_triggered]

    # stp_df[['Perc Triggered', 'Success Rate']].plot()

    stp_df = stp_df.loc[
        (stp_df['Perc Triggered'] > 3) &
        (stp_df['Success Rate'] > 55)
    ]
    stp_df = stp_df.loc[
        stp_df['Success Rate'] == stp_df['Success Rate'].max()
    ]

    if stp_df.shape[0] == 0:
        stp = stp_
    else:
        stp = stp_df.index[0]

    """
    Find Short Stop Loss
    """
    ssl_df: pd.DataFrame = pd.DataFrame(columns=['Quantile', 'Success Rate', 'Perc Triggered'])

    for q in np.linspace(0.01, 0.99, 200):
        # Short Stop Loss
        ssl = np.quantile(long_df['target_return'], q)
        
        # Cases where the coin_low actually triggered the take profits
        triggered_df = df.loc[df['high_return'] > ssl]
        triggered_len = triggered_df.shape[0]
        
        perc_triggered = 100 * triggered_len / full_len
        
        # Percentage of times where the high reached the price and the close was lower than this
        success_df = triggered_df.loc[triggered_df['target_return'] > ssl]
        success_len = success_df.shape[0]
        
        success_rate = 100 * success_len / triggered_len
        
        ssl_df.loc[ssl] = [q, success_rate, perc_triggered]

    # ssl_df[['Perc Triggered', 'Success Rate']].plot()

    ssl_df = ssl_df.loc[
        (ssl_df['Perc Triggered'] < 10) &
        (ssl_df['Success Rate'] > 45)
    ]
    ssl_df = ssl_df.loc[
        ssl_df['Success Rate'] == ssl_df['Success Rate'].max()
    ]

    if ssl_df.shape[0] == 0:
        ssl = ssl_
    else:
        ssl = ssl_df.index[0]

    # Delete dfs from memory
    del df
    del long_df
    del short_df
    del ltp_df
    del lsl_df
    del stp_df
    del ssl_df

    return ltp, lsl, stp, ssl
