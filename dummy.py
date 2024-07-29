from PyTradeX.config.params import Params
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.data_processing.data_extractor import DataExtractor
from PyTradeX.data_processing.data_cleaner import DataCleaner
from PyTradeX.data_processing.data_shifter import DataShifter
from PyTradeX.data_processing.data_refiner import DataRefiner
from PyTradeX.data_processing.feature_selector import FeatureSelector
from PyTradeX.data_processing.data_transformer import DataTransformer
from PyTradeX.pipeline.ml_pipeline import MLPipeline
from PyTradeX.modeling.model_registry import ModelRegistry
from PyTradeX.modeling.model import Model
from PyTradeX.trading.trading_table import TradingTable
from PyTradeX.utils.data_processing.collective_data import (
    load_collective_data,
    get_collective_data
)
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair
)
from PyTradeX.utils.data_processing.collective_data import find_stock_data, correct_stock_data
from PyTradeX.utils.others.s3_helper import load_from_s3, write_to_s3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from tqdm import tqdm
from pprint import pprint, pformat
from copy import deepcopy
from typing import List


"""
TODO:
    - check stock data quality: 
    - check coin_open quality
    - check TA features quality
"""
client = BinanceClient()
coin_name = 'XRP'
full_coin_list: List[str] = Params.fixed_params.get('full_coin_list')
intervals: str = Params.general_params.get('intervals')
n_candidates: int = Params.ml_params.get('n_candidates')
data_params: dict = deepcopy(Params.data_params)
other_coins = Params.other_coins_json[intervals][:data_params['other_coins_n']]
overwrite = False
debug = False

def put_binance_order():
    symbol = 'XRPUSDC'

    # order = client.client.futures_create_order(
    #     symbol=symbol,
    #     side='SELL',  # BUY/SELL same as LONG, SHORT?
    #     type='LIMIT',
    #     quantity=3,
    #     price=3.31,
    #     timeInForce='GTC'
    # )
    # pprint(order)
    # client.client.futures_change_leverage(
    #     symbol=symbol,
    #     leverage=1
    # )

    stop_price = 0.59
    diff = 0.0001

    tp_order = client.client.futures_create_order(
        symbol=symbol,
        side='BUY',
        type='TAKE_PROFIT',  # 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'
        quantity=100,
        price=stop_price-diff,
        stopPrice=stop_price,
        # newClientOrderId=order['orderId'],
        # selfTradePreventionMode='EXPIRE_TAKER',
        timeInForce='GTC'
    )

    # client.client.futures_create_order(
    #     symbol=symbol,
    #     side='BUY',
    #     type='TAKE_PROFIT',  # 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'
    #     quantity=3,
    #     price=3.295, # el precio al que se pone para vender
    #     stopPrice=3.296, # el precio al que triggerea la orden
    #     newClientOrderId=order['orderId'],
    #     selfTradePreventionMode='EXPIRE_TAKER',
    #     timeInForce='GTC'
    # )

    # client.client.futures_create_order(
    #     symbol=symbol,
    #     side='BUY',
    #     type='STOP',  # 'STOP_MARKET'
    #     quantity=3,
    #     price=3.401, # el precio al que se pone para vender
    #     stopPrice=3.4, # el precio al que triggerea la orden
    #     newClientOrderId=order['orderId'],
    #     selfTradePreventionMode='EXPIRE_TAKER',
    #     timeInForce='GTC'
    # )


# conda deactivate
# source .itba_dl/bin/activate
# .itba_dl/bin/python dummy.py
if __name__ == '__main__':
    import pickle
    import os
    import json

    # Instanciate ModelRegistry
    model_registry = ModelRegistry(
        n_candidates=Params.ml_params.get('n_candidates'),
        intervals=intervals
    )

    champion: Model = model_registry.load_prod_model(light=False)

    print(champion)