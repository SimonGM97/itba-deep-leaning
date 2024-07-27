from PyTradeX.config.params import Params
from scripts.trading import trading
from scripts.data_processing import data_processing
from scripts.data_warehousing import data_warehousing
from PyTradeX.utils.general.client import BinanceClient
from PyTradeX.utils.others.s3_helper import load_from_s3
from PyTradeX.utils.general.logging_helper import get_logger
from functools import partial
import argparse
import schedule


def trigger():
    return load_from_s3(
        path=f"{Params.bucket}/utils/triggers/{Params.general_params.get('intervals')}/trading_round_trigger.json"
    )


def find_forced_intervals() -> str:
    if Params.general_params.get('intervals') == '30min':
        return '60min'
    elif Params.general_params.get('intervals') == '60min':
        return '30min'
    else:
        raise Exception(f'Invalid "intervals" parameter was infered: {Params.general_params.get("intervals")}')


def run_workflow(
    account_id: int,
    workflow: dict
) -> None:
    if trigger()['run']:
        LOGGER.info('Running TRADING ROUND workflow:')

        # Run trading job
        trading.lambda_handler(
            event={
                'account_id': account_id,
                'workflow': workflow
            },
            context=None
        )

        # Run data_processing job
        data_processing.lambda_handler(
            event={
                'workflow': workflow,
                'forced_intervals': None
            },
            context=None
        )

        # Find forced_intervals
        forced_intervals = find_forced_intervals()
        
        # Run data_processing job with forced intervals
        data_processing.lambda_handler(
            event={
                'workflow': workflow,
                'forced_intervals': forced_intervals
            },
            context=None
        )

        # Run data_warehousing job
        data_warehousing.lambda_handler(
            event={'workflow': workflow},
            context=None
        )


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


# source deactivate
# conda deactivate
# source .pytradex_venv/bin/activate
# yes | conda uninstall xgboost py-xgboost py-xgboost-cpu libxgboost
# .pytradex_venv/bin/python workflows/trading_round/trading_round_workflow.py
#   --account_id 0
if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description='Trading round workflow.')

    # Add account_id argument
    parser.add_argument(
        '--account_id',
        type=int,
        default=0
    )

    # Extract arguments
    args = parser.parse_args()
    
    # Retrieve account_id
    account_id: int = args.account_id

    # Assert that balances can be retrieved
    client = BinanceClient(account_id=account_id)
    balance_df = client.get_futures_balance(
        include_usd_valuation=True,
        intervals=Params.general_params.get('intervals'),
        debug=False
    )

    assert balance_df.shape[0] > 0

    LOGGER.info('Balances were correctly loaded.')

    # Define workflow function with expected params
    run_workflow_w_params = partial(
        run_workflow,
        account_id=account_id,
        workflow='trading_round'
    )
    
    # Set Up Schedule
    if Params.general_params.get('intervals') == '30min':
        # Define initial time
        time = '00:28:30'

        # Add initial time to schedule
        schedule.every().day.at(time).do(run_workflow_w_params)

        while time != '23:58:30':
            # Add 30min to time
            if time[3:5] == '28':                
                time = time.replace(':28:', ':58:')
            else:
                new_hr = ('0' + str(int(time[:2])+1))[-2:] + ':'
                time = time.replace(time[:3], new_hr)
                time = time.replace(':58:', ':28:')
            
            # Add new time to scheadule
            schedule.every().day.at(time).do(run_workflow_w_params)

    elif Params.general_params.get('intervals') == '60min':
        # Define initial time
        time = '00:58:30'

        # Add initial time to schedule
        schedule.every().day.at(time).do(run_workflow_w_params)

        while time != '23:58:30':
            # Add 60min to time
            new_hr = ('0' + str(int(time[:2])+1))[-2:] + ':'
            time = time.replace(time[:3], new_hr)
            
            # Add new time to scheadule
            schedule.every().day.at(time).do(run_workflow_w_params)

    else:
        LOGGER.critical('Invalid "intervals": %s', Params.general_params.get('intervals'))
    
    # Run Schedule
    while True:
        schedule.run_pending()