from PyTradeX.config.params import Params
from PyTradeX.modeling.modeller import modeling_job
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.scripts_helper import (
    retrieve_params,
    log_params
)
from pprint import pformat
import argparse
import json


def lambda_handler(
    event: dict, 
    context: dict = None
) -> dict:
    """
    :param `event`: (dict) Data sent during lambda function invocation.
    :param `context`: (dict) Generated by the platform and contains information about the underlying infrastructure
        and execution environment, such as allowed runtime and memory.
    """
    # Log event
    LOGGER.info('event:\n%s\n', pformat(event))

    if "AWS_LAMBDA_EVENT_BODY" in event.keys():
        # Access the payload from the event parameter
        payload_str = event.get("AWS_LAMBDA_EVENT_BODY")

        # Parse the JSON payload
        payload: dict = json.loads(payload_str)

        # Extract parameters
        workflow = payload.get("workflow", "default") # "model_updating"
    else:
        # Extract parameters
        workflow = event.get("workflow", "default") # "model_updating"

    # Retrieve parameters
    params = retrieve_params(workflow=workflow)

    # Show params
    log_params(
        logger=LOGGER, 
        log_keys=[
            'intervals', 'full_coin_list', 'methods',
            'tuning_params', 'updating_params'
        ],
        **params
    )

    # Show context
    LOGGER.info('context:\n%s\n', pformat(context))

    # Run Modeling Job
    modeling_job(**params)

    return {
        'statusCode': 200,
        'body': json.dumps('Modeling job ran successfully!')
    }


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
# source .itba_dl/bin/activate
# .itba_dl/bin/python scripts/modeling/modeling.py --workflow model_updating
#   --workflow: default | model_updating | model_building
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Modeling script.')

    # Add script_setup argument
    parser.add_argument(
        '--workflow',
        type=str,
        default='default'
    )

    # Extract arguments
    args = parser.parse_args()

    # Retrieve script variables
    workflow: str = args.workflow

    # Define event
    event: dict = {'workflow': workflow}

    # Run lambda function
    lambda_handler(event=event, context=None)
