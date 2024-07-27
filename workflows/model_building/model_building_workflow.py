from PyTradeX.config.params import Params
from scripts.data_processing import data_processing
from scripts.modeling import modeling
from PyTradeX.utils.others.s3_helper import load_from_s3
from PyTradeX.utils.general.logging_helper import get_logger
import schedule


def trigger():
    return load_from_s3(
        path=f"{Params.bucket}/utils/triggers/{Params.general_params.get('intervals')}/model_building_trigger.json"
    )


def run_workflow(workflow: dict) -> None:
    if trigger()['run']:
        LOGGER.info('Running MODEL BUILDING workflow:')

        # Run data_processing job
        data_processing.lambda_handler(
            event={
                'workflow': workflow,
                'forced_intervals': None
            },
            context=None
        )

        # Run modeling job
        modeling.lambda_handler(
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


# source .pytradex_venv/bin/activate
# yes | conda uninstall xgboost py-xgboost py-xgboost-cpu libxgboost
# .pytradex_venv/bin/python workflows/model_building/model_building_workflow.py
if __name__ == '__main__':
    # Set Up Schedule
    schedule.every().day.at('00:45:00').do(run_workflow, 'model_building')
    
    # Run Schedule
    while True:
        schedule.run_pending()
    