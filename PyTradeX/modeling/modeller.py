from PyTradeX.config.params import Params
from PyTradeX.utils.modeling.modeller_helper import load_modeling_datasets
from PyTradeX.utils.pipeline.pipeline_helper import update_GFM_train_coins
from PyTradeX.utils.trading.trading_helper import update_ltp_lsl_stp_ssl
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing

from PyTradeX.data_processing.feature_selector import FeatureSelector
from PyTradeX.modeling.model import Model
from PyTradeX.modeling.model_registry import ModelRegistry
from PyTradeX.modeling.model_tuning import ModelTuner
from PyTradeX.pipeline.ml_pipeline import MLPipeline

import plotly.graph_objects as go
import pandas as pd
import os
import json
from tqdm import tqdm
from typing import Dict, List


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


def plot_model_returns(model: Model) -> None:
    # Load Colors Dict
    with open(os.path.join("resources", "coin_colors.json"), 'r') as file:
        colors_dict: dict = json.load(file)

    # Concatenate val_table & optimized_table
    opt_complete_table = pd.concat([model.val_table.copy(), model.optimized_table.copy()])
    opt_complete_table['total_cum_returns'] = ((1 + opt_complete_table['real_trading_return']).cumprod() - 1) * 100
    
    # Define plotly figure
    fig = go.Figure()

    # Add val & opt total cumulative returns
    fig.add_trace(go.Scatter(
        x=opt_complete_table.index, 
        y=opt_complete_table['total_cum_returns'], 
        line_color=colors_dict[model.coin_name]['primary_color'], 
        showlegend=False
    ))

    # Fill area with secondary_color
    fig.add_trace(go.Scatter(
        x=opt_complete_table.index, 
        y=opt_complete_table['total_cum_returns'], 
        fill='tozeroy', 
        fillcolor=colors_dict[model.coin_name]['secondary_color'],
        showlegend=False
    ))

    # Find initial & maximum cumulative return 
    ini_test_cum_ret = opt_complete_table.loc[model.test_table.index[0], 'total_cum_returns'] # .values[0]
    max_cum_ret = opt_complete_table['total_cum_returns'].max()

    # Add vertical line to divide val & test
    fig.add_vline(
        x=model.test_table.index[0], 
        line_width=3, 
        line_dash="dash", 
        line_color="black",
        y0=0.055,
        y1=(ini_test_cum_ret / max_cum_ret) + 0.03
    )
    
    # Add title & set width
    fig.update_layout(
        title='Valdation & Test Cumulative Returns',
        yaxis={
            'title': 'Cumulative return [%]'
        },
        width=1300,
        height=600
    )

    # Show figure
    fig.show()


@timing
def modeling_job(
    intervals: str = None,
    full_coin_list: list = None,
    methods: List[str] = None,

    data_params: dict = None,
    ml_params: dict = None,
    trading_params: dict = None,
    
    tuning_params: dict = None,
    updating_params: dict = None
):
    # Tune models
    if (
        tuning_params is not None 
        and tuning_params.get('tune_models')
    ):
        # Load ml_datasets
        ml_datasets = load_modeling_datasets(
            intervals=intervals,
            full_coin_list=full_coin_list,
            methods=methods,
            data_params=data_params,
            from_local=tuning_params.get('load_datasets_from_local'),
            to_local=tuning_params.get('save_datasets_to_local'),
            logger=LOGGER,
            debug=tuning_params.get('debug')
        )

        # Update GFM train_coins
        if tuning_params.get('update_GFM_train_coins'):
            update_GFM_train_coins(
                ml_datasets=ml_datasets,
                intervals=intervals,
                logger=LOGGER,
                debug=tuning_params.get('debug')
            )
        
        # Update ltp_lsl_stp_ssl
        if tuning_params.get('update_ltp_lsl_stp_ssl'):
            update_ltp_lsl_stp_ssl(
                coin_name=None,
                intervals=intervals,
                trading_metric_type='avg_ret',
                plot_returns=False,
                debug=tuning_params.get('debug')
            )
        
        # Load FeatureSelector
        FS = FeatureSelector(
            intervals=intervals,
            **data_params
        )

        # Instanciate ModelTuner
        model_tuner = ModelTuner(
            intervals=intervals,
            coin_names=full_coin_list,
            methods=methods,

            data_params=data_params,
            ml_params=ml_params,
            trading_params=trading_params
        )
        
        # Run tune_models method
        model_tuner.tune_models(
            ml_datasets=ml_datasets,
            selected_features=FS.selected_features,
            reduced_tuning_periods=tuning_params.get('reduced_tuning_periods'),
            debug=tuning_params.get('debug'),
            deep_debug=tuning_params.get('deep_debug')
        )

        # Delete ml_datasets from memory
        del ml_datasets

    # Update models
    if (
        updating_params is not None 
        and updating_params.get('update_models')
    ):
        # Instanciate ModelRegistry
        model_registry = ModelRegistry(
            n_candidates=ml_params.get('n_candidates'),
            intervals=intervals
        )

        # Load champion, challengers & dev models
        champion: Model = model_registry.load_prod_model(light=False)
        challengers: List[Model] = model_registry.load_staging_models(light=False)
        dev_models: List[Model] = model_registry.load_dev_models(light=False)

        # Load complete ml_datasets
        if updating_params.get('re_set_models'):
            ml_datasets = load_modeling_datasets(
                intervals=intervals,
                full_coin_list=full_coin_list,
                methods=methods,
                data_params=data_params,
                from_local=updating_params.get('load_datasets_from_local'),
                to_local=updating_params.get('save_datasets_to_local'),
                logger=LOGGER,
                debug=updating_params.get('debug')
            )

        # Update GFM train_coins
        if updating_params.get('update_GFM_train_coins'):
            update_GFM_train_coins(
                ml_datasets=ml_datasets,
                intervals=intervals,
                debug=updating_params.get('debug')
            )

        # Update ltp_lsl_stp_ssl
        if updating_params.get('update_ltp_lsl_stp_ssl'):
            for coin_name in set([m.coin_name for m in [champion] + challengers + dev_models]):
                update_ltp_lsl_stp_ssl(
                    coin_name=coin_name,
                    intervals=intervals,
                    trading_metric_type='avg_ret',
                    debug=tuning_params.get('debug')
                )
        
        # with ThreadPoolExecutor(max_workers=Params.cpus) as executor:
        for model in tqdm([champion] + challengers + dev_models):
            if isinstance(model, Model):
                # Instanciate MLPipeline
                ml_pipeline = MLPipeline(
                    pipeline_params=model.pipeline_params,
                    ml_params=ml_params,
                    trading_params=trading_params
                )

                # Re-set models
                if updating_params.get('re_set_models'):
                    LOGGER.warning(
                        'Re-setting %s Model %s (%s | %s)', 
                        intervals, model.model_id, model.stage, model.model_class
                    )

                    # Remove train_coins from GFM models
                    if model.model_class == 'GFM' and updating_params.get('update_GFM_train_coins'):
                        model.train_coins = None
                    
                    # Run build_pipeline
                    model = ml_pipeline.build_pipeline(
                        ml_datasets=ml_datasets,
                        model=None, # model.model will be re-created & re-fitted
                        ignore_update=False,
                        find_val_table=True,
                        re_fit_train_val=True,
                        find_test_table=True,
                        find_opt_table=True,
                        tune_opt_table=True,
                        find_feature_importance=True,
                        debug=updating_params.get('debug')
                    )
                else:
                    # Load ml_datasets for required model
                    ml_datasets = load_modeling_datasets(
                        intervals=intervals,
                        full_coin_list=[model.coin_name],
                        methods=methods,
                        data_params=data_params,
                        from_local=updating_params.get('load_datasets_from_local'),
                        to_local=updating_params.get('save_datasets_to_local'),
                        logger=LOGGER,
                        debug=updating_params.get('debug')
                    )
                        
                    # Run update_pipeline
                    model = ml_pipeline.update_pipeline(
                        model=model,
                        ml_datasets=ml_datasets,
                        optimize_trading_parameters=updating_params.get('optimize_trading_parameters'),
                        update_feature_importance=updating_params.get('update_feature_importance'),
                        ignore_last_update_periods=updating_params.get('ignore_last_update_periods'),
                        debug=updating_params.get('debug')
                    )

                LOGGER.info("Updated Model:\n%s\n", model)

                # Delete ml_pipeline
                del ml_pipeline
                
                # Save Model
                model.save()
        
        # Plot results
        if updating_params.get('plot_champion_returns'):
            print('Champion Model:\n')
            print(champion)

            plot_model_returns(model=champion)


        # Find all models
        champion: Model = model_registry.load_prod_model(light=False)
        staging_models: List[Model] = model_registry.load_staging_models(light=False)
        dev_models: List[Model] = model_registry.load_dev_models(light=False)

        models = [m for m in [champion] + staging_models + dev_models if m is not None]
        
        # Drop duplicate models
        models = model_registry.drop_duplicate_models(
            models=models
        )

        # Reassign model_registry.registry
        model_registry.registry = {
            "production": [(m.model_id, m.model_class) for m in models if m.stage == 'production'], 
            "staging": [(m.model_id, m.model_class) for m in models if m.stage == 'staging'], 
            "development": [(m.model_id, m.model_class) for m in models if m.stage == 'development']
        }

        # Update Model stages
        model_registry.update_model_stages(
            update_champion=updating_params.get('update_champion'),
            debug=updating_params.get('debug')
        )

        # Show ModelRegistry
        print(model_registry)

        # Delete ml_datasets from memory
        del ml_datasets


    