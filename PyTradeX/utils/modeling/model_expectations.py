from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import load_from_s3
from PyTradeX.utils.testing.test_helper import dfs_are_equal
from PyTradeX.utils.general.logging_helper import get_logger
import pandas as pd
from collections import Counter


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


"""
Model Checks
"""
# No missing tables
def is_missing_val_table(model):
    # Check val_table
    if model.val_table is None:
        LOGGER.warning(
            'Model %s (%s | %s - %s) val_table is None.', 
            model.model_id, model.stage, model.model_class, model.intervals
        )
        return True
    return False

def is_missing_test_table(model):
    # Check test_table
    if model.stage != 'development' and model.test_table is None:
        LOGGER.warning(
            'Model %s (%s | %s - %s) test_table is None.',
            model.model_id, model.stage, model.model_class, model.intervals
        )
        return True
    return False

def is_missing_optimized_table(model):
    # Check test_table
    if model.stage != 'development' and model.optimized_table is None:
        LOGGER.warning(
            'Model %s (%s | %s - %s) optimized_table is None.',
            model.model_id, model.stage, model.model_class, model.intervals
        )
        return True
    return False


# Table indexes
def table_indexes_are_inconsistent(model):
    if model.stage != 'development':
        # Find validation index
        val_idx = model.val_table.index.copy()
        test_idx = model.test_table.index.copy()
        opt_idx = model.optimized_table.index.copy()

        # Check there is no intersection between val & test/opt
        val_test_intersection = set(val_idx).intersection(set(test_idx))
        val_opt_intersection = set(val_idx).intersection(set(opt_idx))

        if len(val_test_intersection) > 0:
            LOGGER.warning(
                'Model %s (%s | %s - %s) has intersection between val_idx & test_idx.\n'
                'Intersection: %s.\n',
                model.model_id, model.stage, model.model_class, model.intervals, val_test_intersection
            )
            return True
        if len(val_opt_intersection) > 0:
            LOGGER.warning(
                'Model %s (%s | %s - %s) has intersection between val_idx & test_idx.\n'
                'Intersection: %s.\n',
                model.model_id, model.stage, model.model_class, model.intervals, val_opt_intersection
            )
            return True

        # Check that test/opt is a continuation from val
        last_val_idx, first_test_idx, first_opt_idx = val_idx[-1], test_idx[0], opt_idx[0]
        time_delta = pd.Timedelta(minutes=Params.data_params.get("mbp"))

        if last_val_idx + time_delta != first_test_idx:
            LOGGER.warning(
                'Model %s (%s | %s - %s) first_test_idx (%s) is diffent from expected %s.\n'
                'last_val_idx: %s.\n',
                model.model_id, model.stage, model.model_class, model.intervals, 
                first_test_idx, last_val_idx + time_delta, last_val_idx
            )
            return True
        if last_val_idx + time_delta != first_opt_idx:
            LOGGER.warning(
                'Model %s (%s | %s - %s) first_opt_idx (%s) is diffent from expected %s.\n'
                'last_val_idx: %s.\n',
                model.model_id, model.stage, model.model_class, model.intervals, 
                first_opt_idx, last_val_idx + time_delta, last_val_idx
            )
            return True

        # Check that test_idx & opt_idx contain the same elements
        test_opt_idx_diff = set(test_idx).symmetric_difference(set(opt_idx))
        if len(test_opt_idx_diff) > 0:
            LOGGER.warning(
                'Model %s (%s | %s - %s) contain different indexes in test_table & optimized_table.\n'
                'test_opt_idx_diff: %s.\n',
                model.model_id, model.stage, model.model_class, model.intervals, test_opt_idx_diff
            )
            return True

    return False


# Selected features
def has_too_many_selected_features(model):
    max_features = Params.data_params.get('max_features')

    if len(model.selected_features) > max_features:
        LOGGER.warning(
            'Model %s (%s | %s - %s) has too many selected_features.\n'
            'len(model.selected_features): %s.\n'
            'Max allowed: %s.\n',
            model.model_id, model.stage, model.model_class, model.intervals,
            len(model.selected_features), max_features
        )
        return True
    return False

def has_duplicated_selected_features(model):
    duplicates = [k for k,v in Counter(model.selected_features).items() if v>1]
    if len(duplicates) > 0:
        LOGGER.warning(
            'Model %s (%s | %s - %s) has duplicated selected_features.\n'
            'duplicates: %s.\n',
            model.model_id, model.stage, model.model_class, model.intervals, duplicates
        )
        return True
    return False


# Test & opt forecasts consistency
def test_and_opt_forecasts_are_inconsistent(model):
    if model.stage != 'development':
        # Find test & opt forecasts
        test_forecasts = model.test_table['return_forecast'].copy()
        opt_forecasts = model.optimized_table['return_forecast'].copy()

        if not(test_forecasts.equals(opt_forecasts)):
            LOGGER.warning(
                'Model %s (%s | %s - %s) contains different test & opt forecasts.',
                model.model_id, model.stage, model.model_class, model.intervals
            )
            return True
    return False


# Forecasts determinism
def forecasts_are_not_deterministic(model):
    # Define base_data_paths
    base_dt_path = f"{Params.bucket}/data_processing/data_transformer/{model.intervals}/{model.coin_name}"

    # Load y_trans
    y_ml: pd.DataFrame = load_from_s3(
        path=f"{base_dt_path}/{model.coin_name}_y_trans.parquet",
        load_reduced_dataset=True
    )

    if model.pca:
        # Load reduced X_trans_pca
        X_ml: pd.DataFrame = (
            load_from_s3(
                path=f"{base_dt_path}/{model.coin_name}_X_trans_pca.parquet",
                load_reduced_dataset=True
            )
            .filter(items=model.selected_features)
        )
    else:
        # Load reduced X_trans
        X_ml: pd.DataFrame = (
            load_from_s3(
                path=f"{base_dt_path}/{model.coin_name}_X_trans.parquet",
                load_reduced_dataset=True
            )
            .filter(items=model.selected_features)
        )
    
    # Find intersection index
    intersection = (
        y_ml.index
        .intersection(X_ml.index)
    )

    # Filter datasets
    y_ml = y_ml.loc[intersection]
    X_ml = X_ml.loc[intersection]

    # Find train_periods
    train_periods = int(y_ml.shape[0] / 2)

    # Define Train Datasets
    y_ml_train = y_ml.iloc[:train_periods]
    X_ml_train = X_ml.iloc[:train_periods]

    # Define Test Datasets
    y_ml_test = y_ml.iloc[-train_periods:]
    X_ml_test = X_ml.iloc[-train_periods:]

    # Delete unnecessary datasets from memory
    del y_ml
    del X_ml

    # Create first forecast
    forecasts_1: pd.DataFrame = model.return_forecast(
        train_target=y_ml_train.copy(),
        forecast_target=y_ml_test.copy(),
        train_features=X_ml_train.copy(),
        forecast_features=X_ml_test.copy(),
        forecast_dates=y_ml_test.index.copy(),
        add_bias=None,
        steps=None,
        ignore_update=True,
        max_t=None,
        raw_forecasts=False,
        debug=False
    )

    # Create second forecast
    forecasts_2: pd.DataFrame = model.return_forecast(
        train_target=y_ml_train.copy(),
        forecast_target=y_ml_test.copy(),
        train_features=X_ml_train.copy(),
        forecast_features=X_ml_test.copy(),
        forecast_dates=y_ml_test.index.copy(),
        add_bias=None,
        steps=None,
        ignore_update=True,
        max_t=None,
        raw_forecasts=False,
        debug=False
    )

    # Check forecasts are the same
    if not(
        dfs_are_equal(
            df1=forecasts_1,
            df2=forecasts_2,
            df1_name='forecasts_1',
            df2_name='forecasts_2',
            tolerance=forecasts_1['return_forecast'].abs().mean() * 0.001,
            debug=False
        )
    ):
        LOGGER.warning(
            'Model %s (%s | %s - %s) forecasts are NOT deterministic.',
            model.model_id, model.stage, model.model_class, model.intervals
        )
        return True
    return False


# Re-created forecast consistency
def re_created_forecasts_differ_from_actual_forecasts(model):
    # Define base_data_paths
    base_dt_path = f"{Params.bucket}/data_processing/data_transformer/{model.intervals}/{model.coin_name}"

    # Load y_trans
    y_ml: pd.DataFrame = load_from_s3(
        path=f"{base_dt_path}/{model.coin_name}_y_trans.parquet"
    )

    if model.pca:
        # Load reduced X_trans_pca
        X_ml: pd.DataFrame = (
            load_from_s3(path=f"{base_dt_path}/{model.coin_name}_X_trans_pca.parquet")
            .filter(items=model.selected_features)
        )
    else:
        # Load reduced X_trans
        X_ml: pd.DataFrame = (
            load_from_s3(path=f"{base_dt_path}/{model.coin_name}_X_trans.parquet")
            .filter(items=model.selected_features)
        )
    
    # Find intersection index
    intersection = (
        y_ml.index
        .intersection(X_ml.index)
    )

    # Filter datasets
    y_ml = y_ml.loc[intersection]
    X_ml = X_ml.loc[intersection]

    # Find test idx
    test_idx = model.forecasts_df.loc[
        model.forecasts_df.index > model.last_fitting_date
    ].index.copy()

    if len(test_idx) > 0:
        # Define train & test datasets
        y_ml_train = y_ml.loc[y_ml.index.isin(model.train_idx)]
        y_ml_test = y_ml.loc[test_idx]

        X_ml_train = X_ml.loc[X_ml.index.isin(model.train_idx)]
        X_ml_test = X_ml.loc[test_idx]

        # Delete unnecessary datasets from memory
        del y_ml
        del X_ml

        # Create forecasts
        forecasts = model.return_forecast(
            train_target=y_ml_train.copy(),
            forecast_target=y_ml_test.copy(),
            train_features=X_ml_train.copy(),
            forecast_features=X_ml_test.copy(),
            forecast_dates=y_ml_test.index.copy(),
            add_bias=None,
            steps=None,
            ignore_update=True,
            max_t=None,
            raw_forecasts=False,
            debug=False
        )

        if not(
            dfs_are_equal(
                df1=forecasts,
                df2=model.forecasts_df[['return_forecast']],
                df1_name='re-created forecasts',
                df2_name='model forecasts_df',
                tolerance=model.forecasts_df['return_forecast'].abs().mean() * 0.005,
                debug=False
            )
        ):
            LOGGER.warning(
                'Re-created forecasts in Model %s (%s | %s - %s) do not coincide with actual forecasts.',
                model.model_id, model.stage, model.model_class, model.intervals
            )
            return True
    else:
        LOGGER.warning(
            'Unable to perform re_created_forecasts_differ_from_actual_forecasts on Model %s (%s | %s - %s), as expected test_idx is 0.',
            model.model_id, model.stage, model.model_class, model.intervals
        )

    return False


"""
Data Inspection & Repair
"""
def needs_repair(d: dict):
    for value in d.values():
        if value:
            return True
    return False


def find_model_diagnosis_dict(model) -> dict:
    # Define Diagnosis Dict
    return {
        'is_missing_val_table': is_missing_val_table(model=model),
        'is_missing_test_table': is_missing_test_table(model=model),
        'is_missing_optimized_table': is_missing_optimized_table(model=model),
        'table_indexes_are_inconsistent': table_indexes_are_inconsistent(model=model),
        'has_too_many_selected_features': has_too_many_selected_features(model=model),
        'has_duplicated_selected_features': has_duplicated_selected_features(model=model),
        'test_and_opt_forecasts_are_inconsistent': test_and_opt_forecasts_are_inconsistent(model=model),
        # 'forecasts_are_not_deterministic': forecasts_are_not_deterministic(model=model),
        # 're_created_forecasts_differ_from_actual_forecasts': re_created_forecasts_differ_from_actual_forecasts(model=model)
    }