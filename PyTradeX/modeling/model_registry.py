from PyTradeX.config.params import Params
from PyTradeX.utils.others.s3_helper import (
    load_from_s3,
    write_to_s3,
    find_keys,
    delete_from_s3
)
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.data_processing.data_expectations import needs_repair

from PyTradeX.modeling.model import Model
from PyTradeX.trading.trading_table import TradingTable
from tqdm import tqdm
import os
import json
from pprint import pprint, pformat
from typing import List, Dict, Tuple


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


class ModelRegistry:
    """
    Class designed to organize, manage & update model repositories in a centralized fashion.
    """
    
    def __init__(
        self, 
        n_candidates: int,
        intervals: str = Params.general_params.get("intervals")
    ) -> None:
        """
        Initialize the ModelRegistry

        :param `n_candidates`: (int) Number of candidate models that will be considered before 
         defining the production model.
        :param `intervals`: (str) Time between predictions.
        """
        # General Params
        self.intervals = intervals
        self.n_candidates = n_candidates

        # Define self.registry
        self.registry: Dict[str, List[List[str, str]]] = {
            "production": [], 
            "staging": [], 
            "development": []
        }

        # Define save_path
        self.save_path = os.path.join("models", self.intervals, "model_registry.json")

        self.load()

    def load_model(
        self, 
        model_id: str,
        model_class: str,
        light: bool = False
    ) -> Model:
        try:
            # Define load_model
            if light:
                load_model = False
            else:
                load_model = True

            # Instanciate Model
            model = Model(
                model_id=model_id,
                intervals=self.intervals,
                model_class=model_class,
                load_model=load_model
            )

            # Load light version
            if light:
                model.load(
                    pickle_files=True,
                    parquet_files=True,
                    trading_tables=True,
                    model=False
                )
            
            if isinstance(model.test_table, TradingTable) and model.test_table.empty:
                LOGGER.warning('Setting %s test_table to None.', model_id)
                model.test_table = None

            if isinstance(model.optimized_table, TradingTable) and model.optimized_table.empty:
                LOGGER.warning('Setting %s optimized_table to None.', model_id)
                model.optimized_table = None

            return model
        except Exception as e:
            LOGGER.error(
                'Unable to load Model %s %s.\n'
                'Exception: %s\n',
                model_id, model_class, e
            )
            return

    def load_dev_models(
        self, 
        light: bool = False
    ) -> List[Model]:
        # Load Staging Models
        dev_models = []

        LOGGER.info('Loading dev models:')
        for reg in tqdm(self.registry['development']):
            dev_models.append(
                self.load_model(
                    model_id=reg[0],
                    model_class=reg[1],
                    light=light
                )
            )
        
        return [model for model in dev_models if model is not None]
    
    def load_staging_models(
        self,
        light: bool = False
    ) -> List[Model]:
        # Load Staging Models
        stage_models = []

        LOGGER.info('Loading staging models:')
        for reg in tqdm(self.registry['staging']):
            stage_models.append(
                self.load_model(
                    model_id=reg[0],
                    model_class=reg[1],
                    light=light
                )
            )
        
        return [model for model in stage_models if model is not None]
    
    def load_prod_model(
        self,
        light: bool = False
    ) -> Model:
        """
        Method for the production model.

        :return: (Model) Production model.
        """
        # Find champion info
        champion_info = self.registry['production']

        LOGGER.info('Loading prod model')
        if len(champion_info) > 0:
            # Find champion reg
            champ_reg = self.registry['production'][0]

            # Load and return champion model
            return self.load_model(
                model_id=champ_reg[0],
                model_class=champ_reg[1],
                light=light
            )
        return None
    
    @staticmethod
    def sort_models(
        models: List[Model],
        trading_metric: bool = True,
        by_table: str = 'opt'
    ) -> List[Model]: 
        if by_table not in ['val', 'test', 'opt']:
            LOGGER.critical('Invalid "by_table" parameter: %s', by_table)
            raise Exception(f'Invalid "by_table" parameter: {by_table}\n\n')
        
        def sort_fun(model: Model):
            if by_table == 'val':
                # Validation TradingTable
                if trading_metric:
                    # Trading Metric
                    if model.val_table is not None and model.val_table.trading_metric is not None:
                        return model.val_table.trading_metric
                else:
                    # Tuning Metric
                    if model.val_table is not None and model.val_table.tuning_metric is not None:
                        return model.val_table.tuning_metric
                
            elif by_table == 'test':
                # Test TradingTable
                if trading_metric:
                    # Trading Metric
                    if model.test_table is not None and model.test_table.trading_metric is not None:
                        return model.test_table.trading_metric
                else:
                    # Tuning Metric
                    if model.test_table is not None and model.test_table.tuning_metric is not None:
                        return model.test_table.tuning_metric
            
            else:
                # Optimized TradingTable
                if trading_metric:
                    # Trading Metric
                    if model.optimized_table is not None and model.optimized_table.trading_metric is not None:
                        return model.optimized_table.trading_metric
                else:
                    # Tuning Metric
                    if model.optimized_table is not None and model.optimized_table.tuning_metric is not None:
                        return model.optimized_table.tuning_metric
            return 0
        
        models.sort(key=sort_fun, reverse=True)

        # if debug:
        #     print(self)

        return models

    def update_model_stages(
        self,
        update_champion: bool = False,
        debug: bool = False
    ) -> None:
        """
        Method that will re-define model stages, applying the following logic:
            - Top n_candidate dev models will be promoted as "staging" models (also referred as "challenger" models),
              based on their mean cross validation performance.
            - The top staging model will compete with the production model (also referred as "champion" model), 
              based on their test performance.
        """
        def debug_():
            print_perf = None
            if champion is not None:
                if champion.optimized_table is not None:
                    print_perf = champion.optimized_table.trading_metric
                elif champion.test_table is not None:
                    print_perf = champion.test_table.trading_metric
                else:
                    print_perf = champion.val_table.trading_metric

            print(f'MLRegistry:')
            pprint(self.registry)
            print(f'\nlen dev_models: {len(dev_models)}\n'
                  f'len staging_models: {len(staging_models)}\n'
                  f'Champion performance: {print_perf}.\n'
                  f'--------------------------------------------------------------------------\n\n')

        # Load light Models
        champion: Model = self.load_prod_model(light=False)
        staging_models: List[Model] = []
        dev_models: List[Model] = (
            self.load_staging_models(light=False) 
            + self.load_dev_models(light=False)
        )

        # Assert that all dev_models contain a val_table
        assert not(any([m.val_table is None for m in dev_models]))

        # Degrade all models to development (except for champion)
        for model in dev_models:
            # Re-asign stage
            model.stage = 'development'

        # Sort Dev Models (based on validation performance)
        dev_models = self.sort_models(
            models=dev_models,
            trading_metric=True,
            by_table='val'
        )

        # Find top n candidates
        staging_candidates = dev_models[: self.n_candidates]

        # Assert that all staging_candidates contain a test_table & a opt_table
        assert not(any([m.test_table is None or m.optimized_table is None for m in staging_candidates]))
        
        # Test & promote models from staging_candidates
        for model in staging_candidates:
            # Diagnose model
            diagnostics_dict = model.diagnose_model(debug=debug)

            if not needs_repair(diagnostics_dict):
                # Promote Model
                model.stage = 'staging'

                # Add model to staging_models
                staging_models.append(model)

                # Remove model from dev_models
                dev_models.remove(model)
            else:
                LOGGER.warning(
                    '%s was NOT pushed to Staging.\n'
                    'diagnostics_dict:\n%s\n',
                    model.model_id, pformat(diagnostics_dict)
                )

        # Sort Staging Models (based on test performance)
        staging_models = self.sort_models(
            models=staging_models,
            trading_metric=True,
            by_table='test'
        )

        # Show registry
        if debug:
            debug_()

        # Find forced model
        forced_model = None

        if debug:
            print('Foreced Model:')
            pprint(forced_model)
            print('\n\n')

        # Update Champion with forced model
        if forced_model is not None:
            LOGGER.warning('Forced model was detected: %s.', forced_model)

            # Find forced Model
            forced_model_id, forced_model_class = forced_model

            # Check if forced model is the same as current champion
            if champion is not None and forced_model_id == champion.model_id:
                print(f'Forced Model is the same as current Champion.\n')
            else:
                # Re-define old & new champion models
                new_champion = None
                for model in dev_models + staging_models:
                    if model.model_id == forced_model_id:
                        new_champion = model

                if new_champion is None:
                    LOGGER.warning('Forced Model was not found in current models!')
                else:
                    # Define old champion
                    old_champion = champion

                    # Record Previous Stage
                    prev_new_champion_stage = new_champion.stage

                    # Promote New Champion
                    new_champion.stage = 'production'                    

                    # Remove new champion from dev_models or staging_models
                    if prev_new_champion_stage == 'development':
                        dev_models.remove(new_champion)
                    elif prev_new_champion_stage == 'staging':
                        staging_models.remove(new_champion)
                    else:
                        LOGGER.critical(
                            "new_champion (%s) had an invalid stage: %s.",
                            new_champion.model_id, prev_new_champion_stage
                        )
                        raise Exception(f"new_champion ({new_champion.model_id}) had an invalid stage: {prev_new_champion_stage}.\n\n")
                    
                    # Add old champion to staging_models
                    staging_models.append(old_champion)
                    
                    # Save New Champion
                    new_champion.save(
                        pickle_files=True,
                        parquet_files=False,
                        trading_tables=False,
                        model=False
                    )

                    if old_champion is not None:
                        # Demote Current Champion
                        old_champion.stage = 'staging'
                    else:
                        LOGGER.warning('Old champion was not found!')

                    # Re-assign champion variable
                    champion = new_champion

        # Define default champion if current champion is None
        if champion is None:
            LOGGER.warning(
                'There was no previous champion.\n'
                'Therefore, a new provisory champion will be chosen.\n'
            )
            
            # Promote New Champion
            new_champion = self.sort_models(
                models=staging_models,
                trading_metric=True,
                by_table='opt'
            )[0]

            new_champion.stage = 'production'

            # Remove model from staging_models
            staging_models.remove(new_champion)

            # Save new_champion
            new_champion.save(
                pickle_files=True,
                parquet_files=False,
                trading_tables=False,
                model=False
            )

            # Re-assign champion variable
            champion = new_champion

            print(f'New champion model:')
            print(champion)
            print('\n\n')

        elif update_champion:
            # Pick Challenger
            challenger = self.sort_models(
                models=staging_models,
                trading_metric=True,
                by_table='opt'
            )[0]

            if (
                # Challenger trading metric should be greater than the champion trading metric
                challenger.optimized_table.trading_metric > champion.optimized_table.trading_metric

                # Challenger est_monthly_ret should be more than 5% better than the champion trading metric
                and challenger.optimized_table.est_monthly_ret > 1.05 * champion.optimized_table.est_monthly_ret

                # Challenger test_table trading_metric should be greater than 0
                # and challenger.test_table.trading_metric > 0
            ):
                print(f'New Champion mas found (opt performance: {challenger.optimized_table.trading_metric}):')
                print(challenger)
                print(f'Previous Champion (opt performance: {champion.optimized_table.trading_metric}):')
                print(champion)

                # Promote Challenger
                challenger.stage = 'production'

                # Remove challenger from staging_models
                staging_models.remove(challenger)

                # Demote Champion
                champion.stage = 'staging'

                # Add old champion to staging_models
                staging_models.append(champion)

                # Save New Champion
                challenger.save(
                    pickle_files=True,
                    parquet_files=False,
                    trading_tables=False,
                    model=False
                )

                # Re-assign champion variable
                champion = challenger

                print(f'New champion model:')
                print(champion)
                print('\n\n')

        """
        Save Models & Update self.registry
        """
        dev_models = self.sort_models(
            models=dev_models,
            trading_metric=True,
            by_table='val'
        )[: 5]

        # Update Dev Registry
        self.registry['development'] = [
            (m.model_id, m.model_class) for m in dev_models 
            if m.val_table.tuning_metric > 0
        ]

        # Save Dev Models
        for model in dev_models:
            assert model.stage == 'development'

            # Save Model
            model.save(
                pickle_files=True,
                parquet_files=False,
                trading_tables=False,
                model=False
            )

        # Update Staging Registry
        self.registry['staging'] = [
            (m.model_id, m.model_class) for m in staging_models 
            if m.optimized_table.trading_metric > 0 # and m.val_table.trading_metric > 0
        ]

        # Save Staging Models
        for model in staging_models:
            assert model.stage == 'staging'

            # Save Model
            model.save(
                pickle_files=True,
                parquet_files=False,
                trading_tables=False,
                model=False
            )

        # Update Production Registry
        self.registry['production'] = [(champion.model_id, champion.model_class)]
        
        # Save Production Model
        assert champion.stage == 'production'

        champion.save(
            pickle_files=True,
            parquet_files=False,
            trading_tables=False,
            model=False
        )

        if debug:
            debug_()
        
        # Clean File System
        self.clean_models()

        # Save self.registry
        self.save()

    def clean_models(self) -> None:
        keep_regs = (
            self.registry["development"] +
            self.registry["staging"] + 
            self.registry["production"]
        )
        keep_ids = [reg[0] for reg in keep_regs]

        LOGGER.info('keep_ids:\n%s\n', pformat(keep_ids))
        
        def list_files_in_subdirectory(directory):
            # List to store file names
            file_list = []
            
            # Walk the directory tree
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_list.append(os.path.join(root, file))
            
            return file_list

        # Find files
        directory_path = os.path.join("models", self.intervals)
        file_names = list_files_in_subdirectory(directory_path)

        # Clean Models
        for file_name in file_names:
            if not any([model_id in file_name for model_id in keep_ids]):
                print(f"Deleting {file_name}.\n")
                os.remove(file_name)

        def list_subdirectories(directory):
            # List to store subdirectories
            subdirectories = []
            
            # Iterate over the directory contents
            for entry in os.listdir(directory):
                # Create full path
                full_path = os.path.join(directory, entry)
                # Check if it's a directory
                if os.path.isdir(full_path):
                    subdirectories.append(full_path)
            
            return subdirectories

        # Find subdirs
        subdirs = list_subdirectories(directory_path)

        # Clean dubdirs
        for subdir in subdirs:
            if not any([model_id in subdir for model_id in keep_ids]):
                print(f"Deleting {subdir}.\n")
                os.rmdir(subdir)

    def find_repeated_models(
        self,
        new_model: Model, 
        models: List[Model] = None, 
        from_: str = None
    ) -> List[Model]:
        # Validate models
        if models is None:
            if from_ is None:
                LOGGER.critical('If "models" parameter is None, then "from_" parameter cannot be None as well.')
                raise Exception('If "models" parameter is None, then "from_" parameter cannot be None as well.\n\n')
            
            model_regs = self.registry['production'] + self.registry['staging'] + self.registry['development']
            
            if from_ == 'GFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'GFM'
                ]
            elif from_ == 'LFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'LFM'
                ]
            else:
                LOGGER.critical('"from_" parameter got an invalid value: %s (expected "GFM" or "LFM").', from_)
                raise Exception(f'"from_" parameter got an invalid value: {from_} (expected "GFM" or "LFM").\n\n')

        def extract_tuple_attrs(model: Model):
            if model is not None:
                # Define base attrs to add
                attrs = {
                    'coin_name': model.coin_name,
                    'intervals': model.intervals,
                    'lag': model.lag,
                    'algorithm': model.algorithm,
                    'method': model.method,
                    # 'pca': model.pca
                }

                # Add hyperparameters
                attrs.update(model.hyper_parameters)

                return tuple(attrs.items())
            return tuple()

        # Extract new tuple attrs
        new_tuple_attrs = extract_tuple_attrs(new_model)

        # Define repeated Models
        repeated_models: List[Model] = []
        for model in models:
            if new_tuple_attrs == extract_tuple_attrs(model):
                repeated_models.append(model)

        # Add repeated_models, by looking at model_id
        for model in models:
            if new_model.model_id == model.model_id:
                repeated_models.append(model)

        return repeated_models

    def drop_duplicate_models(
        self,
        models: List[Model] = None, 
        from_: str = None,
        trading_metric: bool = True,
        by_table: str = 'opt',
        debug: bool = False
    ):
        # Validate models
        if models is None:
            if from_ is None:
                LOGGER.critical('If "models" parameter is None, then "from_" parameter cannot be None as well.')
                raise Exception('If "models" parameter is None, then "from_" parameter cannot be None as well.\n\n')
            
            model_regs = self.registry['production'] + self.registry['staging'] + self.registry['development']
            
            if from_ == 'GFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'GFM'
                ]
            elif from_ == 'LFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'LFM'
                ]
            else:
                LOGGER.critical('"from_" parameter got an invalid value: %s (expected "GFM" or "LFM").', from_)
                raise Exception(f'"from_" parameter got an invalid value: {from_} (expected "GFM" or "LFM").\n\n')
        
        # Find repeated models
        repeated_models_dict: Dict[Model, List[Model]] = {}

        for model in models:
            # Extract idx
            model_idx = models.index(model)

            # Add repeated models
            repeated_models_dict[model] = self.find_repeated_models(
                new_model=model,
                models=[models[idx] for idx in range(len(models)) if idx != model_idx]
            )
        
        # Drop repeated models
        for model, repeated_models in repeated_models_dict.items():
            if len(repeated_models) > 0:
                LOGGER.warning('Model %s (%s | %s) has repeated models.', model.model_id, model.stage, model.model_class)

                # Sort models
                sorted_models = self.sort_models(
                    models=[model] + repeated_models,
                    trading_metric=trading_metric,
                    by_table=by_table
                )

                for drop_model in sorted_models[1:]:
                    try:
                        models.remove(drop_model)
                    except Exception as e:
                        LOGGER.warning(
                            'Unable to delete Model %s (%s | %s).\n'
                            'Exception: %s.\n',
                            drop_model.model_id, drop_model.stage, drop_model.model_class, e
                        )

        # Delete repeated_models_dict & sorted_models
        del repeated_models_dict
        try:
            del sorted_models
        except:
            pass
        
        return models

    def load(self) -> None:
        # Read registry
        try:
            with open(self.save_path, 'r') as file:
                self.registry: Dict[str, List[List[str, str]]] = json.load(file)
        except Exception as e:
            LOGGER.critical(
                'Unable to load self.registry (%s).\n'
                'Exception: %s.\n',
                self.intervals, e
            )

    def save(self) -> None:
        # Save dictionary to a JSON file
        with open(self.save_path, 'w') as file:
            json.dump(self.registry, file, indent=4)

    def __repr__(self) -> str:
        LOGGER.info('Model Registry:')

        # Prod Model
        champion = self.load_prod_model(light=True)
        if champion is not None:
            if champion.test_table is not None:
                test_score = champion.test_table.trading_metric
            else:
                test_score = None

            if champion.optimized_table is not None:
                opt_score = champion.optimized_table.trading_metric
            else:
                opt_score = None

            LOGGER.info(
                'Champion Model: %s (%s)\n'
                '    - Validation score: %s\n'
                '    - Test score: %s\n'
                '    - Optimized score: %s\n',
                champion.model_id, champion.model_class, champion.val_table.trading_metric,
                test_score, opt_score
            )
        else:
            LOGGER.warning('loaded champion is None!.')

        # Staging Models
        for model in self.load_staging_models(light=True):
            if model.test_table is not None:
                test_score = model.test_table.trading_metric
            else:
                test_score = None

            if model.optimized_table is not None:
                opt_score = model.optimized_table.trading_metric
            else:
                opt_score = None

            LOGGER.info(
                'Staging Model: %s (%s)\n'
                '    - Validation score: %s\n'
                '    - Test score: %s\n'
                '    - Optimized score: %s\n',
                model.model_id, model.model_class, model.val_table.trading_metric, 
                test_score, opt_score
            )
        
        # Dev Models
        for model in self.load_dev_models(light=True):
            LOGGER.info(
                'Dev Model: %s (%s)\n'
                '    - Validation score: %s\n',
                model.model_id, model.model_class, model.val_table.trading_metric
            )

        return '\n\n'
    