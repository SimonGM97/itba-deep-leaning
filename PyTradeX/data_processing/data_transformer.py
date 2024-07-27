from PyTradeX.config.params import Params
from PyTradeX.data_processing.data_shifter import DataShifter
from PyTradeX.data_processing.feature_selector import FeatureSelector
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.data_processing.data_expectations import (
    find_data_diagnosis_dict,
    needs_repair,
    has_missing_rows
)
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing

from sklearn.preprocessing import (
    StandardScaler, 
    # MinMaxScaler, 
    OneHotEncoder, 
    # PowerTransformer
)
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from copy import deepcopy
from pprint import pprint, pformat
from typing import Dict, List, Any
from sklearn.exceptions import InconsistentVersionWarning
import warnings

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)


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


class DataTransformer(DataShifter): # FeatureSelector

    methods: List[str] = Params.ml_params.get("methods")
    
    load_parquet = [
        'y_trans',
        'X_trans',
        'X_trans_pca'
    ]
    load_pickle = [
        'ohe', 
        'num_transformer', 
        'trunc_parameters', 
        'pca', 
        'pca_explained_variance',

        'consistency_storage_X_trans',
        'consistency_storage_X_trans_pca'
    ]

    def __init__(
        self,
        coin_name: str,
        intervals: str = Params.general_params.get("intervals"),
        overwrite: bool = True,
        mock: bool = False,
        **data_params
    ) -> None:
        # General params
        self.coin_name: str = coin_name
        self.intervals: str = intervals

        # Load param
        self.overwrite: bool = overwrite
        self.mock: bool = mock

        # Data params
        self.periods: int = None
        self.new_n: int = None
        self.lag: int = None
        self.mbp: int = None

        # Shifter attr
        self.save_distance: int = None
        self.compare_distance: int = None
        self.shift_threshold: float = None
        self.max_shift: int = None

        default_params = [
            'periods',
            'new_n',
            'lag',
            'mbp',
            'save_distance',
            'compare_distance',
            'shift_threshold',
            'max_shift'
        ]
        for param in default_params:
            if param not in data_params.keys():
                data_params[param] = getattr(Params, 'data_params')[param]
            
            setattr(self, param, data_params[param])

        # Update periods
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < self.periods:
            # LOGGER.warning(
            #     '%s DT has less periods than required.\n'
            #     'Expected periods: %s\n'
            #     'raw_data_periods: %s\n'
            #     'Thus, self.periods will be reverted to %s.\n',
            #     self.coin_name, self.periods, raw_data_periods, raw_data_periods
            # )
            self.periods: int = raw_data_periods

        # Load parameters
        self.y_trans: pd.DataFrame = None
        self.X_trans: pd.DataFrame = None
        self.X_trans_pca: pd.DataFrame = None

        self.ohe: Dict[str, OneHotEncoder] = {
            'target': None,
            'features': None,
        }
        self.num_transformer: Dict[str, StandardScaler] = {
            'target': None, 
            'features': None
        }
        self.trunc_parameters: Dict[str, Dict] = {
            'target': {
                'stand': None,
                'no_stand': None
            },
            'features': {
                'stand': None,
                'no_stand': None
            }
        }
        self.pca: Dict[str, PCA] = {
            'target': None, 
            'features': None
        }

        self.pca_explained_variance: float = None

        # Shift related attrs
        self.consistency_storage: List[pd.DataFrame] = None
        self.abs_diff_dfs: List[pd.DataFrame] = None

        self.consistency_storage_X_trans: List[pd.DataFrame] = None
        self.consistency_storage_X_trans_pca: List[pd.DataFrame] = None

        self.load(debug=False)

    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/data_processing/data_transformer/{self.intervals}/{self.coin_name}"
        else:
            return f"{Params.bucket}/data_processing/data_transformer/{self.intervals}/{self.coin_name}"

    @staticmethod
    def filter_X(
        X: pd.DataFrame,
        selected_features: Dict[str, List[str]],
        is_transformed: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        # Define features to keep
        keep_features: List[str] = []
        for key in selected_features:
            keep_features.extend([f for f in selected_features[key] if f not in keep_features])

        # print('keep_features:')
        # for f in keep_features:
        #     print(f'    - {f}')
        # print('\n\n')

        if debug:
            print(f'Initial X.shape: {X.shape}\n')
        
        if not is_transformed:
            change_features = [c for c in keep_features if c.startswith('manual') and 'dummy_pred' not in c]
            unchanged_features = [c for c in keep_features if c not in change_features]
            change_features = [c for c in X.columns.tolist() if any(c in c2 for c2 in change_features)]
            keep_features = change_features + unchanged_features

            # print(f'change_features:')
            # pprint(change_features)
            # print('\n\n')

        # Filter X
        try:
            X = X[keep_features] # .filter(items=keep_features)
        except Exception as e:
            missing_features = [c for c in keep_features if c not in X.columns]
            LOGGER.warning(
                'Unable to filter X.\n'
                'Exception: %s\n'
                'Dummy missing features will be added.\n'
                'Missing features:\n%s\n', 
                e, pformat(missing_features)
            )
        
            # Add missing features
            X[missing_features] = 0

            # Re-filter X
            X = X[keep_features]

        if debug:
            print(f'len(keep_features): {len(keep_features)}\n'
                  f'X.shape: {X.shape}\n\n')
        
        return X

    def fit_transformers(
        self,
        df: pd.DataFrame, 
        key: str,
        selected_features: List[str] = None,
        debug: bool = False
    ) -> None:
        if key not in ['target', 'features']:
            LOGGER.critical("Incorrect %s parameter.", key)
            raise Exception(f"Incorrect {key} parameter.\n")
        
        LOGGER.info("Re-fitting %s transformers %s.", key, self.coin_name)
        # Reduce DataFrames
        # df = df.iloc[-int(self.periods * 0.4):]
        LOGGER.info('Fitting transformers with df.shape: %s', df.shape)
        
        # Define train_periods
        train_test_split = Params.ml_params.get('train_test_split')
        train_periods = int(df.shape[0] * train_test_split)
        
        # Find Columns
        str_cols = list(df.select_dtypes(include=['object']).columns)
        num_cols = list(df.select_dtypes(include=['number']).columns)

        """
        ONE HOT ENCODING
        """
        if len(str_cols) > 0:
            LOGGER.info("Re-fitting %s OneHotEncoder %s.", key, self.coin_name)
            self.ohe[key] = OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore'
            )
            self.ohe[key].fit(df[str_cols])

            df_str = pd.DataFrame(
                self.ohe[key].transform(df[str_cols]),
                columns=self.ohe[key].get_feature_names_out(str_cols),
                index=df.index
            )

            df = pd.concat([df[num_cols], df_str], axis=1)

        # if len(str_cols) > 0:
        #     df_str = pd.get_dummies(df[str_cols])
        #     df = pd.concat([df[num_cols], df_str], axis=1)

        # if key == 'features':
        #     df = self.filter_X(
        #         X=df,
        #         selected_features=selected_features,
        #         is_transformed=True,
        #         debug=debug
        #     )
        
        """
        TRUNC PARAMETERS (no stand)
        """
        features_trunc_no_stand_params = {}
        if len(num_cols) > 0:
            LOGGER.info("Re-fitting %s features_trunc_no_stand_params %s.", key, self.coin_name)
            for c in num_cols:
                mean = df.iloc[:train_periods][c].mean()
                std = df.iloc[:train_periods][c].std()

                q_high = mean + 1.96 * std
                q_low = mean - 1.96 * std

                features_trunc_no_stand_params[c] = {
                    'q_high': q_high,
                    'q_low': q_low
                }

        self.trunc_parameters[key]['no_stand'] = deepcopy(features_trunc_no_stand_params)

        """
        NUM TRANSFORMERS
        """
        # self.log_transformer['target'] = PowerTransformer()
        # self.log_transformer['target'].fit(df.iloc[:train_periods][num_cols])

        # self.min_max_transformer['target'] = MinMaxScaler()
        # self.min_max_transformer['target'].fit(df.iloc[:train_periods][num_cols])

        if len(num_cols) > 0:
            LOGGER.info("Re-fitting %s StandardScaler %s.", key, self.coin_name)

            # Instanciate StandardScaler
            scaler = StandardScaler(
                with_mean=True, 
                with_std=True
            )

            # Fit transformer on train dataset (in a non-parallel whay)
            scaler.fit(df[num_cols].head(train_periods))
            
            # Assign fitted scaler to self.num_transformer
            self.num_transformer[key] = scaler

            # Transform numerical columns
            df[num_cols] = self.num_transformer[key].transform(df[num_cols])

        """
        TRUNC PARAMETERS (stand)
        """
        features_trunc_stand_params = {}
        if len(num_cols) > 0:
            LOGGER.info("Re-fitting %s features_trunc_stand_params %s.", key, self.coin_name)
            max_reduction = 0.75
            for c in num_cols:
                mean = float(df.iloc[:train_periods][c].mean())
                std = float(df.iloc[:train_periods][c].std())

                q_high = mean + 1.96 * std
                q_low = mean - 1.96 * std

                features_trunc_stand_params[c] = {
                    'q_high': q_high,
                    'q_low': q_low
                }

                df[c] = df[c].apply(
                    lambda x: q_high + (x - q_high) * (1 - max_reduction) if x > q_high
                    else q_low - (q_low - x) * (1 - max_reduction) if x < q_low
                    else x
                )

        self.trunc_parameters[key]['stand'] = deepcopy(features_trunc_stand_params)

        """
        PCA
        """
        LOGGER.info("Re-fitting %s PCA %s.", key, self.coin_name)
        self.pca[key] = PCA(0.95)
        self.pca[key].fit(df.head(train_periods))

    def transform_data(
        self,
        df: pd.DataFrame,
        key: str,
        selected_features: List[str] = None,
        ohe: bool = False,
        scale: bool = False, 
        trunc: bool = False, 
        pca: bool = False,
        # columns_filter: list = None,
    ) -> pd.DataFrame:
        """
        SET PARAMETERS
        """
        if pca:
            ohe = True
            scale = True
            trunc = True

        """
        EXPECTED FEATURES
        """
        # Num Transformer
        if self.num_transformer[key] is not None:
            ss_features_in = list(self.num_transformer[key].feature_names_in_.copy())
            ss_features_out = list(self.num_transformer[key].get_feature_names_out(ss_features_in).copy())
        else:
            ss_features_in = []
            ss_features_out = []

        # One Hot Encoder
        if self.ohe[key] is not None:
            ohe_features_in = list(self.ohe[key].feature_names_in_.copy())
            ohe_features_out = list(self.ohe[key].get_feature_names_out(ohe_features_in).copy())
        else:
            ohe_features_in = []
            ohe_features_out = []

        # PCA
        if self.pca[key] is not None:
            pca_features_in = list(self.pca[key].feature_names_in_.copy())
        else:
            pca_features_in = []

        """
        CATEGORICAL TRANSFORMATIONS
        """
        if ohe and len(ohe_features_in) > 0:
            df_str = pd.DataFrame(
                self.ohe[key].transform(df[ohe_features_in]),
                columns=ohe_features_out,
                index=df.index
            )

            add_cols = list(set(ohe_features_out) - set(df_str.columns))
            if len(add_cols) > 0:
                LOGGER.warning(
                    '%s transforemed df required additional columns, after OHE.\n'
                    'add_cols: %s\n',
                    self.coin_name, add_cols
                )
                df_str = df_str.combine_first(pd.DataFrame(0, index=df.index, columns=ohe_features_out))
            
            remove_cols = list(set(df_str.columns) - set(ohe_features_out))
            if len(remove_cols) > 0:
                LOGGER.warning(
                    '%s transforemed df removing columns, after OHE.\n'
                    'remove_cols: %s\n',
                    self.coin_name, remove_cols
                )
                df_str.drop(columns=remove_cols, inplace=True)
            
            # df_str = pd.get_dummies(df[str_cols], drop_first=False)
            df = pd.concat([df[ss_features_in], df_str], axis=1)

            # if key == 'features':
            #     df = self.filter_X(
            #         X=df,
            #         selected_features=selected_features,
            #         is_transformed=True,
            #         debug=False
            #     )

        """
        NUMERICAL TRANSFORMATIONS
        """
        if scale and len(ss_features_in) > 0:
            parallel = False
            if parallel:
                def transform_batch(data_batch: pd.DataFrame):
                    return pd.DataFrame(
                        self.num_transformer[key].transform(data_batch),
                        columns=data_batch.columns.tolist(),
                        index=data_batch.index.tolist()
                    )

                # Transform data paralelly
                max_workers = np.max([np.min([8, Params.cpus//2]), 4])
                batch_size = (df.shape[0] // max_workers) + 1
                print(f'max_workers: {max_workers}\n'
                      f'batch_size: {batch_size}\n'
                      f'len(ss_features_in): {len(ss_features_in)}\n\n')

                # Define empty transformed_df
                transformed_df: pd.DataFrame = pd.DataFrame()
                t0 = time.time()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Define list to append results
                    concat_list: list = []

                    for i in range(0, df.shape[0], batch_size):
                        # Extract batch data to transform
                        batch = df.iloc[i:i+batch_size][ss_features_in]

                        # Define partial_transfomer function
                        partial_transformer = partial(
                            transform_batch,
                            data_batch=batch
                        )

                        # Append results to concat_list
                        concat_list.append(executor.submit(partial_transformer))

                    # Loop through completed tasks
                    for dataset in as_completed(concat_list):
                        transformed_df = (
                            pd.concat([transformed_df, dataset._result], axis=0)
                            .sort_index(ascending=True)
                        )

                # Update transformations
                df[ss_features_in] = transformed_df[ss_features_in]

                # Delete transformed_df
                del transformed_df

                t1 = time.time()
                print(f'Parallel transforming took {t1-t0} sec.\n')
            else:
                df[ss_features_out] = self.num_transformer[key].transform(df[ss_features_in])

        """
        OUTLIERS
        """
        # Remove genuinely extreme values, based on standard deviation
        if trunc:
            max_reduction = 0.75
            for c in ss_features_in:
                if scale:
                    q_high = self.trunc_parameters[key]['stand'][c]['q_high']
                    q_low = self.trunc_parameters[key]['stand'][c]['q_low']
                else:
                    q_high = self.trunc_parameters[key]['no_stand'][c]['q_high']
                    q_low = self.trunc_parameters[key]['no_stand'][c]['q_low']

                df[c] = df[c].apply(
                    lambda x: q_high + (x - q_high) * (1 - max_reduction) if x > q_high
                    else q_low - (q_low - x) * (1 - max_reduction) if x < q_low
                    else x
                )

        df.replace([np.inf, -np.inf], 0, inplace=True)

        """
        REDUCE DIMENSIONALITY
        """
        if pca and len(pca_features_in) > 0:
            df = pd.DataFrame(
                self.pca[key].transform(df[pca_features_in]), 
                index=df.index
            )
            df.columns = [f'pca_{n}' for n in range(1, df.shape[1] + 1)]

            self.pca_explained_variance = self.pca[key].explained_variance_ratio_

        """
        FEATURE COLUMNS
        """
        # if columns_filter is not None:
        #     df = df.filter(items=columns_filter)
        
        return df

    # @timing
    def transformer_pipeline(
        self,
        df: pd.DataFrame,
        df_name: str,
        selected_features: Dict[str, List[str]] = None,
        key: str = 'features',
        ohe: bool = False,
        scale: bool = False, 
        trunc: bool = False, 
        pca: bool = False,
        refit_transformers: bool = False,
        validate_data: bool = True,
        save_mock: bool = False,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        # Save mock input df
        if save_mock:
            # Save input df
            if df_name == 'y_trans':
                asset_name = 'transformer_pipeline_input_y'
            elif df_name in ['X_trans', 'X_trans_pca']:
                asset_name = 'transformer_pipeline_input_X'
            else:
                raise Exception(f'Invalid "df_name" was received: {df_name}\n')

            self.save_mock_asset(
                asset=df,
                asset_name=asset_name
            )

            # Save selected features
            self.save_mock_asset(
                asset=selected_features,
                asset_name='transformer_pipeline_input_selected_features'
            )

        # Validate Input
        if key not in ['target', 'features']:
            LOGGER.critical("Incorrect %s parameter.\n", key)
            raise Exception(f'Invalid "key" parameter: {key}.\n\n')

        # Prepare input X
        if df_name in ['X_trans', 'X_trans_pca']:            
            df = self.filter_X(
                X=df,
                selected_features=selected_features,
                is_transformed=False,
                debug=debug
            )
        
        # Refit Transfomers
        if refit_transformers:
            self.fit_transformers(
                df=df.copy(),
                key=key,
                selected_features=selected_features
            )
        
        # Transform DataFrame
        df = self.transform_data(
            df=df.copy(),
            key=key,
            selected_features=selected_features,
            ohe=ohe,
            scale=scale,
            trunc=trunc,
            pca=pca
        )

        # Prepare output X
        if df_name == 'X_trans':
            df = self.filter_X(
                X=df,
                selected_features=selected_features,
                is_transformed=True,
                debug=debug
            )

        # Validate Dataset
        if validate_data:
            df = self.validate_data(
                df=df.copy(),
                df_name=df_name,
                selected_features=selected_features,
                repair=True,
                debug=debug,
                **update_expectations
            )
        
        # Save mock input df
        if save_mock:
            # Save output df
            if df_name == 'y_trans':
                asset_name = 'transformer_pipeline_output_y_trans'
            elif df_name == 'X_trans':
                asset_name = 'transformer_pipeline_output_X_trans'
            elif df_name == 'X_trans_pca':
                asset_name = 'transformer_pipeline_output_X_trans_pca'
            else:
                raise Exception(f'Invalid "df_name" was received: {df_name}\n')

            self.save_mock_asset(
                asset=df,
                asset_name=asset_name
            )

        return df

    def _update_consistency_storage(
        self,
        reset_consistency_storage: bool = False,
        debug: bool = False
    ) -> None:
        # Update X_trans storage
        self.consistency_storage = self.consistency_storage_X_trans

        self.update_consistency_storage(
            df=self.X_trans.tail(100).copy(),
            reset_consistency_storage=reset_consistency_storage,
            debug=debug
        )

        self.consistency_storage_X_trans = deepcopy(self.consistency_storage)

        saved_idxs = [df_.index[-1] for df_ in self.consistency_storage_X_trans]
        if debug:
            LOGGER.debug('self.consistency_storage_X_trans saved_idxs:\n%s\n', pformat(saved_idxs))

        # Update X_trans_pca storage
        self.consistency_storage = self.consistency_storage_X_trans_pca

        self.update_consistency_storage(
            df=self.X_trans_pca.tail(100).copy(),
            reset_consistency_storage=reset_consistency_storage,
            debug=debug
        )

        self.consistency_storage_X_trans_pca = deepcopy(self.consistency_storage)

        saved_idxs = [df_.index[-1] for df_ in self.consistency_storage_X_trans_pca]
        if debug:
            LOGGER.debug('self.consistency_storage_X_trans_pca saved_idxs:\n%s\n', pformat(saved_idxs))

        # Re-set consistency storage
        self.consistency_storage = None

    # @timing
    def update(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        selected_features: Dict[str, List[str]],
        debug: bool = False,
        **update_params
    ) -> None:
        # Set Up Update Parameters
        complete_update_params = {
            'update_data': False,
            'rewrite_data': False,
            'update_expectations': False,
            'refit_transformers': False,
            'validate_data': False,
            'validate_transformers': False,
            'update_consistency_storage: true': False,
            'save': False
        }
        for k, v in complete_update_params.items():
            if k not in update_params.keys():
                update_params[k] = v

        # print('received features:\n')
        # for key in selected_features:
        #     print(f'len(selected_features[{key}]): {len(selected_features[key])}')
        # print('\n\n')

        # Update y
        if self.y_trans is None or update_params['rewrite_data']:
            if update_params['rewrite_data']:
                LOGGER.warning('%s (%s) self.y_trans will be re-written.', self.coin_name, self.intervals)
            else:
                LOGGER.warning(
                    '%s (%s): self.y_trans is None, thus it will be re-written.\n'
                    'self.y_trans: %s\n',
                    self.coin_name, self.intervals, self.y_trans
                )
                
            self.y_trans: pd.DataFrame = self.transformer_pipeline(
                df=y.copy(),
                df_name='y_trans',
                selected_features=selected_features,
                key='target',
                ohe=False,
                scale=False,
                trunc=True,
                pca=False,
                refit_transformers=update_params['refit_transformers'],                
                validate_data=False,
                debug=debug
            )

            LOGGER.info('self.y_trans.shape: %s', self.y_trans.shape)
        elif y.index[-1] > self.y_trans.index[-1] and update_params['update_data']:
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_periods = int((utc_now - self.y_trans.index[-1]).seconds / (60 * self.mbp)) + self.new_n
            new_y = y.iloc[-new_periods:]

            new_y_trans: pd.DataFrame = self.transformer_pipeline(
                df=new_y.copy(),
                df_name='y_trans',
                selected_features=selected_features,
                key='target',
                ohe=False,
                scale=False,
                trunc=True,
                pca=False,
                refit_transformers=False,
                validate_data=update_params['validate_data'],
                debug=debug,
                **{
                    'expected_periods': new_periods
                }
            )

            col_diff = list(set(new_y_trans.columns).symmetric_difference(set(self.y_trans.columns)))
            if len(col_diff) > 0:                
                LOGGER.warning(
                    'self.y_trans and new_y_trans have different columns (%s).\n'
                    'new_y_trans.columns: %s\n'
                    'self.y_trans.columns: %s\n'
                    'col_diff: %s\n',
                    self.coin_name, new_y_trans.columns, self.y_trans.columns, col_diff
                )

            new_cols = [c for c in new_y_trans.columns.tolist() if c in self.y_trans]

            self.y_trans = (
                self.y_trans[new_cols].iloc[:-24]
                .combine_first(new_y_trans)
                .combine_first(self.y_trans[new_cols])
                .sort_index(ascending=True)
            )

            self.y_trans = self.y_trans.loc[~self.y_trans.index.duplicated(keep='last')]

        # Update X_trans
        if self.X_trans is None or update_params['rewrite_data']:
            if update_params['rewrite_data']:
                LOGGER.warning('%s (%s) self.X_trans will be re-written.', self.coin_name, self.intervals)
            else:
                LOGGER.warning(
                    '%s (%s): self.X_trans is None, thus it will be re-written.\n'
                    'self.X_trans: %s\n', 
                    self.coin_name, self.intervals, self.X_trans
                )
                
            self.X_trans: pd.DataFrame = self.transformer_pipeline(
                df=X.copy(),
                df_name='X_trans',
                selected_features=selected_features,
                key='features',
                ohe=True,
                scale=True,
                trunc=True,
                pca=False,
                refit_transformers=update_params['refit_transformers'],
                validate_data=False,
                debug=debug
            )
            
            LOGGER.info('self.X_trans.shape: %s', self.X_trans.shape)
        elif X.index[-1] > self.X_trans.index[-1] and update_params['update_data']:
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_periods = int((utc_now - self.X_trans.index[-1]).seconds / (60 * self.mbp)) + self.new_n
            new_X = X.iloc[-new_periods:]

            new_X_trans: pd.DataFrame = self.transformer_pipeline(
                df=new_X.copy(),
                df_name='X_trans',
                selected_features=selected_features,
                key='features',
                ohe=True,
                scale=True,
                trunc=True,
                pca=False,
                refit_transformers=False,
                validate_data=update_params['validate_data'],
                debug=debug,
                **{
                    'expected_periods': new_periods
                }
            )

            col_diff = list(set(new_X_trans.columns).symmetric_difference(set(self.X_trans.columns)))
            if len(col_diff) > 0:                
                LOGGER.warning(
                    'self.X_trans and new_X_trans have different columns (%s).\n'
                    'new_X_trans.columns: %s\n'
                    'self.X_trans.columns: %s\n'
                    'col_diff: %s\n',
                    self.coin_name, new_X_trans.columns, self.X_trans.columns, col_diff
                )

            new_cols = [c for c in new_X_trans.columns.tolist() if c in self.X_trans]

            self.X_trans = (
                self.X_trans[new_cols].iloc[:-24]
                .combine_first(new_X_trans)
                .combine_first(self.X_trans[new_cols])
                .sort_index(ascending=True)
            )

            self.X_trans = self.X_trans.loc[~self.X_trans.index.duplicated(keep='last')]

        # Update X_trans_pca
        if self.X_trans_pca is None or update_params['rewrite_data']:
            if update_params['rewrite_data']:
                LOGGER.warning('%s (%s) self.X_trans_pca will be re-written.', self.coin_name, self.intervals)
            else:
                LOGGER.warning(
                    '%s (%s): self.X_trans_pca is None, thus it will be re-written.\n'
                    'self.X_trans: %s\n',
                    self.coin_name, self.intervals, self.X_trans
                )
                
            self.X_trans_pca: pd.DataFrame = self.transformer_pipeline(
                df=X.copy(),
                df_name='X_trans_pca',
                selected_features=selected_features,
                key='features',
                ohe=True,
                scale=True,
                trunc=True,
                pca=True,
                refit_transformers=False,
                validate_data=False,
                debug=debug
            )

            LOGGER.info('self.X_trans_pca.shape: %s', self.X_trans_pca.shape)
        elif X.index[-1] > self.X_trans_pca.index[-1] and update_params['update_data']:
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
            new_periods = int((utc_now - self.X_trans_pca.index[-1]).seconds / (60 * self.mbp)) + self.new_n
            new_X = X.iloc[-new_periods:]

            new_X_trans_pca: pd.DataFrame = self.transformer_pipeline(
                df=new_X.copy(),
                df_name='X_trans_pca',
                selected_features=selected_features,
                key='features',
                ohe=True,
                scale=True,
                trunc=True,
                pca=True,
                refit_transformers=False,
                validate_data=update_params['validate_data'],
                debug=debug,
                **{
                    'expected_periods': new_periods
                }
            )

            col_diff = list(set(new_X_trans_pca.columns).symmetric_difference(set(self.X_trans_pca.columns)))
            if len(col_diff) > 0:
                LOGGER.warning(
                    'self.X_trans_pca and new_X_trans_pca have different columns (%s).\n'
                    'new_X_trans_pca.columns: %s\n'
                    'self.X_trans_pca.columns: %s\n'
                    'col_diff: %s\n',
                    self.coin_name, new_X_trans_pca.columns, self.X_trans_pca.columns, col_diff
                )

            new_cols = [c for c in new_X_trans_pca.columns.tolist() if c in self.X_trans_pca]

            self.X_trans_pca = (
                self.X_trans_pca[new_cols].iloc[:-24]
                .combine_first(new_X_trans_pca)
                .combine_first(self.X_trans_pca[new_cols])
                .sort_index(ascending=True)
            )

            self.X_trans_pca = self.X_trans_pca.loc[~self.X_trans_pca.index.duplicated(keep='last')]

        # Update Expectations
        if update_params['update_expectations']:
            self.update_expectations(debug=debug)

        # Validate Data
        if update_params['validate_data']:
            # Validate "y_trans"
            self.y_trans = self.validate_data(
                df=self.y_trans.copy(),
                df_name='y_trans',
                selected_features=selected_features,
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )
            
            # Validate "X_trans"
            self.X_trans = self.validate_data(
                df=self.X_trans.copy(),
                df_name='X_trans',
                selected_features=selected_features,
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )

            # Validate "X_trans_pca"
            self.X_trans_pca = self.validate_data(
                df=self.X_trans_pca.copy(),
                df_name='X_trans_pca',
                selected_features=selected_features,
                repair=True,
                debug=debug,
                **{'expected_periods': self.periods}
            )

        # Validate Transformers
        if update_params['validate_transformers']:
            self.validate_transformers(
                selected_features=selected_features,
                debug=debug
            )

        # Update consistency_storage
        if update_params['update_consistency_storage']:
            self._update_consistency_storage(
                reset_consistency_storage=False,
                debug=debug
            )

        # Save Data Transformer
        if update_params['save']:
            self.save(debug=debug)

    def update_expectations(
        self,
        X_trans_pca: pd.DataFrame = None,
        debug: bool = False
    ) -> None:
        # Validate input data
        if X_trans_pca is None:
            X_trans_pca = self.X_trans_pca.copy()

        # Define asset_paths
        base_path = f"{Params.bucket}/data_processing/data_transformer/{self.intervals}/{self.coin_name}"
        s3_y_trans_asset_path = f"{base_path}/{self.coin_name}_y_trans.parquet"
        s3_X_trans_asset_path = f"{base_path}/{self.coin_name}_X_trans.parquet"
        s3_X_trans_pca_asset_path = f"{base_path}/{self.coin_name}_X_trans_pca.parquet"

        # Find y_trans_expected_cols
        y_trans_expected_cols: List[str] = [
            'target_price', 'target_return', 'target_acceleration', 'target_jerk'
        ]

        # Instanciate FeatureSelector
        FS = FeatureSelector(
            intervals=self.intervals,
            **Params.data_params.copy()
        )

        # Find X_trans_expected_cols        
        X_trans_expected_cols: List[str] = []
        for key in FS.selected_features.keys():
            X_trans_expected_cols.extend([f for f in FS.selected_features[key] if f not in X_trans_expected_cols])

        # Find X_trans_pca_expected_cols
        X_trans_pca_expected_cols: List[str] = X_trans_pca.columns.tolist()

        # Define y_trans_expected_schema
        y_trans_expected_schema = {
            col: 'float' for col in y_trans_expected_cols
        }

        # Define X_trans_expected_schema
        X_trans_expected_schema = {
            col: 'float' for col in X_trans_expected_cols
        }

        # Define X_trans_pca_expected_schema
        X_trans_pca_expected_schema = {
            col: 'float' for col in X_trans_pca_expected_cols
        }

        # Define y_trans_max_values_allowed
        y_trans_max_values_allowed = None

        # Define X_trans_max_values_allowed
        X_trans_max_values_allowed = None

        # Define X_trans_pca_max_values_allowed
        X_trans_pca_max_values_allowed = None

        # Define y_trans_min_values_allowed
        y_trans_min_values_allowed = None

        # Define X_trans_min_values_allowed
        X_trans_min_values_allowed = None

        # Define X_trans_pca_min_values_allowed
        X_trans_pca_min_values_allowed = None

        # Define y_trans_unique_values_allowed
        y_trans_unique_values_allowed = None

        # Define X_trans_unique_values_allowed
        X_trans_unique_values_allowed = None

        # Define X_trans_pca_unique_values_allowed
        X_trans_pca_unique_values_allowed = None

        # Define y_trans_null_perc_allowed
        y_trans_null_perc_allowed: Dict[str, float] = {
            col: 0.0 for col in y_trans_expected_cols
        }

        # Define X_trans_null_perc_allowed
        X_trans_null_perc_allowed: Dict[str, float] = {
            col: 0.0 for col in X_trans_expected_cols
        }

        # Define X_trans_pca_null_perc_allowed
        X_trans_pca_null_perc_allowed: Dict[str, float] = {
            col: 0.0 for col in X_trans_pca_expected_cols
        }

        # Define y_trans_duplicate_rows_subset
        y_trans_duplicate_rows_subset = None

        # Define X_trans_duplicate_rows_subset
        X_trans_duplicate_rows_subset = X_trans_expected_cols.copy()

        # Define X_trans_pca_duplicate_rows_subset
        X_trans_pca_duplicate_rows_subset = X_trans_pca_expected_cols.copy()

        # Expected periods
        expected_periods = Params.data_params.get("periods")
        raw_data_periods = Params.raw_data_shapes[self.intervals][self.coin_name][0]
        if raw_data_periods < expected_periods:
            expected_periods = raw_data_periods
        
        # Other Coins
        other_coins_n = Params.data_params.get("other_coins_n")
        other_coins = Params.other_coins_json[self.intervals][:other_coins_n]

        # Define y_trans expectations
        s3_y_trans_expectations = {
            "asset_path": s3_y_trans_asset_path,
            "check_new_missing_data": False,
            "check_missing_cols": True,
            "check_unexpected_cols": True,            
            "check_missing_rows": True,
            "check_null_values": True,
            "check_duplicated_idx": True,
            "check_duplicates_rows": True,
            "check_duplicated_cols": True,
            "check_max_values_allowed": True,
            "check_min_values_allowed": True,
            "check_unique_values_allowed": True,
            "check_inconsistent_prices": False,
            "check_extreme_values": False,
            "check_excess_features": False,
            "check_short_length": True,

            "expected_cols": y_trans_expected_cols,
            "expected_schema": y_trans_expected_schema,
            "max_values_allowed": y_trans_max_values_allowed,
            "min_values_allowed": y_trans_min_values_allowed,
            "unique_values_allowed": y_trans_unique_values_allowed,
            "null_perc_allowed": y_trans_null_perc_allowed,

            "duplicate_rows_subset": y_trans_duplicate_rows_subset,
            "outliers_dict": None,
            "max_features_perc": None,
            "other_coins": None,
            "expected_periods": expected_periods
        }
        
        # Define X_trans expectations
        s3_X_trans_expectations = {
            "asset_path": s3_X_trans_asset_path,
            "check_new_missing_data": False,
            "check_missing_cols": True,
            "check_unexpected_cols": False,            
            "check_missing_rows": True,
            "check_null_values": True,
            "check_duplicated_idx": True,
            "check_duplicates_rows": True,
            "check_duplicated_cols": True,
            "check_max_values_allowed": True,
            "check_min_values_allowed": True,
            "check_unique_values_allowed": True,
            "check_inconsistent_prices": False,
            "check_extreme_values": False,
            "check_excess_features": False,
            "check_short_length": True,

            "expected_cols": X_trans_expected_cols,
            "expected_schema": X_trans_expected_schema,
            "max_values_allowed": X_trans_max_values_allowed,
            "min_values_allowed": X_trans_min_values_allowed,
            "unique_values_allowed": X_trans_unique_values_allowed,
            "null_perc_allowed": X_trans_null_perc_allowed,

            "duplicate_rows_subset": X_trans_duplicate_rows_subset,
            "outliers_dict": None,
            "max_features_perc": None,
            "other_coins": other_coins,
            "expected_periods": expected_periods
        }

        # Define "X_trans_pca" expectations
        s3_X_trans_pca_expectations = {
            "asset_path": s3_X_trans_pca_asset_path,
            "check_new_missing_data": False,
            "check_missing_cols": True,
            "check_unexpected_cols": False,            
            "check_missing_rows": True,
            "check_null_values": True,
            "check_duplicated_idx": True,
            "check_duplicates_rows": True,
            "check_duplicated_cols": True,
            "check_max_values_allowed": True,
            "check_min_values_allowed": True,
            "check_unique_values_allowed": True,
            "check_inconsistent_prices": False,
            "check_extreme_values": False,
            "check_excess_features": False,
            "check_short_length": True,

            "expected_cols": X_trans_pca_expected_cols,
            "expected_schema": X_trans_pca_expected_schema,
            "max_values_allowed": X_trans_pca_max_values_allowed,
            "min_values_allowed": X_trans_pca_min_values_allowed,
            "unique_values_allowed": X_trans_pca_unique_values_allowed,
            "null_perc_allowed": X_trans_pca_null_perc_allowed,

            "duplicate_rows_subset": X_trans_pca_duplicate_rows_subset,
            "outliers_dict": None,
            "max_features_perc": None,
            "other_coins": other_coins,
            "expected_periods": expected_periods
        }

        if debug:
            LOGGER.debug('y_trans_expectations:\n%s\n', pformat(s3_y_trans_expectations))
            LOGGER.debug('X_trans_expectations:\n%s\n', pformat(s3_X_trans_expectations))
            LOGGER.debug('X_trans_pca_expectations:\n%s\n', pformat(s3_X_trans_pca_expectations))

        # Save Expectations
        s3_expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"

        write_to_s3(
            asset=s3_y_trans_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_y_trans_expectations.json"
        )
        write_to_s3(
            asset=s3_X_trans_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_X_trans_expectations.json"
        )
        write_to_s3(
            asset=s3_X_trans_pca_expectations,
            path=f"{s3_expectations_base_path}/{self.coin_name}_X_trans_pca_expectations.json"
        )

    def diagnose_data(
        self,
        df: pd.DataFrame = None,
        df_name: str = 'y_trans',
        debug: bool = False,
        **update_expectations: dict
    ) -> Dict[str, bool]:
        # Validate df_name
        if df_name not in ['y_trans', 'X_trans', 'X_trans_pca']:
            LOGGER.critical('Invalid "df_name" parameter: %s', df_name)
            raise Exception(f'Invalid "df_name" parameter: {df_name}\n\n')
        
        # Extract Datasets
        if df is None:
            if df_name == 'y_trans':
                df = self.y_trans.copy()
            elif df_name == 'X_trans':
                df = self.X_trans.copy()
            else:
                df = self.X_trans_pca.copy()

        # Find Diagnostics Dict
        diagnostics_dict = find_data_diagnosis_dict(
            df_name=df_name,
            intervals=self.intervals,
            coin_name=self.coin_name,
            df=df,
            debug=debug,
            **update_expectations
        )

        if debug:
            print(f'{self.coin_name} {df_name} diagnostics_dict:')
            pprint(diagnostics_dict)
            print(f'\n\n')
        
        return diagnostics_dict

    def diagnose_transformers(
        self,
        selected_features: List[str] = None,
        key: str = 'features',
        debug: bool = False
    ) -> Dict[str, bool]:
        def _warn(cols_a: list, cols_b: list, cols_a_name: str, cols_b_name: str):
            cols_a_not_b = set(cols_a) - set(cols_b)
            cols_b_not_a = set(cols_b) - set(cols_a)
            if len(cols_a_not_b) > 0 or len(cols_b_not_a) > 0:
                LOGGER.warning("%s and %s don't match.", cols_a_name, cols_b_name)
                if len(cols_a_not_b) > 0:
                    LOGGER.warning(
                        "Columns seen in %s and not in %s (%s):\n%s\n",
                        cols_a_name, cols_b_name, len(cols_a_not_b), 
                        pformat(list(cols_a_not_b)[:10])
                    )
                if len(cols_b_not_a) > 0:
                    LOGGER.warning(
                        "Columns seen in %s and not in %s (%s):\n%s\n",
                        cols_b_name, cols_a_name, len(cols_b_not_a), 
                        pformat(list(cols_b_not_a)[:10])
                    )

        # Validate input
        if key not in ['target', 'features']:
            LOGGER.critical('Invalid "key" parameter: %s', key)
            raise Exception(f'Invalid "key" parameter: {key}\n\n')
        
        # Load refiner expectations
        # s3_expectations_base_path = f"{Params.bucket}/utils/expectations/{self.intervals}/{self.coin_name}"
        # if key == 'target':
        #     refiner_expectations = load_from_s3(
        #         path=f"{s3_expectations_base_path}/{self.coin_name}_y_expectations.json"
        #     )
        # else:
        #     refiner_expectations = load_from_s3(
        #         path=f"{s3_expectations_base_path}/{self.coin_name}_X_expectations.json"
        #     )

        # # Extract expected_schema
        # expected_schema: dict = refiner_expectations.get('expected_schema')

        # Load transformer_df
        s3_refiner_base_path = f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/{self.coin_name}"
        if key == 'target':
            transform_df: pd.DataFrame = load_from_s3(
                path=f"{s3_refiner_base_path}/{self.coin_name}_y.parquet",
                load_reduced_dataset=True
            )
        else:
            transform_df: pd.DataFrame = load_from_s3(
                path=f"{s3_refiner_base_path}/{self.coin_name}_X.parquet",
                load_reduced_dataset=True
            )
            transform_df = self.filter_X(
                X=transform_df,
                selected_features=selected_features,
                is_transformed=False
            )

        # Define diagnostics_dict
        diagnostics_dict = {
            'inconsistent_ohe_input': False,
            'inconsistent_ohe_output': False,

            'inconsistent_ss_input': False,
            'inconsistent_ss_output': False,

            'inconsistent_pca_input': False,
            'inconsistent_pca_output': False,

            'inconsistent_relationships': False
        }
        
        # Extract Expected Columns
        ss_features_in = list(self.num_transformer[key].feature_names_in_.copy())
        ohe_features_in = []
        pca_features_in = []

        if self.ohe[key] is not None:
            ohe_features_in = list(self.ohe[key].feature_names_in_.copy())

        if self.pca[key] is not None:
            pca_features_in = list(self.pca[key].feature_names_in_.copy())
        
        ss_features_out = list(self.num_transformer[key].get_feature_names_out(ss_features_in).copy())
        ohe_features_out = []
        pca_features_out = []

        if self.ohe[key] is not None:
            ohe_features_out = list(self.ohe[key].get_feature_names_out(ohe_features_in).copy())
            
        if self.pca[key] is not None:
            pca_features_out = list(self.pca[key].get_feature_names_out(pca_features_in).copy())
        
        # Extract Actual Columns
        # str_cols_in = [col for col in expected_schema.keys() if expected_schema[col] == 'str']
        # num_cols_in = [col for col in expected_schema.keys() if expected_schema[col] == 'float']
        
        str_cols_in = list(transform_df.select_dtypes(include=['object']).columns)
        num_cols_in = list(transform_df.select_dtypes(include=['number']).columns)        

        if key == 'target':
            stand_df_cols = list(self.y_trans.columns)        
            pca_df_cols = []
        else:
            stand_df_cols = list(self.X_trans.columns)        
            pca_df_cols = list(self.X_trans_pca.columns)

        """
        OneHotEncoder Check
        """
        if self.ohe[key] is not None:
            # Input columns must coincide
            diff = set(str_cols_in).symmetric_difference(set(ohe_features_in))
            if len(diff) > 0:
                diagnostics_dict['inconsistent_ohe_input'] = True
                _warn(
                    str_cols_in, ohe_features_in, 
                    'str_cols_in', 'ohe_features_in'
                )

            # All output columns must be found in stand_df
            keep_features: List[str] = []
            for keep_features_key in selected_features:
                keep_features.extend([f for f in selected_features[keep_features_key] if f not in keep_features])

            diff = set([f for f in ohe_features_out if f in keep_features]) - set(stand_df_cols)
            if len(diff) > 0:
                diagnostics_dict['inconsistent_ohe_output'] = True
                _warn(
                    ohe_features_out, stand_df_cols, 
                    'ohe_features_out', 'stand_df_cols'
                )

        """
        StandardScaler Check
        """
        if self.num_transformer[key] is not None:
            # Input columns must coincide
            diff = set(num_cols_in).symmetric_difference(set(ss_features_in))
            if len(diff) > 0:
                diagnostics_dict['inconsistent_ss_input'] = True
                _warn(
                    num_cols_in, ss_features_in, 
                    'num_cols_in', 'ss_features_in'
                )

            # All output columns must be found in stand_df
            diff = set(ss_features_out) - set(stand_df_cols)
            if len(diff) > 0:
                diagnostics_dict['inconsistent_ss_output'] = True
                _warn(
                    ss_features_out, stand_df_cols, 
                    'ss_features_out', 'stand_df_cols'
                )

        """
        PCA Check
        """
        if self.pca[key] is not None and key != 'target':
            # Input columns must coincide
            diff = set(ss_features_out + ohe_features_out).symmetric_difference(set(pca_features_in))
            if len(diff) > 0:
                diagnostics_dict['inconsistent_pca_input'] = True
                _warn(
                    ss_features_out + ohe_features_out, pca_features_in, 
                    'ss_features_out + ohe_features_out', 'pca_features_in'
                )

            # All output columns must coincide with actual pca_dataframe
            # diff = set(pca_features_out).symmetric_difference(set(pca_df_cols))
            diff = len(pca_features_out) - len(pca_df_cols)
            if diff != 0:
                diagnostics_dict['inconsistent_pca_output'] = True
                LOGGER.warning(
                    "'pca_features_out' (%s) and 'pca_df_cols' (%s) don't match.",
                    len(pca_features_out), len(pca_df_cols)
                )

        if debug:
            print(f'{self.coin_name} transformers_diagnostics_dict:')
            pprint(diagnostics_dict)
            print(f'\n\n')

        return diagnostics_dict
    
    # @timing
    def validate_data(
        self,
        df: pd.DataFrame = None,
        df_name: str = 'y_trans',
        selected_features: List[str] = None,
        repair: bool = True,
        debug: bool = False,
        **update_expectations: dict
    ) -> pd.DataFrame:
        # Validate df_name
        if df_name not in ['y_trans', 'X_trans', 'X_trans_pca']:
            LOGGER.critical('Invalid "df_name" parameter: %s', df_name)
            raise Exception(f'Invalid "df_name" parameter: {df_name}\n\n')
        if selected_features is None:
            LOGGER.critical('"selected_features" is None.')
            raise Exception(f'"selected_features" is None.\n\n')
        
        # Extract Datasets
        if df is None:
            if df_name == 'y_trans':
                df = self.y_trans.copy()
            elif df_name == 'X_trans':
                df = self.X_trans.copy()
            else:
                df = self.X_trans_pca.copy()

        # Find Diagnostics Dict
        diagnostics_dict = self.diagnose_data(
            df=df,
            df_name=df_name,
            debug=debug,
            **update_expectations
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                "%s %s needs repair.\n"
                "diagnostics_dict:\n%s\n",
                self.coin_name, df_name, pformat(diagnostics_dict)
            )
            
            if repair:
                LOGGER.info("Repairing %s %s...", self.coin_name, df_name)
                """
                Diagnostics Dict:
                    - has_missing_new_data
                    - has_missing_columns
                    - has_unexpected_columns
                    - has_missing_rows
                    - has_null_values
                    - has_duplicated_idx
                    - has_duplicated_columns
                    - has_unexpected_negative_values
                    - has_negative_prices
                    - has_extreme_values
                    - has_excess_features
                    - has_short_length
                }
                """
                # Load DataRefiner dataset
                base_data_refiner_path = f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/{self.coin_name}"
                if df_name == 'y_trans':
                    input_df: pd.DataFrame = load_from_s3(
                        path=f"{base_data_refiner_path}/{self.coin_name}_y.parquet"
                    )
                else:
                    input_df: pd.DataFrame = load_from_s3(
                        path=f"{base_data_refiner_path}/{self.coin_name}_X.parquet"
                    )
                
                # Define Key
                if df_name == 'y_trans':
                    key = 'target'
                else:
                    key = 'features'

                # Define ohe, scale, trunc, pca
                if df_name == 'y_trans':
                    ohe = False
                    scale = False
                    trunc = True
                    pca = False
                elif df_name == 'X_trans':
                    ohe = True
                    scale = True
                    trunc = True
                    pca = False
                else:
                    ohe = True
                    scale = True
                    trunc = True
                    pca = True

                # Re-create dataset
                df = self.transformer_pipeline(
                    df=input_df,
                    df_name=df_name,
                    selected_features=selected_features,
                    key=key,
                    ohe=ohe,
                    scale=scale,
                    trunc=trunc,
                    pca=pca,
                    refit_transformers=False,
                    validate_data=False,
                    debug=False
                ).tail(self.periods)

            else:
                LOGGER.warning(
                    '%s cleaned_data needed repair, but "repair" parameter was set to False.\n'
                    'Thus, %s cleaned_data will NOT be repaired.\n',
                    self.coin_name, self.coin_name
                )
        
        return df

    # @timing
    def validate_transformers(
        self,
        selected_features: List[str] = None,
        debug: bool = False
    ) -> None:
        # Validate Traget Transformers
        target_diagnostics_dict = self.diagnose_transformers(
            selected_features=selected_features,
            key='target',
            debug=debug
        )

        # Validate Features Transformers
        features_diagnostics_dict = self.diagnose_transformers(
            selected_features=selected_features,
            key='features',
            debug=debug
        )

        # File System
        # base_data_refiner_path = os.path.join(
        #     Params.base_cwd, Params.bucket, "data_processing", "data_refiner", self.intervals
        # )

        # S3
        base_data_refiner_path = f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/{self.coin_name}"

        if needs_repair(target_diagnostics_dict):
            print(f"Repairing {self.coin_name} target transformers.\n"
                  f"target_diagnostics_dict:")
            pprint(target_diagnostics_dict)
            print('\n\n')
            
            # File System
            # y: pd.DataFrame = pd.read_parquet(
            #     os.path.join(base_data_refiner_path, f"{self.coin_name}_y.parquet")
            # )

            # S3
            y: pd.DataFrame = load_from_s3(
                path=f"{base_data_refiner_path}/{self.coin_name}_y.parquet"
            )

            self.fit_transformers(
                df=y.copy(),
                key='target',
                selected_features=selected_features,
                debug=debug
            )

        if needs_repair(features_diagnostics_dict):
            print(f"Repairing {self.coin_name} features transformers.\n"
                  f"features_diagnostics_dict:")
            pprint(features_diagnostics_dict)
            print('\n\n')
            
            # File System
            # X: pd.DataFrame = pd.read_parquet(
            #     os.path.join(base_data_refiner_path, f"{self.coin_name}_X.parquet")
            # )

            # S3
            X: pd.DataFrame = load_from_s3(
                path=f"{base_data_refiner_path}/{self.coin_name}_X.parquet"
            )

            self.fit_transformers(
                df=X.copy(),
                key='features',
                selected_features=selected_features,
                debug=debug
            )

    def save_mock_asset(
        self,
        asset: Any,
        asset_name: str
    ) -> None:
        # if isinstance(asset, pd.DataFrame):
        #     print(f'Saving {asset_name} - [shape: {asset.shape}]')
        # else:
        #     print(f'Saving {asset_name}')

        # Define base_path
        base_path = f"{Params.bucket}/mock/data_processing/data_transformer/{self.intervals}/{self.coin_name}"

        # Define save_path
        if asset_name == 'transformer_pipeline_input_selected_features':
            save_path = f"{base_path}/transformer_pipeline_input_selected_features.pickle"
        elif asset_name == 'transformer_pipeline_input_y':
            save_path = f"{base_path}/transformer_pipeline_input_y.parquet"
        elif asset_name == 'transformer_pipeline_input_X':
            save_path = f"{base_path}/transformer_pipeline_input_X.parquet"
        elif asset_name == 'transformer_pipeline_output_y_trans':
            save_path = f"{base_path}/transformer_pipeline_output_y_trans.parquet"
        elif asset_name == 'transformer_pipeline_output_X_trans':
            save_path = f"{base_path}/transformer_pipeline_output_X_trans.parquet"
        elif asset_name == 'transformer_pipeline_output_X_trans_pca':
            save_path = f"{base_path}/transformer_pipeline_output_X_trans_pca.parquet"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Write asset to S3
        write_to_s3(asset=asset, path=save_path, overwrite=True)
    
    def load_mock_asset(
        self,
        asset_name: str,
        re_create: bool = False,
        re_create_periods: int = None
    ) -> pd.DataFrame | Dict[str, List[str]]:
        # print(f'Loading {asset_name}')

        # Load transformer_pipeline_input_selected_features
        if asset_name == 'transformer_pipeline_input_selected_features':
            if re_create:
                # Load selected features from FeatureSelector
                asset = load_from_s3(
                    path=f"{Params.bucket}/data_processing/feature_selector/{self.intervals}/global/feature_selector_attr.pickle",
                    ignore_checks=True
                )['selected_features']
            else:
                # Load selected features
                asset = load_from_s3(
                    path=f"{Params.bucket}/mock/data_processing/data_transformer/{self.intervals}/{self.coin_name}/transformer_pipeline_input_selected_features.pickle",
                    ignore_checks=True
                )
        else:
            # Define base paths
            re_create_base_path = f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/{self.coin_name}"
            base_path = f"{Params.bucket}/mock/data_processing/data_transformer/{self.intervals}/{self.coin_name}"
            
            # Define transformer_pipeline_input_y load path
            if asset_name == 'transformer_pipeline_input_y':
                if re_create:
                    path = f"{re_create_base_path}/{self.coin_name}_y.parquet"
                else:
                    path = f"{base_path}/transformer_pipeline_input_y.parquet"

            # Define transformer_pipeline_input_X load path
            elif asset_name == 'transformer_pipeline_input_X':
                if re_create:
                    path = f"{re_create_base_path}/{self.coin_name}_X.parquet"
                else:
                    path = f"{base_path}/transformer_pipeline_input_X.parquet"
            
            # Define transformer_pipeline_output_y_trans load path
            elif asset_name == 'transformer_pipeline_output_y_trans':
                path = f"{base_path}/transformer_pipeline_output_y_trans.parquet"
            
            # Define transformer_pipeline_output_X_trans load path
            elif asset_name == 'transformer_pipeline_output_X_trans':
                path = f"{base_path}/transformer_pipeline_output_X_trans.parquet"

            # Define transformer_pipeline_output_X_trans_pca load path
            elif asset_name == 'transformer_pipeline_output_X_trans_pca':
                path = f"{base_path}/transformer_pipeline_output_X_trans_pca.parquet"
            
            else:
                raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
            # Load asset from S3
            asset = load_from_s3(path=path, ignore_checks=True)

            if (
                re_create
                and re_create_periods is not None 
                and isinstance(asset, pd.DataFrame)
            ):
                asset = asset.tail(re_create_periods)
        
        # if isinstance(asset, pd.DataFrame):
        #     print(f'Loaded {asset_name} - [shape: {asset.shape}]')

        return asset

    def save(
        self, 
        debug: bool = False
    ) -> None:
        """
        Save .pickle files
        """
        pickle_attrs = {key: value for (key, value) in self.__dict__.items() if key in self.load_pickle}

        # Write pickled attributes
        write_to_s3(
            asset=pickle_attrs,
            path=f"{self.save_path}/{self.coin_name}_data_transformer_attr.pickle"
        )
        
        if debug:
            for attr_key, attr_value in pickle_attrs.items():            
                print(f'Saved pickle {attr_key}:')
                pprint(attr_value)
                print('\n')

        """
        Step 2) Save .parquet files
        """
        for attr_name in self.load_parquet:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                # Write parquet files
                write_to_s3(
                    asset=df,
                    path=f"{self.save_path}/{self.coin_name}_{attr_name}.parquet",
                    overwrite=self.overwrite
                )
            else:
                LOGGER.warning('%s (%s) is None!', attr_name, self.coin_name)

    def load(
        self, 
        debug: bool = False
    ) -> None:
        """
        Load .pickle files
        """
        pickle_attrs = None
        try:
            # Load pickled attributes
            pickle_attrs: dict = load_from_s3(
                path=f"{self.save_path}/{self.coin_name}_data_transformer_attr.pickle"
            )

            for attr_key, attr_value in pickle_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

                    if debug:
                        print(f'Loaded pickle {attr_key}:')
                        pprint(attr_value)
                        print('\n')
        except Exception as e:
            LOGGER.critical(
                'Unable to load data_transformer (%s: %s).\n'
                'Exception: %s\n',
                self.coin_name, self.intervals, e
            )

        """
        Load .parquet files
        """
        if self.overwrite:
            load_reduced_dataset = False
        else:
            load_reduced_dataset = True

        for attr_name in self.load_parquet:
            try:
                # Find periods to keep
                keep_periods = self.periods
                if attr_name in ['X_trans', 'X_trans_pca']:
                    keep_periods += self.lag

                # Load parquet attributes
                setattr(self, attr_name, load_from_s3(
                    path=f"{self.save_path}/{self.coin_name}_{attr_name}.parquet",
                    load_reduced_dataset=load_reduced_dataset
                ).iloc[-keep_periods:])
            except Exception as e:
                LOGGER.critical(
                    'Unable to load %s_%s.parquet.\n'
                    'Exception: %s\n',
                    self.coin_name, attr_name, e
                )
        
        # Update periods if required
        if load_reduced_dataset:
            self.periods = self.y_trans.shape[0]


# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer

# # Create a StandardScaler object
# scaler = StandardScaler()

# # Fit the StandardScaler object to the data
# scaler.fit(X_train)

# # Create a ColumnTransformer object
# column_transformer = ColumnTransformer([
#     ('scaler', scaler, [0, 1]),
# ], remainder='passthrough')