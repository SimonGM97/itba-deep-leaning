from PyTradeX.config.params import Params
from PyTradeX.data_processing.boruta_py import BorutaPy
from PyTradeX.utils.others.s3_helper import write_to_s3, load_from_s3
from PyTradeX.utils.data_processing.data_expectations import needs_repair
from PyTradeX.utils.general.logging_helper import get_logger
from PyTradeX.utils.others.timing import timing
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression, 
    mutual_info_regression, 
    chi2
)
import scipy.stats as st
from sklearn.preprocessing import OneHotEncoder
import featuretools as ft
import tsfresh as tsf
# import kxy
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from nancorrmp.nancorrmp import NaNCorrMp
import os
import time
import random
from pprint import pformat, pprint
from typing import Dict, List, Tuple, Any
from copy import deepcopy
from sklearn.exceptions import InconsistentVersionWarning
import warnings

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)


# Get logger
LOGGER = get_logger(
    name=__name__,
    level='DEBUG', # Params.log_params.get('level'),
    txt_fmt=Params.log_params.get('txt_fmt'),
    json_fmt=Params.log_params.get('json_fmt'),
    filter_lvls=Params.log_params.get('filter_lvls'),
    log_file=Params.log_params.get('log_file'),
    backup_count=Params.log_params.get('backup_count')
)


class FeatureSelector:

    algorithms = Params.ml_params.get("algorithms")
    methods = Params.ml_params.get("methods")

    load_pickle = [
        'selected_features',
        'primary_filter'
    ]
    forced_features = [
        'coin_price', 
        'coin_return', 
        'coin_acceleration', 
        'coin_jerk', 

        'manual_dummy_pred',

        'manual_trend_identifier_bull_trend',
        'manual_trend_identifier_bear_trend',
        'manual_trend_identifier_indecision',

        'manual_is_candle_type_strong_bull_candle',
        'manual_is_candle_type_strong_bear_candle',
        'manual_is_candle_type_bull_candle',
        'manual_is_candle_type_bear_candle',

        'manual_is_ha_candle_type_strong_bull_ha_candle',
        'manual_is_ha_candle_type_strong_bear_ha_candle'
    ]
    
    def __init__(
        self, 
        intervals: str = Params.general_params.get("intervals"),
        mock: bool = False,
        **data_params
    ) -> None:
        # General params
        self.intervals: str = intervals

        # Load param
        self.mock: bool = mock

        # Data params
        self.periods: int = None

        self.tf_q_thresh: float = None
        self.ff_thresh: float = None
        self.cat_perc: float = None
        self.reduce_cm_datasets: float = None

        self.rfe_best_n: int = None
        self.reg_k_best: int = None
        self.binary_k_best: int = None
        self.tsfresh_p_value: float = None
        self.tsfresh_best_n: int = None
        self.binary_target_p_value: float = None
        self.binary_target_n: int = None
        self.max_features: float = None

        default_params = [
            'periods',

            'tf_q_thresh',
            'ff_thresh',
            'cat_perc',
            'reduce_cm_datasets',
            
            'rfe_best_n',
            'reg_k_best',
            'binary_k_best',
            'tsfresh_p_value',
            'tsfresh_best_n',
            'binary_target_p_value',
            'binary_target_n',
            'max_features'
        ]
        for param in default_params:
            if param not in data_params.keys():
                data_params[param] = getattr(Params, 'data_params')[param]

            setattr(self, param, data_params[param])

        # Load features
        self.cm: pd.DataFrame = None
        self.primary_filter: Dict[str, List[str]] = None
        self.selected_features: Dict[str, List[str]] = None

        self.load(debug=False)
    
    @property
    def save_path(self) -> str:
        if self.mock:
            return f"{Params.bucket}/mock/data_processing/feature_selector/{self.intervals}/global"
        else:
            return f"{Params.bucket}/data_processing/feature_selector/{self.intervals}/global"

    def prepare_datasets(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        target_col: str = None,
        reduce_datasets: bool = False,
        # fill_other_coins_nulls: bool = False,
        fill_additional_nulls: bool = False,
        apply_primary_filter: bool = False,
        only_num_cols: bool = False,
        only_cat_cols: bool = False,
        apply_ohe: bool = False,
        correl_sort: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Find Intersection
        intersection = y.index.intersection(X.index)

        y = y.loc[intersection]
        X = X.loc[intersection]

        # Reduce datasets
        if reduce_datasets:
            def reduce_df(gb_df: pd.DataFrame):
                return gb_df.tail(int(gb_df.shape[0] * self.reduce_cm_datasets))

            # Reduce y
            y = y.groupby('coin_name').apply(reduce_df)

            # Reduce X
            X = X.groupby('coin_name').apply(reduce_df)

        # Fill other_coins nulls
        # if fill_other_coins_nulls:
        #     if 'coin_price' in X.columns:
        #         for col in [c for c in X.columns if c.startswith('other_coins') and c.endswith('_price')]:
        #             X[col] = X[col].fillna(X['coin_price'])
        #     else:
        #         LOGGER.warning('Unable to fill other_coins null values, as "coin_price" is not in X.columns.')

        # Fill the rest
        if fill_additional_nulls:
            X = X.ffill().bfill()
        
        # Drop "coin_name" columns
        y = y.drop(columns=['coin_name'], errors='ignore')
        X = X.drop(columns=['coin_name'], errors='ignore')

        # Keep features from primary_filter
        if apply_primary_filter:
            keep_features = self.primary_filter['num'] + self.primary_filter['cat']
            X = X.filter(items=keep_features)

        # Filter num_cols only
        if only_num_cols:
            num_cols = list(X.select_dtypes(include=['number']).columns)
            X = X.loc[:, num_cols]

        # Filter cat_cols only
        if only_cat_cols:
            str_cols = list(X.select_dtypes(include=['object']).columns)
            X = X.loc[:, str_cols]

        # Apply OneHotEncoding to categorical features
        if apply_ohe:
            # Instanciate OneHotEncoder
            ohe = OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore'
            )
            ohe.fit(X[self.primary_filter['cat']])

            X_str = pd.DataFrame(
                ohe.transform(X[self.primary_filter['cat']]),
                columns=ohe.get_feature_names_out(self.primary_filter['cat']),
                index=X.index
            )

            X = pd.concat([X[self.primary_filter['num']], X_str], axis=1)

        if correl_sort:
            # Calculate Correlations with target
            tf_corr_df = pd.DataFrame(columns=[target_col])
            for c in X.columns:
                tf_corr_df.loc[c] = abs(y.corr(X[c]))
            
            # tf_corr_df['avg'] = tf_corr_df.mean(axis=1)
            tf_corr_df = tf_corr_df.sort_values(by=[target_col], ascending=False)

            LOGGER.info("tf_corr_df:\n%s\n", tf_corr_df)

            X = X.loc[:, tf_corr_df.index.tolist()]

        return y, X
    
    def target_feature_correl_filter(
        self,
        target_name: str,
        y: pd.Series,
        X: pd.DataFrame,
        q_thresh: float = None,
        debug: bool = False
    ) -> pd.DataFrame:
        # Validate input
        if q_thresh is None:
            q_thresh = self.tf_q_thresh

        # Prepare DataFrames
        y, X = self.prepare_datasets(
            y=y, X=X,
            reduce_datasets=True,
            # fill_other_coins_nulls=True,
            fill_additional_nulls=True,
            apply_primary_filter=False,
            only_num_cols=True,
            only_cat_cols=False,
            apply_ohe=False,
            correl_sort=False
        )

        # Calculate Correlations with target
        tf_corr_df = pd.DataFrame(columns=[target_name])
        for c in X.columns:
            tf_corr_df.loc[c] = [abs(y[target_name].corr(X[c]))]
        
        tf_corr_df: pd.DataFrame = tf_corr_df.sort_values(by=[target_name], ascending=False)

        # Define threshold
        threshold = np.quantile(tf_corr_df[target_name].dropna(), q_thresh)

        if debug:
            LOGGER.info("threshold: %s", threshold)
            LOGGER.info("tf_corr_df.head():\n%s\n", tf_corr_df.head())
            
        # Delete DataFrames from memory
        del y
        del X
        
        return tf_corr_df.loc[tf_corr_df[target_name] > threshold].index.tolist()
    
    def colinear_feature_filter(
        self,
        y=pd.DataFrame, 
        X=pd.DataFrame,
        # cm_columns: List[str],
        thresh: float = 0.9,
        debug: bool = False
    ) -> List[str]:
        # Define selected_features
        # selected_features: List[str] = deepcopy(cm_columns)

        # Filter correlation matrix
        # cm_copy: pd.DataFrame = self.cm.loc[selected_features, selected_features]

        # if debug:
        #     print(f'cm_copy.iloc[:7, :7]:\n{cm_copy.iloc[:7, :7]}\n\n')

        # Define X
        _, X = self.prepare_datasets(
            y=y, X=X,
            reduce_datasets=True,
            # fill_other_coins_nulls=True,
            fill_additional_nulls=True,
            apply_primary_filter=False,
            only_num_cols=True,
            only_cat_cols=False,
            apply_ohe=False,
            correl_sort=False
        )

        # Find correlation matrix
        t0 = time.time()
        n_jobs = Params.cpus
        cm: pd.DataFrame = (
            NaNCorrMp
            .calculate(X, n_jobs=n_jobs)
            .abs()
            * 100
        ).fillna(100)
        LOGGER.info("cm took %s sec to be created.", int(time.time()-t0))

        # Delete y & X from memory
        del y
        del X

        # Define selected_features
        selected_features: List[str] = deepcopy(cm.columns)

        # Filter selected_features
        i = 0
        while i < len(selected_features):
            keep_feature = selected_features[i]
            skip_features = cm.loc[
                (cm[keep_feature] < 100) &
                (cm[keep_feature] >= thresh*100)
            ][keep_feature].index.tolist()
            
            if len(skip_features) > 0:
                selected_features = [c for c in selected_features if c not in skip_features]
            i += 1
        
        if debug:
            print(f'cm_copy.shape: {cm.shape}\n'
                  f'len(selected_features): {len(selected_features)}\n\n')
        
        # Delete cm_copy
        del cm
        
        return selected_features
    
    def categorical_features_filter(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        perc: float = 0.1
    ) -> List[str]:
        # Prepare DataFrames
        y, X = self.prepare_datasets(
            y=y, X=X,
            reduce_datasets=False,
            # fill_other_coins_nulls=False,
            fill_additional_nulls=True,
            apply_primary_filter=False,
            only_num_cols=False,
            only_cat_cols=True,
            apply_ohe=False,
            correl_sort=False
        )

        # Prepare target
        if y.name == 'target_return':
            binary_y = y.map(lambda x: 1 if x >=0 else 0)
            binary_y.name = 'binary_target'
        else:
            LOGGER.critical("Target should be a series containing the target_return.")
            raise Exception(f"Target should be a series containing the target_return.\n")

        # Delete y from memory
        del y
        
        if X.shape[1] > 1:
            initial_cols = X.columns.tolist().copy()

            # One-hot encode the categorical features
            OHE = OneHotEncoder()
            OHE.fit(X[X.columns.tolist()])

            X = pd.DataFrame(
                OHE.transform(X[X.columns.tolist()]).toarray(),
                columns=OHE.get_feature_names_out(X.columns.tolist()),
                index=X.index
            )
            
            # Apply SelectKBest with chi2
            k = max([1, int(perc * X.shape[1])])
            selector = SelectKBest(score_func=chi2, k=k)  # You can adjust the value of k
            selector.fit(X, binary_y)
            
            # Get the selected features
            selected_features = X.columns[selector.get_support()].tolist()

            # Delete binary_y & X from memory
            del binary_y
            del X

            return [c for c in initial_cols if any(c in c2 for c2 in selected_features)]
        else:
            LOGGER.warning("Not enough information to filter out X:\n%s\n", X.tail())
            # Delete binary_y & X from memory
            del binary_y
            del X

            return []

    @timing
    def update_primary_filter(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        debug: bool = False
    ) -> None:
        LOGGER.info("Updating self.primary_filter (%s).", self.intervals)

        # Reset self.primary_filter
        self.primary_filter: Dict[str, List[str]] = {
            'num': [],
            'cat': []
        }

        # Find target_features correlation filters
        tf_filters = {}
        for method in self.methods:
            tf_filters[f'target_{method}'] =  self.target_feature_correl_filter(
                target_name=f'target_{method}',
                y=y[[f'target_{method}', 'coin_name']].copy(),
                X=X.copy(),
                q_thresh=self.tf_q_thresh,
                debug=debug
            )

        # print(f"tf_filters:")
        seen_features = []
        for key in tf_filters.keys():
            new_features = [f for f in tf_filters[key] if f not in seen_features]
            seen_features.extend(new_features)
            LOGGER.info("len(tf_filters[%s]): %s (new_features: %s)", key, len(tf_filters[key]), len(new_features))
        
        """
        # Find X_copy columns
        X_copy_cols: List[str] = []
        for key in tf_filters:
            X_copy_cols.extend([c for c in tf_filters[key] if c not in X_copy_cols])

        # Define X_copy
        _, X_copy = self.prepare_datasets(
            y=y.copy(), 
            X=X[X_copy_cols + ['coin_name']].copy(),
            reduce_datasets=True,
            # fill_other_coins_nulls=True,
            fill_additional_nulls=True,
            apply_primary_filter=False,
            only_num_cols=True,
            only_cat_cols=False,
            apply_ohe=False,
            correl_sort=False
        )

        # Find correlation matrix
        # self.cm = pd.DataFrame(X_copy.corr().abs() * 100).fillna(100)
        t0 = time.time()
        n_jobs = Params.cpus
        self.cm: pd.DataFrame = (
            NaNCorrMp
            .calculate(X_copy, n_jobs=n_jobs)
            .abs()
            * 100
        ).fillna(100)
        print(f'self.cm took {int(time.time()-t0)} sec to be created.\n')

        if debug:
            print(f'self.cm.iloc[:7, :7]:\n{self.cm.iloc[:7, :7]}\n\n')

        # Delete X_copy from memory
        del X_copy
        """
        
        # Find colinear_features filter
        ff_filters = {}
        for method in self.methods:
            ff_filters[f'target_{method}'] = self.colinear_feature_filter(
                y=y.copy(), 
                X=X[tf_filters[f'target_{method}'] + ['coin_name']].copy(),
                # cm_columns=tf_filters[key],
                thresh=self.ff_thresh,
                debug=debug
            )

        # print(f"ff_filters:")
        seen_features = []
        for key in ff_filters.keys():
            new_features = [f for f in ff_filters[key] if f not in seen_features]
            seen_features.extend(new_features)
            LOGGER.info("len(ff_filters[%s]): %s (new_features: %s)", key, len(ff_filters[key]), len(new_features))

        # Reset self.cm from memory
        # self.cm: pd.DataFrame = None

        # Populate self.primary_filter
        for key in ff_filters.keys():
            self.primary_filter['num'].extend([c for c in ff_filters[key] if c not in self.primary_filter['num']])
            LOGGER.info('len(self.primary_filter["num"]): %s', len(self.primary_filter["num"]))

        # Find categorical features
        self.primary_filter['cat'].extend(self.categorical_features_filter(
            y=y['target_return'].copy(),
            X=X.copy(),
            perc=self.cat_perc
        ))

        LOGGER.info(
            "New primary_filter:\n"
            "len(self.primary_filter['cat']): %s\n"
            "len(self.primary_filter['num']): %s\n",
            len(self.primary_filter['cat']), 
            len(self.primary_filter['num'])
        )
    
    def find_dummy_model(
        self,
        algorithm: str
    ):
        if algorithm == 'random_forest':
            return RandomForestRegressor(
                max_depth=10,
                n_jobs=-1,
                random_state=23111997
            )
        if algorithm == 'lightgbm':
            return LGBMRegressor(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=23111997,
                importance_type='gain',
                verbose=-1
            )
        if algorithm == 'xgboost':
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return XGBRegressor(
                verbosity=0,
                use_rmm=True,
                device='cuda', # 'cpu', 'cuda' # cuda -> GPU
                nthread=-1,
                n_gpus=-1,
                random_state=23111997,
                n_jobs=-1
            )

        raise Exception(f'Invalid "algorithm" was received: {algorithm}')

    @timing
    def select_boruta_features(
        self,
        y: pd.Series,
        X: pd.DataFrame,
    ) -> List[str]:
        # Instanciate dummy model
        model = self.find_dummy_model(algorithm='lightgbm')

        # Define selected features
        selected_features: List[str] = []

        # Instanciate and fit the BorutaPy estimator
        boruta_estimator = BorutaPy(
            model,
            n_estimators='auto',
            max_iter=70,
            verbose=0, 
            random_state=23111997
        )

        boruta_estimator.fit(
            np.array(X), 
            np.array(y)
        )

        # Add selected features
        selected_features.extend([f for f in X.columns[boruta_estimator.support_].tolist() if f not in selected_features])

        # Delete model & boruta_estimator
        del model
        del boruta_estimator

        # Extract features
        return selected_features
    
    @timing
    def select_rfe_features(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        rfe_best_n: int = 150
    ) -> List[str]:
        # Instanciate dummy model
        model = self.find_dummy_model(algorithm='lightgbm')
        
        # Instanciate and fit the BorutaPy estimator
        selector = RFE(
            model, 
            n_features_to_select=rfe_best_n, 
            step=0.05
        )

        selector.fit(
            np.array(X), 
            np.array(y)
        )

        # Extract selected features
        selected_features: List[str] = X.columns[selector.support_].tolist()

        # Delete model & selector
        del model
        del selector

        # Extract features
        return selected_features

    @timing
    def select_k_best_features(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        reg_k_best: int = 200,
        binary_k_best: int = 150
    ) -> List[str]:
        # Select regression features with SelectKBest
        reg_selector = SelectKBest(
            score_func=mutual_info_regression, 
            k=reg_k_best
        )

        reg_selector.fit(X, y)

        reg_features: List[str] = X.columns[reg_selector.get_support()].tolist()

        # Select binary classification features with SelectKBest
        classif_selector = SelectKBest(
            score_func=mutual_info_regression, 
            k=binary_k_best
        )

        binary_y = y.map(lambda x: 1 if x >=0 else 0)

        classif_selector.fit(X, binary_y)

        classif_features: List[str] = X.columns[classif_selector.get_support()].tolist()

        # Prepare selected_features
        selected_features: List[str] = deepcopy(reg_features)
        selected_features.extend([f for f in classif_features if f not in selected_features])
            
        return selected_features

    @timing
    def select_tsfresh_features(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        p_value_thresh: float = 0.01,
        tsfresh_best_n: int = None
    ) -> List[str]:
        # Run TSFresh Feature Selection
        relevance_table: pd.DataFrame = tsf.feature_selection.relevance.calculate_relevance_table(
            X, y, n_jobs=1
        )

        # Sort & filter relevance_table 
        relevance_table = relevance_table[relevance_table['relevant']].sort_values("p_value")
        relevance_table = relevance_table.loc[relevance_table['p_value'] < p_value_thresh]
        
        # Define selected_features
        selected_features: List[str] = list(relevance_table["feature"].values)
        
        # Reduce features (if needed)
        if tsfresh_best_n is not None:
            # top_k = int(perc * len(selected_features))
            return selected_features[:tsfresh_best_n]
        
        return selected_features
    
    @timing
    def select_binary_target_features(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        p_value_thresh: float = 0.02,
        binary_target_n: int = None
    ) -> List[str]:
        """
        "Hypothesis Testing" Filter Method:
            - Bianary Features filter: Binary variables will be kept only if we can reject the null hypothesis that
                the variable and the target are independant.
            - Non-Binary Features filter: Non-Bianry variables will be kept if we can reject the null hypothesis that 
                the means of the distributions of the variable, grouped by target, are the same
        """
        # Start empty filtered list
        sorting_dict = {}

        # Prepare target
        binary_y = y.map(lambda x: 1 if x >=0 else 0)
        binary_y.name = 'binary_target'

        # Concatenate X & binary_y
        full_df = pd.concat([X, binary_y], axis=1)
        
        # Apply binary columns filter
        binary_cols = [col for col in X.columns if X[col].nunique() == 2]

        for col in binary_cols:
            # Crosstab DF
            crosstab_df = pd.crosstab(full_df[col], full_df['binary_target'])
            
            # Perform Hipothesis Testing
            _, p_value, _, _ = st.chi2_contingency(crosstab_df)
            
            # If we can reject the null hypothesis (i.e.: p_value < 0.05), then we will add the feature, 
            # as it probably adds important information about the target.
            if p_value <= p_value_thresh:
                sorting_dict.update({col: p_value})

        # Apply non-Binary columns filter
        non_binary_cols = [col for col in X.columns if X[col].nunique() > 2]

        for col in non_binary_cols:
            # Find Distribution to compare
            dist_1 = full_df.loc[full_df['binary_target'] == 0, col]
            dist_2 = full_df.loc[full_df['binary_target'] == 1, col]
            
            # Perform Hipothesis Testing
            _, p_value = st.ttest_ind(dist_1, dist_2, equal_var=False)
            
            # If we can reject the null hypothesis (i.e.: p_value < 0.05), then we will add the feature, 
            # as it probably adds important information about the target.
            if p_value <= p_value_thresh:
                sorting_dict.update({col: p_value})

        # Define selected_features
        selected_features: List[str] = sorted(sorting_dict.keys(), key=lambda col: sorting_dict[col])

        # Reduce features (if needed)
        if binary_target_n is not None:
            # top_k = int(perc * len(selected_features))
            return selected_features[:binary_target_n]
        
        return selected_features
    
    @timing
    def select_kxy_features(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        debug: bool = False
    ) -> List[str]:
        os.environ['KXY_API_KEY'] = 'pXdUYRKRYO9T0tLQ6jrU73oUZuwwMlOljD8Ob0T4'

        # Prepare DataFrames
        y, X = self.prepare_datasets(
            y=y, X=X,
            only_num_cols=False,
            correl_sort=False
        )

        full_df = pd.concat([X, y], axis=1)

        data_val_df = full_df.kxy.variable_selection(
            y.name,
            problem_type='regression',
            anonymize=True
        )

        filtered_cols: List[str] = data_val_df['Variable'].tolist()[1:]

        if debug:
            print(f'data_val_df: \n{data_val_df}\n\n')

        if debug:
            print(f'len(filtered_cols): {len(filtered_cols)}\n'
                  f'filtered_cols')
            pprint(filtered_cols)

        return filtered_cols

    @staticmethod
    def concatenate_features(
        *args: Tuple[List[str]],
        max_features: int = 0.05
    ) -> List[str]:
        # Define selected features
        selected_features: List[str] = []

        # Append new features
        for arg in args:
            selected_features.extend([f for f in arg if f not in selected_features])

        # Validate number of features and filter if > threshold
        if len(selected_features) > max_features:
            LOGGER.warning(
                "There are too many selected_features.\n"
                "len(selected_features): %s\n"
                "max_features: %s\n",
                len(selected_features), max_features
            )            
            selected_features = selected_features[:max_features]

        return selected_features

    @timing
    def selector_pipeline(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        target_col: str,
        rfe_best_n: int = 150,
        reg_k_best: int = 200,
        binary_k_best: int = 150,
        tsfresh_p_value: float = 0.01,
        tsfresh_best_n: int = 150,
        binary_target_p_value: float = 0.01,
        binary_target_n: int = 150,
        max_features: float | int = 0.05,
        debug: bool = False
    ) -> List[str]:
        if debug:
            LOGGER.debug(
                "Finding %s selected_features:\n"
                "Initial features: %s\n",
                y.name, len(X.columns)
            )
        
        # Set random seed
        np.random.seed(23111997)
        random.seed(23111997)
        
        # Prepare datasets
        y, X = self.prepare_datasets(
            y=y,
            X=X,
            target_col=target_col,
            reduce_datasets=False,
            # fill_other_coins_nulls=True,
            fill_additional_nulls=True,
            apply_primary_filter=True,
            only_num_cols=False,
            only_cat_cols=False,
            apply_ohe=True,
            correl_sort=True
        )

        if debug:
            LOGGER.debug(
                "y.shape: %s\n"
                "X.shape: %s\n",
                y.shape, X.shape
            )
        
        # # Select Boruta features
        # boruta_features = self.select_boruta_features(
        #     y=y, X=X
        # )

        # if debug:
        #     LOGGER.debug(
        #         "len(boruta_features): %s\n"
        #         "boruta_features (first 5):\n%s\n",
        #         len(boruta_features), pformat(boruta_features[:5])
        #     )

        # Select Recursive Feature Elimination features
        rfe_features = self.select_rfe_features(
            y=y, X=X,
            rfe_best_n=rfe_best_n
        )

        if debug:
            LOGGER.debug(
                "len(rfe_features): %s\n"
                "rfe_features (first 5):\n%s\n",
                len(rfe_features), pformat(rfe_features[:5])
            )

        # Select KXY Features (turned off)
        # kxy_features = self.select_kxy_features(
        #     y=y.copy(),
        #     X=X.copy(),
        #     debug=debug
        # )
        
        # Select K-Best features
        k_best_features = self.select_k_best_features(
            y=y, X=X,
            reg_k_best=reg_k_best,
            binary_k_best=binary_k_best
        )

        if debug:
            LOGGER.debug(
                "len(k_best_features): %s\n"
                "k_best_features (first 5):\n%s\n",
                len(k_best_features), pformat(k_best_features[:5])
            )        
        
        # Select TSFresh features
        tsfresh_features = self.select_tsfresh_features(
            y=y, X=X,
            p_value_thresh=tsfresh_p_value,
            tsfresh_best_n=tsfresh_best_n
        )

        if debug:
            LOGGER.debug(
                "len(tsfresh_features): %s\n"
                "tsfresh_features (first 5):\n%s\n",
                len(tsfresh_features), pformat(tsfresh_features[:5])
            )

        # Select Binary Target features
        binary_target_features = self.select_binary_target_features(
            y=y, X=X,
            p_value_thresh=binary_target_p_value,
            binary_target_n=binary_target_n
        )

        if debug:
            LOGGER.debug(
                "len(binary_target_features): %s\n"
                "binary_target_features (first 5):\n%s\n",
                len(binary_target_features), pformat(binary_target_features[:5])
            )

        # Concatenate selected_features & forced_features
        if isinstance(max_features, float):
            max_features = int(X.shape[0] * max_features)

        selected_features: List[str] = self.concatenate_features(
            self.forced_features,
            # boruta_features,
            rfe_features,
            # kxy_features,
            k_best_features,            
            tsfresh_features,
            binary_target_features,            
            max_features=max_features
        )
        
        LOGGER.info(
            "%s selected_features: %s\n"
            "First 5:\n%s\n",
            target_col, len(selected_features), pformat(selected_features[:5])
        )
        
        # Return selected features
        return selected_features

    @timing
    def update(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        debug: bool = False,
        **update_params
    ) -> None:
        # Set Up Update Parameters
        complete_update_params = {
            'update_primary_filter': False,
            'update_selected_features': False,
            'validate_selected_features': False,
            'save': False
        }
        for k, v in complete_update_params.items():
            if k not in update_params.keys():
                update_params[k] = v

        # Update primary_filter
        if self.primary_filter is None or update_params['update_primary_filter']:
            self.update_primary_filter(
                y=y, X=X,
                debug=True # debug
            )

            # Save primary features
            self.save()

        # Re-run self.selector_pipeline if needed
        if update_params['update_selected_features']:
            LOGGER.info("Re-setting self.selected_features:")

            self.selected_features: Dict[str, List[str]] = {
                method: [] for method in self.methods     
            }

            def fill_selected_features(method: str):
                self.selected_features[method] = self.selector_pipeline(
                    y=y[f'target_{method}'],
                    X=X,
                    target_col=f'target_{method}',
                    rfe_best_n=self.rfe_best_n,
                    reg_k_best=self.reg_k_best,
                    binary_k_best=self.binary_k_best,
                    tsfresh_p_value=self.tsfresh_p_value,
                    tsfresh_best_n=self.tsfresh_best_n,
                    binary_target_p_value=self.binary_target_p_value,
                    binary_target_n=self.binary_target_n,
                    max_features=self.max_features,
                    debug=True # debug
                )

            # with ThreadPoolExecutor(max_workers=Params.cpus) as executor:
            #     for trans in ['scale', 'pca']:
            #         for method in self.methods:
            #             selector_fun = partial(
            #                 fill_selected_features,
            #                 k1=trans,
            #                 k2=method
            #             )
            #             executor.submit(selector_fun)

            for method in self.methods:
                fill_selected_features(method=method)

            LOGGER.info(
                "new lenghts:\n%s\n",
                pformat({
                    method_: len(self.selected_features[method_])
                    for method_ in self.methods
                })
            )

            # Save selected_features
            self.save()

        # Validate selected features
        if update_params['validate_selected_features']:
            self.validate_selected_features(
                y=y,
                X=X,
                repair=True,
                debug=debug
            )

        # Save selected_features
        if update_params['save']:
            self.save()

    def diagnose_selected_features(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        prepare_datasets: bool = True,
        debug: bool = False
    ) -> Dict[str, bool]:
        # Diagnostics Dict
        diagnostics_dict = {}
        for method in self.methods:
            diagnostics_dict[f'missing_{method}_refined_data_columns'] = False
        
        # print('initial diagnostics_dict')
        # pprint(diagnostics_dict)
        # print('\n\n')
        
        if prepare_datasets:
            # Prepare datasets
            _, X = self.prepare_datasets(
                y=y,
                X=X,
                target_col=None,
                reduce_datasets=False,
                # fill_other_coins_nulls=True,
                fill_additional_nulls=True,
                apply_primary_filter=True,
                only_num_cols=False,
                only_cat_cols=False,
                apply_ohe=True,
                correl_sort=False
            )

        # Find refined_data_columns
        refined_data_columns = X.columns.tolist()

        # Find Missing Features
        for method in self.methods:
            # Check Refined Data Columns
            selected_features = self.selected_features[method].copy()

            # Features DF
            missing_features = [f for f in selected_features if f not in refined_data_columns]
            if len(missing_features) > 0:
                LOGGER.warning(
                    "There are missing selected_features features in refined_data_columns (%s):\n%s\n",
                    method, pformat(missing_features)
                )

                diagnostics_dict[f'missing_{method}_refined_data_columns'] = True
        
        if debug:
            print(f'diagnostics_dict:')
            pprint(diagnostics_dict)
            print(f'\n\n')

        return diagnostics_dict

    def validate_selected_features(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        repair: bool = True,
        debug: bool = False
    ) -> pd.DataFrame:
        # Prepare datasets
        y, X = self.prepare_datasets(
            y=y,
            X=X,
            target_col=None,
            reduce_datasets=True,
            # fill_other_coins_nulls=True,
            fill_additional_nulls=True,
            apply_primary_filter=True,
            only_num_cols=False,
            only_cat_cols=False,
            apply_ohe=True,
            correl_sort=False
        )

        # Find Diagnostics Dict
        diagnostics_dict = self.diagnose_selected_features(
            y=y,
            X=X,
            prepare_datasets=False,
            debug=debug
        )

        if needs_repair(diagnostics_dict):
            LOGGER.warning(
                "selected_features needs repair.\n"
                "diagnostics_dict:\n%s\n",
                pformat(diagnostics_dict)
            )

            if repair:
                # Find refined_data_columns
                refined_data_columns = X.columns.tolist()

                # Repair selected_features
                for method in self.methods:
                    # Repair Refined Data Features
                    if diagnostics_dict[f'missing_{method}_refined_data_columns']:
                        self.selected_features[method] = list(filter(
                            lambda f: f in refined_data_columns,
                            self.selected_features[method]
                        ))

    def save_mock_asset(
        self,
        asset: Any,
        asset_name: str,
        debug: bool = False
    ) -> None:
        if debug:
            if isinstance(asset, pd.DataFrame):
                print(f'Saving {asset_name} - [shape: {asset.shape}]')
            else:
                print(f'Saving {asset_name}')

        # Define base_path
        base_path = f"{Params.bucket}/mock/data_processing/feature_selector/{self.intervals}/global"

        # Define save_path
        if asset_name == 'selector_pipeline_input_y':
            save_path = f"{base_path}/selector_pipeline_input_y.parquet"
        elif asset_name == 'selector_pipeline_input_X':
            save_path = f"{base_path}/selector_pipeline_input_X.parquet"
        elif asset_name == 'selector_pipeline_output':
            save_path = f"{base_path}/selector_pipeline_output.pickle"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Save asset in S3
        write_to_s3(asset=asset, path=save_path, overwrite=True)
    
    def load_mock_asset(
        self,
        asset_name: str,
        re_create: bool = False,
        re_create_periods: int = None,
        debug: bool = False
    ) -> pd.DataFrame | Dict[str, List[str]]:
        # Define base_paths
        re_create_base_path = f"{Params.bucket}/data_processing/data_refiner/{self.intervals}/ADA"
        base_path = f"{Params.bucket}/mock/data_processing/feature_selector/{self.intervals}/global"

        # Define load_path
        if asset_name == 'selector_pipeline_input_y':
            if re_create:
                load_path = f"{re_create_base_path}/ADA_y.parquet"
            else:
                load_path = f"{base_path}/selector_pipeline_input_y.parquet"
        elif asset_name == 'selector_pipeline_input_X':
            if re_create:
                load_path = f"{re_create_base_path}/ADA_X.parquet"
            else:
                load_path = f"{base_path}/selector_pipeline_input_X.parquet"
        elif asset_name == 'selector_pipeline_output':
            load_path = f"{base_path}/selector_pipeline_output.pickle"
        else:
            raise Exception(f'Invalid "asset_name" parameter was received: {asset_name}.\n')
        
        # Load asset from S3
        asset = load_from_s3(path=load_path, ignore_checks=True)

        if (
            re_create
            and re_create_periods is not None 
            and isinstance(asset, pd.DataFrame)
        ):
            asset = asset.tail(re_create_periods)
            
        if debug:
            if isinstance(asset, pd.DataFrame):
                print(f'Loaded {asset_name} - [shape: {asset.shape}]')
            else:
                print(f'Loaded {asset_name}')

        return asset

    def save(
        self, 
        debug: bool = False
    ) -> None:
        """
        Step 1) Save .pickle files
        """
        pickle_attrs = {key: value for (key, value) in self.__dict__.items() if key in self.load_pickle}
        
        # S3
        write_to_s3(
            asset=pickle_attrs,
            path=f"{self.save_path}/feature_selector_attr.pickle"
        )

        if debug:
            print('Saving self.selected_features:')
            pprint(self.selected_features)
            print('\n\n')

    def load(
        self, 
        debug: bool = False
    ) -> None:
        """
        Step 1) Load selected_features
        """
        pickled_attrs = None
        try:
            # Load pickled attributes
            pickled_attrs: dict = load_from_s3(
                path=f"{self.save_path}/feature_selector_attr.pickle"
            )
            for attr_key, attr_value in pickled_attrs.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)
        except Exception as e:
            LOGGER.critical(
                'Unable to load feature_selector (%s).\n'
                'Exception: %s\n',
                self.intervals, e
            )
                    
        if debug:
            print('self.selected_features')
            pprint(self.selected_features)
            print('\n\n')
