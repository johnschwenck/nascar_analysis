# ============================================
#            UTILITY FUNCTIONS
# ============================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

import xgboost as xgb
from xgboost import XGBRegressor

# from logger_config import setup_logger

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_data(config):
    path = config.get("data_path")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)

# def setup_logging(level='INFO', format_str='%(asctime)s - %(levelname)s - %(message)s'):
#     logging.basicConfig(level=getattr(logging, level), format=format_str)
#     return logging.getLogger("RacingMLPipeline")

def prepare_features(df, predictors, inflation_predictors=None, logger=None):
    
    def process(X):
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        return X.astype(float).fillna(0)
    
    if inflation_predictors is not None:
        X_count = process(df[predictors])
        X_inflate = process(df[inflation_predictors])
        if logger:
            logger.info(f"X_count shape: {X_count.shape}, X_inflate shape: {X_inflate.shape}")
        return X_count, X_inflate
    
    else:
        X = process(df[predictors])
        if logger:
            logger.info(f"X shape: {X.shape}")
        return X
    
def scale_and_add_constant(X_train, X_test, scaler, logger=None):
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_const = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_const = sm.add_constant(X_test_scaled, has_constant='add')
    
    if logger:
        logger.info(f"X_train_const shape: {X_train_const.shape}, X_test_const shape: {X_test_const.shape}")
    
    return X_train_const, X_test_const

def compute_metrics(model_type, y_train, y_train_pred, y_test, y_test_pred, logger=None):
    
    metrics = {
        'train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_MAE': mean_absolute_error(y_train, y_train_pred),
        'test_MAE': mean_absolute_error(y_test, y_test_pred),
        'train_R2': r2_score(y_train, y_train_pred),
        'test_R2': r2_score(y_test, y_test_pred)
    }

    if logger:
        logger.info(f"\nPerformance Metrics for {model_type}:\n{metrics}\n")
    return metrics

def get_enabled_model_types(config, logger = None):
    """Returns a list of enabled model types based on config."""
    model_types = []
    if config.get('models', {}).get('ols', False):
        model_types.append('ols')
    if config.get('models', {}).get('poisson', False):
        model_types.append('poisson')
    if config.get('models', {}).get('neg_binomial', False):
        model_types.append('negbin')
    if config.get('models', {}).get('zero_inflated_poisson', False):
        model_types.append('zip')
    if config.get('models', {}).get('zero_inflated_neg_binomial', False):
        model_types.append('zinb')

    # Add ML models if hyperparameters are specified
    if 'xgboost' in config.get('hyperparameters', {}):
        model_types.append('xgboost')
    if 'gbm' in config.get('hyperparameters', {}):
        model_types.append('gbm')

    logger.info(f"Model Types to be run:\n{model_types}")
    return model_types
    
def save_model(model, model_type, model_id, logger=None):

    model_dir = os.path.join('models', model_type, model_id)
    os.makedirs(model_dir, exist_ok=True)

    model_file = os.path.join(model_dir, f"{model_id}.pkl")
    joblib.dump(model, model_file)
    
    if logger:
        logger.info(f"Saved model to {model_file}")
    
    if model_type == 'xgboost':
        try:
            model_json_file = os.path.join(model_dir, f"{model_id}.json")
            model.save_model(model_json_file)
            if logger:
                logger.info(f"Saved XGBoost model in JSON format to {model_json_file}")
        except AttributeError:
            if logger:
                logger.warning(f"Could not save {model_id} as JSON.")

def load_model(model_id, logger=None):

    model_type = model_id.split('_')[0]
    model_file = os.path.join('models', model_type, model_id, f"{model_id}.pkl")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
    model = joblib.load(model_file)
    
    if logger:
        logger.info(f"Loaded model from {model_file}")

    return model

def fit_and_evaluate_single_model(model_type, data_parts, cv_folds, config, random_seed = 42, logger=None):
        """
        Fit a single model and evaluate its performance.

        Returns:
            dict: Performance metrics.
        """
        is_zip_model = model_type in ['zip', 'zinb']

        # Unpack data
        if is_zip_model:
            (X_count_train, X_count_test, X_count_train_const, X_count_test_const,
            X_inflate_train, X_inflate_test, X_inflate_train_const, X_inflate_test_const,
            y_train, y_test) = data_parts
        else:
            X_train, X_test, X_train_const, X_test_const, y_train, y_test = data_parts

        logger.info(f"\n\nFitting {model_type} model...\n\n")

        if model_type == 'ols':
            model = sm.OLS(y_train, X_train_const).fit()
            y_train_pred = model.predict(X_train_const)
            y_test_pred = model.predict(X_test_const)

        elif model_type == 'poisson':
            model = sm.GLM(y_train, X_train_const, family=sm.families.Poisson()).fit()
            y_train_pred = model.predict(X_train_const)
            y_test_pred = model.predict(X_test_const)

        elif model_type == 'negbin':
            model = sm.GLM(y_train, X_train_const, family=sm.families.NegativeBinomial()).fit()
            y_train_pred = model.predict(X_train_const)
            y_test_pred = model.predict(X_test_const)

        elif model_type == 'zip':
            model = ZeroInflatedPoisson(endog=y_train,
                                        exog=X_count_train_const,
                                        exog_infl=X_inflate_train_const,
                                        inflation='logit').fit(maxiter=500)
            y_train_pred = model.predict(X_count_train_const, exog_infl=X_inflate_train_const)
            y_test_pred = model.predict(X_count_test_const, exog_infl=X_inflate_test_const)

        elif model_type == 'zinb':
            model = ZeroInflatedNegativeBinomialP(endog=y_train,
                                                exog=X_count_train_const,
                                                exog_infl=X_inflate_train_const,
                                                inflation='logit').fit(maxiter=500)
            y_train_pred = model.predict(X_count_train_const, exog_infl=X_inflate_train_const)
            y_test_pred = model.predict(X_count_test_const, exog_infl=X_inflate_test_const)

        elif model_type == 'xgboost':
            model, y_train_pred, y_test_pred = fit_ml_model(
                xgb.XGBRegressor(objective='reg:squarederror', random_state=random_seed),
                X_train, X_test, y_train, y_test,
                config['hyperparameters']['xgboost']['param_grid'],
                model_type,
                logger = logger
            )

        elif model_type == 'gbm':
            model, y_train_pred, y_test_pred = fit_ml_model(
                GradientBoostingRegressor(random_state=random_seed),
                X_train, X_test, y_train, y_test,
                config['hyperparameters']['gbm']['param_grid'],
                model_type,
                logger=logger
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Compute metrics
        metrics = compute_metrics(model_type, y_train, y_train_pred, y_test, y_test_pred, logger = logger)

        # Perform CV if enabled
        if cv_folds:
            cv_metrics = cross_validate(model_type, model, data_parts, cv_folds, random_seed = random_seed, logger = logger)
            metrics.update(cv_metrics)
        else:
            metrics['CV_RMSE_Mean'] = None
            metrics['CV_RMSE_Std'] = None


        return metrics, model

# def fit_and_evaluate_single_model(model_type, data_parts, cv_folds, config, random_seed, logger):
#     # Load hyperparameter settings from config
#     hp_config = config.get('hyperparameters', {}).get(model_type, {})
#     search_type = hp_config.get('search_type', 'grid')
#     n_iter = hp_config.get('n_iter', 50) if search_type == 'randomized' else None
#     param_grid = hp_config.get('param_dist', hp_config.get('param_grid', {}))
#     early_stopping_rounds = hp_config.get('early_stopping_rounds', None)
#     n_jobs = hp_config.get('n_jobs', 2)

#     # Unpack data_parts based on model type
#     is_zero_inflated = model_type in ['zero_inflated_poisson', 'zero_inflated_neg_binomial', 'zip', 'zinb']
#     if is_zero_inflated:
#         if len(data_parts) != 10:
#             raise ValueError(f"Expected 10 data parts for {model_type}, got {len(data_parts)}")
#         X_count_train, X_count_test, X_count_train_const, X_count_test_const, \
#         X_inflate_train, X_inflate_test, X_inflate_train_const, X_inflate_test_const, y_train, y_test = data_parts
#     else:
#         if len(data_parts) != 6:
#             raise ValueError(f"Expected 6 data parts for {model_type}, got {len(data_parts)}")
#         X_train, X_test, X_train_const, X_test_const, y_train, y_test = data_parts

#     # Convert to DataFrames if needed (assuming NumPy arrays from data_partitioning)
#     all_columns = list(config['predictors']['all_predictors'])
#     if is_zero_inflated:
#         count_indices = [all_columns.index(col) for col in config['predictors']['count_predictors'] if col in all_columns]
#         inflation_indices = [all_columns.index(col) for col in config['predictors']['inflation_predictors'] if col in all_columns]
#         X_count_train_df = pd.DataFrame(X_count_train[:, count_indices], columns=config['predictors']['count_predictors'])
#         X_inflate_train_df = pd.DataFrame(X_inflate_train[:, inflation_indices], columns=config['predictors']['inflation_predictors'])
#         X_count_test_df = pd.DataFrame(X_count_test[:, count_indices], columns=config['predictors']['count_predictors'])
#         X_inflate_test_df = pd.DataFrame(X_inflate_test[:, inflation_indices], columns=config['predictors']['inflation_predictors'])
#     else:
#         indices = [all_columns.index(col) for col in config['predictors']['all_predictors'] if col in all_columns]
#         X_train_df = pd.DataFrame(X_train[:, indices], columns=config['predictors']['all_predictors'])
#         X_test_df = pd.DataFrame(X_test[:, indices], columns=config['predictors']['all_predictors'])

#     # Define model
#     if model_type == 'ols':
#         model = Ridge(random_state=random_seed)
#     elif model_type == 'poisson':
#         model = PoissonRegressor()
#     elif model_type == 'negbin' or model_type == 'neg_binomial':
#         model = NegativeBinomial(y_train, X_train_const)  # Use constants for statsmodels
#     elif model_type == 'zero_inflated_poisson' or model_type == 'zip':
#         model = ZeroInflatedPoisson(y_train, X_count_train_const, exog_infl=X_inflate_train_const)
#     elif model_type == 'xgboost':
#         model = XGBRegressor(random_state=random_seed)
#     elif model_type == 'gbm':
#         model = GradientBoostingRegressor(random_state=random_seed)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")

#     # Perform hyperparameter search
#     if param_grid and model_type in ['ols', 'poisson', 'xgboost', 'gbm']:
#         if search_type == 'randomized':
#             logger.info(f"Running RandomizedSearchCV for {model_type} with param_dist: {param_grid}")
#             search = RandomizedSearchCV(
#                 estimator=model,
#                 param_distributions=param_grid,
#                 n_iter=n_iter,
#                 cv=cv_folds,
#                 n_jobs=n_jobs,
#                 scoring='neg_mean_squared_error',
#                 random_state=random_seed,
#                 verbose=1
#             )
#         else:
#             logger.info(f"Running GridSearchCV for {model_type} with param_grid: {param_grid}")
#             search = GridSearchCV(
#                 estimator=model,
#                 param_grid=param_grid,
#                 cv=cv_folds,
#                 n_jobs=n_jobs,
#                 scoring='neg_mean_squared_error',
#                 verbose=1
#             )

#         # Fit with early stopping if specified
#         if early_stopping_rounds and model_type in ['xgboost', 'gbm']:
#             X_train_split, X_val, y_train_split, y_val = train_test_split(
#                 X_train_df if is_zero_inflated else X_train, y_train, test_size=0.2, random_state=random_seed
#             )
#             search.fit(
#                 X_train_split, y_train_split,
#                 eval_set=[(X_val, y_val)],
#                 early_stopping_rounds=early_stopping_rounds,
#                 verbose=False
#             )
#         else:
#             search.fit(X_train_df if is_zero_inflated else X_train, y_train)

#         best_model = search.best_estimator_
#         best_params = search.best_params_
#         logger.info(f"Best parameters for {model_type}: {best_params}")
#     elif model_type in ['negbin', 'zero_inflated_poisson', 'neg_binomial', 'zip'] and param_grid:
#         # Manual grid search for statsmodels models
#         best_score = float('inf')
#         best_model = None
#         best_params = None

#         from itertools import product
#         param_combinations = list(product(*[v for k, v in param_grid.items()]))
#         param_names = list(param_grid.keys())

#         for params in param_combinations:
#             param_dict = dict(zip(param_names, params))
#             logger.info(f"Testing parameters for {model_type}: {param_dict}")

#             if model_type in ['negbin', 'neg_binomial']:
#                 model_instance = NegativeBinomial(y_train, X_train_const).fit_regularized(
#                     method='elastic_net',
#                     alpha=param_dict.get('alpha', 0.1),
#                     l1_ratio=param_dict.get('l1_ratio', 0.5),
#                     disp=False
#                 )
#             elif model_type in ['zero_inflated_poisson', 'zip']:
#                 model_instance = ZeroInflatedPoisson(y_train, X_count_train_const, exog_infl=X_inflate_train_const).fit_regularized(
#                     method='elastic_net',
#                     l1_ratio=param_dict.get('l1_ratio', 0.5),
#                     disp=False
#                 )

#             # Manual cross-validation
#             from sklearn.model_selection import KFold
#             kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
#             scores = []
#             for train_idx, val_idx in kf.split(X_train_df if is_zero_inflated else X_train):
#                 if model_type in ['zero_inflated_poisson', 'zip']:
#                     X_count_train_cv = X_count_train_const[train_idx]
#                     X_inflation_train_cv = X_inflate_train_const[train_idx]
#                     y_train_cv = y_train.iloc[train_idx]
#                     X_count_val_cv = X_count_train_const[val_idx]
#                     X_inflation_val_cv = X_inflate_train_const[val_idx]
#                     y_val_cv = y_train.iloc[val_idx]
#                     y_pred_cv = model_instance.predict(X_count_val_cv, exog_infl=X_inflation_val_cv)
#                 else:
#                     X_train_cv = X_train_const[train_idx] if not is_zero_inflated else X_train_df.iloc[train_idx]
#                     y_train_cv = y_train.iloc[train_idx]
#                     X_val_cv = X_train_const[val_idx] if not is_zero_inflated else X_train_df.iloc[val_idx]
#                     y_val_cv = y_train.iloc[val_idx]
#                     y_pred_cv = model_instance.predict(X_val_cv)

#                 score = mean_squared_error(y_val_cv, y_pred_cv, squared=False)
#                 scores.append(score)

#             avg_score = np.mean(scores)
#             if avg_score < best_score:
#                 best_score = avg_score
#                 best_model = model_instance
#                 best_params = param_dict

#         logger.info(f"Best parameters for {model_type}: {best_params}")
#     else:
#         # Fit without grid search
#         if model_type in ['negbin', 'neg_binomial']:
#             best_model = NegativeBinomial(y_train, X_train_const).fit(disp=False)
#         elif model_type in ['zero_inflated_poisson', 'zip']:
#             best_model = ZeroInflatedPoisson(y_train, X_count_train_const, exog_infl=X_inflate_train_const).fit(disp=False)
#         else:
#             best_model = model.fit(X_train_df if is_zero_inflated else X_train, y_train)
#         best_params = {}

#     # Evaluate model
#     if model_type in ['zero_inflated_poisson', 'zip']:
#         y_pred = best_model.predict(X_count_test_const, exog_infl=X_inflate_test_const)
#         y_train_pred = best_model.predict(X_count_train_const, exog_infl=X_inflate_train_const)
#     else:
#         y_pred = best_model.predict(X_test_const if model_type in ['ols', 'poisson', 'negbin', 'neg_binomial'] else X_test_df)
#         y_train_pred = best_model.predict(X_train_const if model_type in ['ols', 'poisson', 'negbin', 'neg_binomial'] else X_train_df)

#     metrics = {
#         'train_RMSE': mean_squared_error(y_train, y_train_pred, squared=False),
#         'test_RMSE': mean_squared_error(y_test, y_pred, squared=False),
#         'train_MAE': mean_absolute_error(y_train, y_train_pred),
#         'test_MAE': mean_absolute_error(y_test, y_pred),
#         'train_R2': r2_score(y_train, y_train_pred),
#         'test_R2': r2_score(y_test, y_pred)
#     }

#     return metrics, best_model

def fit_ml_model(base_model, X_train, X_test, y_train, y_test, param_grid, model_type, logger=None):
    """
    Fits a machine learning model with optional hyperparameter tuning via GridSearchCV.

    Returns:
        model, y_train_pred, y_test_pred
    """
    if param_grid:
        logger.info(f"Running GridSearchCV for {model_type} with params: {param_grid}")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
    else:
        model = base_model
        model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return model, y_train_pred, y_test_pred

def cross_validate(model_type, model, data_parts, cv_folds, random_seed = 42, logger=None):
    """
    Runs cross-validation and returns aggregated metrics.
    """
    logger.info(f"\nRunning {cv_folds}-fold cross-validation for {model_type}...\n")

    if model_type in ['ols', 'poisson', 'negbin']:
        X_train, X_test, X_train_const, X_test_const, y_train, y_test = data_parts
        X = np.vstack([X_train_const, X_test_const])
        y = np.concatenate([y_train, y_test])
        model_func = lambda endog, exog: sm.OLS(endog, exog).fit()

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        rmse_scores = []
        for train_index, test_index in kf.split(X):
            result = model_func(y[train_index], X[train_index])
            preds = result.predict(X[test_index])
            rmse = np.sqrt(mean_squared_error(y[test_index], preds))
            rmse_scores.append(rmse)

        return {
            'CV_RMSE_Mean': np.mean(rmse_scores),
            'CV_RMSE_Std': np.std(rmse_scores)
        }

    elif model_type in ['xgboost', 'gbm']:
        X_train, X_test, _, _, y_train, y_test = data_parts
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])

        scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)

        return {
            'CV_RMSE_Mean': np.mean(rmse_scores),
            'CV_RMSE_Std': np.std(rmse_scores)
        }

    else:
        logger.warning(f"CV not implemented for model type: {model_type}")
        return {}
    
def save_performance_metrics(metrics_list, config, rd_thresh = 6, logger = None):
    """
    Saves all model performance metrics to leaderboard CSV.
    """
    leaderboard_file = os.path.join('models', config['output']['leaderboard_csv'])
    df = pd.DataFrame(metrics_list)

    # Create directory if needed
    os.makedirs(os.path.dirname(leaderboard_file), exist_ok=True)

    # Columns to compare (exclude IDs and timestamp)
    comparison_columns = [
        'train_RMSE', 'test_RMSE',
        'train_MAE', 'test_MAE',
        'train_R2', 'test_R2',
        'CV_RMSE_Mean', 'CV_RMSE_Std'
    ]

    # Round metrics to avoid floating point precision issues
    df[comparison_columns] = df[comparison_columns].round(rd_thresh)

    # Standardize model_type
    df['model_type'] = df['model_type'].str.strip().str.lower()

    # Check if the leaderboard file exists
    if os.path.exists(leaderboard_file):

        logger.info(f'Leaderboard found at: {leaderboard_file}')
        # Read the existing leaderboard
        existing_df = pd.read_csv(leaderboard_file)

        # Round and standardize existing data too!
        existing_df[comparison_columns] = existing_df[comparison_columns].round(rd_thresh)
        existing_df['model_type'] = existing_df['model_type'].str.strip().str.lower()

        # Combine old + new results
        combined_df = pd.concat([existing_df, df], ignore_index=True)

    else:
        # If no leaderboard exists, use new data
        logger.info('No leaderboard found. Generating new file...')
        combined_df = df

    deduped_df = (
        combined_df
        .sort_values(by = 'timestamp', ascending = False)
        .drop_duplicates(subset = ['model_type'] + comparison_columns, keep = 'first')
        # .replace(-9999, np.nan)
    )

    # Log skipped rows
    skipped_count = len(combined_df) - len(deduped_df)
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} duplicate model entries based on identical metrics.")

    # Drop any unwanted columns (if needed)
    deduped_df.drop(columns=['model'], errors='ignore', inplace=True)

    # Save updated leaderboard
    deduped_df['train_test_diff'] = deduped_df['train_RMSE'] - deduped_df['test_RMSE'] # TODO: make dynamic - update config: include metric, but dont specify train/test i.e. RMSE not train_RMSE

    # Sort by absolute difference (closest to zero)
    sorted_by_rmse_and_diff = deduped_df.sort_values(
        by=['test_RMSE', 'train_test_diff'], 
        ascending=[True, True]
    )
    sorted_by_rmse_and_diff.to_csv(leaderboard_file, index=False)

    # deduped_df.sort_values(by = 'train_test_diff').to_csv(leaderboard_file, index=False)
    logger.info(f"\nLeaderboard metrics saved to {leaderboard_file}\n")

    return sorted_by_rmse_and_diff

def update_readme_leaderboard(leaderboard_path='models/leaderboard.csv', 
                               readme_path='README.md',
                               logger=None):
    """
    Updates the README.md file with the current leaderboard in Markdown table format.
    
    Args:
        leaderboard_path (str): Path to the leaderboard CSV file.
        readme_path (str): Path to the README.md file.
        top_n (int): Number of leaderboard entries to include.
        logger (logging.Logger): Optional logger for logging updates.
        
    Returns:
        None
    """
    
    try:
        # Load leaderboard CSV
        df = pd.read_csv(leaderboard_path)

        if df.empty:
            msg = f"Leaderboard CSV at {leaderboard_path} is empty. Nothing to update in README."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return

        # Top N
        top_n = len(df)

        # Sort and select top N models (by test_RMSE as an example)
        df_sorted = df.sort_values(by='test_RMSE').head(top_n).reset_index(drop=True)

        # Convert to Markdown table format
        markdown_table = df_sorted.to_markdown(index=False)

        # Generate timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Read existing README content
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_contents = f.read()

        # Define markers where to inject leaderboard table
        marker_start = '<!-- LEADERBOARD_START -->'
        marker_end = '<!-- LEADERBOARD_END -->'

        # Create the new leaderboard section with timestamp
        new_leaderboard_section = (
            f"{marker_start}\n\n"
            f"**Leaderboard (Top {top_n} Models)**  \n"
            f"_Last updated: {current_time}_\n\n"
            f"{markdown_table}\n\n"
            f"{marker_end}"
        )

        # Check for existing markers in README
        if marker_start in readme_contents and marker_end in readme_contents:
            # Replace existing leaderboard block
            before = readme_contents.split(marker_start)[0]
            after = readme_contents.split(marker_end)[1]
            updated_readme = before + new_leaderboard_section + after
        else:
            # If no markers found, append at the end of the README
            updated_readme = (
                readme_contents +
                "\n\n## Current Model Leaderboard\n" +
                new_leaderboard_section
            )

        # Write the updated README file
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(updated_readme)

        msg = f"README.md successfully updated with the top {top_n} models from {leaderboard_path} at {current_time}."
        if logger:
            logger.info(msg)
        else:
            print(msg)

    except Exception as e:
        err_msg = f"Failed to update README leaderboard: {e}"
        if logger:
            logger.error(err_msg)
        else:
            print(err_msg)