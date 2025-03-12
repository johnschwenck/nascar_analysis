# ===============================
# NASCAR Race Results ML Pipeline
# ===============================
#
# Directory Structure:

# │   └── plots/                            # Visualizations
# │       ├── wins_distribution.png
# │       ├── over_under_wins.png
# │       └── championship_accuracy.png
# │
# └── notebooks/                            # Jupyter notebooks for EDA (optional)

# NASCAR_ANALYSIS/
# │
# ├── data/                                                 # Input data files
# │   ├── aggregated_driver_data.csv                        # Grouped source data by driver by year
# │   ├── champions_2014_2024.csv                           # Web-sourced driver data (winning driver by year + associated team)
# │   ├── cup_race_results_2014_2024.csv                    # Source data
# │   └── team_wins_by_year.csv                             # Web-sourced team data (total team wins by year)
# │
# ├── models/                                               # Model outputs (trained models + analysis)
# │   ├── ols/
# │   ├── poisson/
# │   │   └── poisson_20250312_0907/
# │   │       ├── analysis/                                 # Case Study Questions + Answers
# |   |            ├── driver_championship_comparison.csv
# |   |            ├── driver_over_under_performance.csv
# |   |            ├── team_championship_comparison.csv
# |   |            ├── team_over_under_performance.csv
# │   │       ├── plots/                                    # Visualizations
# │   │       ├── poisson_20250312_0907.pkl
# │   │       └── predictor_selection.csv
# │   ├── ...                                               # More model types: gbm, zip, etc. 
# │   ├── leaderboard.csv                                   # Model Leaderboard
# │
# ├── src/                                                  # Source code
# │   ├── championship_evaluator.py
# │   ├── logger_config.py
# │   ├── main.py
# │   ├── utils.py
# │   └── visualization.py
# │
# ├── .gitignore                                            # For GitHub
# ├── config.yaml                                           # User-defined config for modeling execution
# ├── nascar_analysis.ipynb                                 # Jupyter Notebook version for easier workflow
# ├── README.md                                             # Project instructions & documentation
# └── requirements.txt                                      # Package dependencies


# ===============================

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, PoissonRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import statsmodels.api as sm

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

from championship_evaluator import ChampionshipEvaluator
from utils import (load_config, load_data, prepare_features, 
                  scale_and_add_constant, get_enabled_model_types, load_model,
                  fit_and_evaluate_single_model, save_model, save_performance_metrics) # compute_metrics in utils

# Run PDP helper - automatically detects model type
from visualization import PDP_wrapper
import visualization as viz

from logger_config import setup_logger
# logger = setup_logger(__name__)


# ============================================
#            MAIN PIPELINE CLASS
# ============================================
class RacingMLPipeline:
    """
    A machine learning pipeline designed to process NASCAR race data and fit several regression models
    (OLS, Poisson, Negative Binomial, Zero-Inflated Poisson, Zero-Inflated Negative Binomial).

    It reads configuration from a YAML file, performs data preprocessing, splits and scales data,
    fits the models, evaluates performance, and saves a leaderboard of model performances.
    
    It also evaluates driver over/under-performance and compares modeled predictions to actual championships.
    """

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.__class__.__name__)
        self.random_seed = self.config['random_seed']
        np.random.seed(self.random_seed)
        self.scaler = StandardScaler()
        self.results = []
        self.metrics_df = None

        self.model_type = None
        self.best_model_type = None
        self.model_id = None
        self.best_model_id = None

        self.manual_model_id = self.config.get('model_selection', {}).get('manual_model_id')

        self.logger.info(f"Initialized RacingMLPipeline with config: {config_path}")

    # ============================================
    #            DATA PREPROCESSING
    # ============================================
    def preprocess_data(self, df):
        """
        Preprocesses the raw dataset to compute driver performance metrics.

        Args:
            df (pd.DataFrame): Raw input dataset.

        Returns:
            pd.DataFrame: Processed dataset with aggregated features per driver per season.
        """
        self.logger.info("\n\tDATA PREPROCESSING AND AGGREGATION\n")

        # Aggregate relevant performance metrics for each driver-season
        metrics_df = df.groupby(['race_season', 'driver_id', 'driver_fullname']).agg(
            num_races=('race_id', 'count'),
            wins=('finishing_position', lambda x: (x == 1).sum()),
            top_5=('finishing_position', lambda x: (x <= 5).sum()),
            top_10=('finishing_position', lambda x: (x <= 10).sum()),
            finish_2_5=('finishing_position', lambda x: ((x <= 5) & (x > 1)).sum()),
            finish_6_10=('finishing_position', lambda x: ((x <= 10) & (x > 5)).sum()),
            lead_lap_finishes=('diff_laps', lambda x: (x == 0).sum()),
            total_laps_led=('laps_led', 'sum'),
            total_laps=('actual_laps', 'sum'),
            avg_starting_position=('starting_position', 'mean'),
            avg_finishing_position=('finishing_position', 'mean'),
            avg_diff_laps=('diff_laps', 'mean'),
            avg_diff_time_sec=('diff_time', 'mean' ),
            avg_diff_time_min=('diff_time', lambda x: (x.mean())/60 )
        ).reset_index()

        # Calculate percentage of laps led
        metrics_df['pct_laps_led'] = metrics_df['total_laps_led'] / metrics_df['total_laps']

        # Retrieve the primary team for each driver-season
        team_df = df.groupby(['race_season', 'driver_id'])['team_name'].agg(lambda x: x.mode()[0]).reset_index()
        self.team_df = team_df

        # Calculate accidents per season per driver
        accidents_df = df[df['finishing_status'] == 'Accident'].groupby(['race_season', 'driver_id']).agg(
            num_accidents=('race_id', 'count')
        ).reset_index()

        # Calculate most likely cause of not finishing (DNF) per driver-season
        dnf_df = df[df['finishing_status'] != 'Running'].groupby(['race_season', 'driver_id'])['finishing_status'].agg(
                lambda x: x.value_counts().index[0] if not x.value_counts().empty else 'None'
            ).reset_index(name='most_likely_dnf_cause')

        # Merge team, accident, and DNF data into metrics_df
        metrics_df = metrics_df.merge(team_df, on=['race_season', 'driver_id'], how='left')
        metrics_df = metrics_df.merge(accidents_df, on=['race_season', 'driver_id'], how='left')
        metrics_df = metrics_df.merge(dnf_df, on=['race_season', 'driver_id'], how='left')

        # Fill NaN values in num_accidents with 0 (drivers with no accidents)
        metrics_df['num_accidents'] = metrics_df['num_accidents'].fillna(0).astype(int)

        # Fill NaN values in most_likely_dnf_cause with 'None' (drivers with no DNFs)
        metrics_df['most_likely_dnf_cause'] = metrics_df['most_likely_dnf_cause'].fillna('None')

        self.metrics_df = metrics_df

        return metrics_df


    def select_predictors(self, df, model_type, target=None, vif_threshold=10, min_variance=0.001):
        """
        Dynamically selects predictors for a Poisson/ZIP regression model from an input dataset.

        Args:
            df (pd.DataFrame): Input dataset.
            target (str, optional): Target variable name. If None, tries to infer from config or defaults to 'wins'.
            vif_threshold (float): Threshold for VIF to detect multicollinearity (default: 10).
            min_variance (float): Minimum variance threshold to exclude low-variation columns (default: 0.001).

        Returns:
            dict: Selected predictors split into count and inflation models (for ZIP), plus all predictors.
        """
        self.logger.info('\n\n\tFEATURE SELECTION\n')

        # Step 1: Identify target variable
        target = target or self.config.get('target_variable', 'wins')
        if target not in df.columns:
            self.logger.error(f"Target variable '{target}' not found in dataset.")
            raise ValueError(f"Target '{target}' not in dataset.")
        
        self.logger.info(f"Target variable identified: {target}")

        # Step 2: Classify columns
        predictors = [col for col in df.columns if col != target]
        numeric_cols = df[predictors].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[predictors].select_dtypes(exclude=[np.number]).columns.tolist()

        # Explicitly treat certain columns as categorical
        force_categorical = self.config.get('predictors', {}).get('force_categorical', ['race_season'])
        self.logger.info(f"Forcing {force_categorical} to be treated as categorical")
        for col in force_categorical:
            if col in numeric_cols:
                numeric_cols.remove(col)
            if col not in categorical_cols and col in predictors:
                categorical_cols.append(col)

        self.logger.info(f"\n\nNumeric columns: {numeric_cols}")
        self.logger.info(f"\nCategorical columns: {categorical_cols}\n\n")

        # Step 3: Exclude identifiers and low-variance columns
        exclude_patterns = ['_id', 'name', 'fullname']  # Common identifier keywords
        excluded_cols = []
        suggested_exclusions = []

        # Check config for explicit inclusions / exclusions
        if 'predictors' in self.config:
            if 'include' in self.config['predictors'] and self.config['predictors']['include']:
                predictors = self.config['predictors']['include']
                self.logger.info(f"Using manually specified predictors: {predictors}")
                return {
                    'all_predictors': predictors, 
                    'count_predictors': [col for col in predictors if col in numeric_cols], 
                    'inflation_predictors': [col for col in predictors if col in categorical_cols]
                }
            if 'exclude' in self.config['predictors']:
                excluded_cols.extend(self.config['predictors']['exclude'])
                self.logger.info(f"Exclusions from config to remove: {self.config['predictors']['exclude']}")

        # Exclude identifiers and low-variance columns
        for col in predictors:
            if any(pattern in col.lower() for pattern in exclude_patterns):
                excluded_cols.append(col)
                suggested_exclusions.append({
                    'Predictor': col,
                    'Reason': 'Identifier-like column name'
                })
                self.logger.info(f"Excluding identifier column: {col}")

            elif col in numeric_cols and df[col].var() < min_variance:
                excluded_cols.append(col)
                suggested_exclusions.append({
                    'Predictor': col,
                    'Reason': f'Low variance ({df[col].var():.5f})'
                })
                self.logger.info(f"Excluding low-variance column {col} (variance: {df[col].var():.4f})")
        
        predictors = [col for col in predictors if col not in excluded_cols]
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        categorical_cols = [col for col in categorical_cols if col not in excluded_cols]
        
        # Step 4: Calculate VIF for numeric predictors
        vif_data = pd.DataFrame(columns=['Predictor', 'VIF'])
        high_vif_cols = []
        
        X_numeric = df[numeric_cols].dropna()  # Drop NaNs for VIF calculation
        if len(numeric_cols) > 1 and len(X_numeric) > len(numeric_cols):
            vif_data['Predictor'] = numeric_cols
            vif_data['VIF'] = [variance_inflation_factor(X_numeric.values, i)
                            for i in range(X_numeric.shape[1])]

            high_vif_cols = vif_data[vif_data['VIF'] > vif_threshold]['Predictor'].tolist()

            # Add high VIF columns to suggested exclusions (optional for user)
            for col in high_vif_cols:
                suggested_exclusions.append({
                    'Predictor': col,
                    'Reason': f'High VIF ({vif_data[vif_data["Predictor"] == col]["VIF"].values[0]:.2f})'
                })

            self.logger.info(f"\n\nVIF results:\n{vif_data.to_string(index=False)}\n")
        else:
            self.logger.info("\nSkipping VIF check: too few numeric columns or rows.")

        # Step 5: Heuristic split for ZIP model
        # Count model: performance-related numeric predictors
        count_predictors = [col for col in numeric_cols if col not in ['num_accidents']]
        # Inflation model: categorical + accident-related predictors
        inflation_predictors = categorical_cols + ['num_accidents'] if 'num_accidents' in numeric_cols else categorical_cols
        self.logger.info(f"\nCount model predictors (Poisson): {count_predictors}")
        self.logger.info(f"\nInflation model predictors (Zero-Inflation): {inflation_predictors}")

        # Step 6: Combine all predictors for standard Poisson
        all_predictors = predictors

        # Step 7: Log and save results
        result = {
            'all_predictors': all_predictors,
            'count_predictors': count_predictors,
            'inflation_predictors': inflation_predictors
        }
        self.logger.info(f"\n\nFinal selected predictors: {result}\n")
        
        # Save to CSV
        predictor_df = pd.DataFrame({
            'Predictor': all_predictors,
            'Type': ['Numeric' if col in numeric_cols else 'Categorical' for col in all_predictors]
        })
        output_path = os.path.join('models', model_type, self.model_id, f'predictor_selection.csv')
        predictor_df.to_csv(output_path, index=False)
        self.logger.info(f"Selected predictors saved to {output_path}")

        return result
    

    def data_partitioning(self, df, model_type):
        """
        Splits the data into training and testing sets and applies feature scaling.

        Args:
            df (pd.DataFrame): Preprocessed dataset.
            model_type (str): Type of model to prepare data for.

        Returns:
            tuple: Scaled and constant-added training and testing datasets and corresponding labels.
        """
        self.logger.info('\n\n\tDATA PARTITIONING\n')

        is_zip_model = model_type.lower() in ['zip', 'zinb']
        predictors = self.select_predictors(df, model_type)
        
        # Extract X and y depending on model type
        if is_zip_model:
            X_count, X_inflate = prepare_features(df, predictors['count_predictors'], predictors['inflation_predictors'], logger=self.logger)
        else:
            X = prepare_features(df, predictors['all_predictors'], logger=self.logger)

        y = df[self.config['target_variable']].astype(float)

        stratify = df[self.config['train_test_split']['stratify']] if self.config['train_test_split']['stratify'] else None

        # Split data
        if is_zip_model:
            X_count_train, X_count_test, X_inflate_train, X_inflate_test, y_train, y_test = train_test_split(
                X_count, X_inflate, y,
                test_size=self.config['train_test_split']['test_size'],
                random_state=self.random_seed,
                stratify=stratify
            )
            X_count_train_const, X_count_test_const = scale_and_add_constant(X_count_train, X_count_test, self.scaler)
            X_inflate_train_const, X_inflate_test_const = scale_and_add_constant(X_inflate_train, X_inflate_test, self.scaler)

            return (
                X_count_train, X_count_test, X_count_train_const, X_count_test_const,
                X_inflate_train, X_inflate_test, X_inflate_train_const, X_inflate_test_const,
                y_train, y_test
            )

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['train_test_split']['test_size'],
                random_state=self.random_seed,
                stratify=stratify
            )
            X_train_const, X_test_const = scale_and_add_constant(X_train, X_test, self.scaler)

            return X_train, X_test, X_train_const, X_test_const, y_train, y_test
        
    
    # ============================================
    #            MODEL FITTING & EVALUATION
    # ============================================
    def fit_models(self, df, model_types=None, cv_folds=None):
        """
        Fits multiple models to the dataset based on the config.

        Args:
            df (pd.DataFrame): Preprocessed dataset.
            model_types (list): List of models to fit (from config).
            cv_folds (int or None): Number of CV folds if cross-validation is enabled.
        """
        self.logger.info('\n\n\tMODEL FITTING STARTED...\n')

        # Default to all models from config
        model_types = model_types or get_enabled_model_types(self.config, logger=self.logger)

        # Cross-validation setup
        cv_folds = cv_folds or self.config.get('cross_validation', {}).get('folds')
        if cv_folds:
            self.logger.info(f"Cross-validation enabled: {cv_folds} folds")
        else:
            self.logger.info("Cross-validation disabled")

        save_models = self.config.get('output', {}).get('save_models', False)
        performance_results = []

        # Create model directory if saving models
        if save_models:
            os.makedirs('models', exist_ok=True)

        # Iterate over each model type
        for model_type in model_types:
            self.logger.info(f"\n\n\t\t+------- Processing model: {model_type} -------+\n")

            # Add model_id (model_type + timestamp)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            self.model_id = f"{model_type}_{timestamp}"

            # Create model-specific directory structure
            model_base_dir = os.path.join('models', model_type, self.model_id)
            model_dirs = {
                'base': model_base_dir,
                'plots': os.path.join(model_base_dir, 'plots'),
                'analysis': os.path.join(model_base_dir, 'analysis')
            }
            for dir_path in model_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory structure for model_id {self.model_id}: \n{model_base_dir}")

            # # Save preprocessed data in the model folder
            # metrics_df_path = os.path.join(model_base_dir, 'processed_metrics.csv')
            # df.to_csv(metrics_df_path, index=False)
            # self.logger.info(f"Saved processed metrics to {metrics_df_path}")

            # Partition data and get predictors
            data_parts = self.data_partitioning(df, model_type)
            predictors = self.select_predictors(df, model_type)  # Get predictors for PDP

            # # Determine if this is a zero-inflated model based on predictors
            # is_zero_inflated = 'inflation_predictors' in predictors and predictors['inflation_predictors']

            # # Unpack data parts based on model type
            # if is_zero_inflated:
            #     # Assume data_parts = (X_count_train, X_inflation_train, X_count_test, X_inflastion_test, y_train, y_test)
            #     if len(data_parts) != 6:
            #         raise ValueError(f"Expected 6 data parts for zero-inflated model {model_type}, got {len(data_parts)}")
            #     X_count_train, X_inflation_train, X_count_test, X_inflation_test, y_train, y_test = data_parts

            #     # Get the full set of column names (assuming encoding has been applied)
            #     all_columns = predictors['all_predictors']
            #     encoded_columns = list(df.columns)  # Use the encoded column names from the preprocessed df

            #     # Map original predictors to their encoded indices
            #     count_indices = [encoded_columns.index(col) for col in predictors['count_predictors'] if col in encoded_columns]
            #     inflation_indices = [encoded_columns.index(col) for col in predictors['inflation_predictors'] if col in encoded_columns]

            #     # Slice the arrays to match the predictor sets
            #     X_count_train_df = pd.DataFrame(X_count_train[:, count_indices], columns=predictors['count_predictors'])
            #     X_inflation_train_df = pd.DataFrame(X_inflation_train[:, inflation_indices], columns=predictors['inflation_predictors'])
            #     X_count_test_df = pd.DataFrame(X_count_test[:, count_indices], columns=predictors['count_predictors'])
            #     X_inflation_test_df = pd.DataFrame(X_inflation_test[:, inflation_indices], columns=predictors['inflation_predictors'])

            #     # Combine for general use
            #     X_train = pd.concat([X_count_train_df, X_inflation_train_df], axis=1)
            #     X_test = pd.concat([X_count_test_df, X_inflation_test_df], axis=1)
            # else:
            #     # Assume data_parts = (X_train, X_test, y_train, y_test)
            #     if len(data_parts) != 4:
            #         raise ValueError(f"Expected 4 data parts for non-zero-inflated model {model_type}, got {len(data_parts)}")
            #     X_train, X_test, y_train, y_test = data_parts

            # Get metrics (partial) and model from evaluation
            metrics_partial, model = fit_and_evaluate_single_model(model_type, data_parts, cv_folds, config = self.config, random_seed = self.random_seed, logger=self.logger)

            # Add metadata
            metrics = {
                'model_id': self.model_id,
                'model_type': model_type,
                'timestamp': timestamp,
                **metrics_partial
            }
            
            self.logger.info(f"\n\nModel ID: {self.model_id}\n")

            performance_results.append(metrics)

            # Save the model if requested
            if save_models:
                save_model(model, model_type, self.model_id, logger = self.logger)

            # # Create Subdirectories for various visualizations
            # feat_dist_dir = os.path.join(model_dirs['plots'], 'feat_dist')
            # corr_dir = os.path.join(model_dirs['plots'], 'corr')
            # feature_importance_dir = os.path.join(model_dirs['plots'], 'feature_importance')
            # residuals_dir = os.path.join(model_dirs['plots'], 'residuals')

            # for path in [feat_dist_dir, corr_dir, feature_importance_dir, residuals_dir]:
            #     os.makedirs(path, exist_ok=True)

            # # === Visualizations ===
            
            # # 1. Feature Distributions
            # viz.plot_feature_distributions(
            #     df[predictors['all_predictors']],  # Use only selected predictors
            #     save_path=feat_dist_dir,
            #     exclude_columns=['driver_id', 'race_season']
            # )

            # # 2. Correlation Heatmap
            # viz.plot_correlation_heatmap(
            #     df[predictors['all_predictors']],
            #     save_path=corr_dir,
            #     exclude_columns=['driver_id']
            # )

            # # 3. Feature Importance (using training data features)
            # viz.plot_feature_importance(
            #     model,
            #     X_train.columns,  # Use column names from X_train
            #     save_path=feature_importance_dir,
            #     top_n=10
            # )

            # # 4. Residuals Plot (using test data predictions)
            # y_pred_test = model.predict(X_test)
            # viz.plot_residuals(
            #     y_test,
            #     y_pred_test,
            #     save_path=residuals_dir
            # )

            # Check visualization settings from config
            vis_config = self.config.get('visualization', {})
            display_pdps = vis_config.get('default_display', True)  # Defaults to True unless explicitly False
            features_to_plot = vis_config.get('pdp_features', None)

            if display_pdps:

                # PDPs: Generate Partial Dependence Plots and save them
                self.logger.info(f"Generating PDPs for model {self.model_id}...")

                X_train = data_parts[0]  # Raw features
                PDP_wrapper(
                    model=model,
                    X=X_train,
                    predictors=predictors,  # Pass predictors dictionary
                    save_path=model_dirs['plots'],
                    features=features_to_plot,
                    logger=self.logger
                )
                self.logger.info(f"Finished generating PDPs for model {self.model_id}")
            else:
                self.logger.info("PDP generation disabled in config. Skipping PDP plots.")

            ## Championship Evaluation
            model_results = self.generate_predictions(df, model = model, model_id = self.model_id, model_type = model_type)

            # 4.1(F). Championship Evaluation for manual model
            # from championship_evaluator import ChampionshipEvaluator
            evaluator = ChampionshipEvaluator(self.config, logger=self.logger)
            
            driver_comparison_df = evaluator.evaluate_championship_predictions(
                df,
                model_results,
                entity='driver',
                model_type=model_type,
                model_id=self.model_id
            )

            team_comparison_df = evaluator.evaluate_championship_predictions(
                df,
                model_results,
                entity='team',
                model_type=model_type,
                model_id=self.model_id
            )

            # # 7. Over/Under Performance Plot
            # viz.plot_over_under_performance(
            #     metrics_df=df,
            #     predictions=model_results['predictions'],
            #     output_dir=os.path.join(model_dirs['plots'], 'performance')
            # )

            # # 8. Champion Prediction Accuracy Plot
            # viz.plot_champion_accuracy(
            #     champion_df=driver_comparison_df,
            #     output_dir=os.path.join(model_dirs['plots'], 'champion_accuracy')
            # )

        # Save leaderboard CSV
        model_LB = save_performance_metrics(performance_results, self.config, logger = self.logger)

        self.logger.info("\nAll models have been fitted and evaluated.\n")
        self.logger.info(f"Current Model Leaderboard:\n{model_LB}")

    #     ## select top model and assign to self.best_model_id and self.best_model_type
    #     self.select_best_model(performance_results)

    # # ============================================
    # #            MODEL SELECTION
    # # ============================================
    # def select_best_model(self, performance_results):
    #     """
    #     Selects the best model based on config criteria or manual override.
    #     """
    #     manual_model_id = self.config.get('model_selection', {}).get('manual_model_id')
    #     rank_metric = self.config.get('model_selection', {}).get('metric', 'Test_RMSE')
    #     minimize = self.config.get('model_selection', {}).get('minimize', True)

    #     # 1. Manual override
    #     if manual_model_id:
    #         for result in performance_results:
    #             if result['model_id'] == manual_model_id:
    #                 self.best_model_id = manual_model_id
    #                 self.best_model_type = result['model_type']
    #                 self.logger.info(f"Manual model override selected: {self.best_model_id}")
    #                 return

    #         # Manual ID not found
    #         self.logger.error(f"Manual model_id {manual_model_id} not found in fitted models")
    #         raise ValueError(f"Manual model_id {manual_model_id} not found.")

    #     # 2. Automatic selection by metric
    #     sorted_models = sorted(
    #         performance_results,
    #         key=lambda x: x.get(rank_metric, float('inf' if minimize else '-inf')),
    #         reverse=not minimize
    #     )

    #     if not sorted_models:
    #         self.logger.error("No models were fitted. Cannot select best model.")
    #         raise ValueError("No models found for selection.")

    #     # print(sorted_models)
    #     best_model = sorted_models[0]
    #     # print(best_model)
    #     self.best_model_id = best_model['model_id']
    #     self.best_model_type = best_model['model_type']
    #     # print(best_model)

    #     self.logger.info(f"Automatically selected best model based on {rank_metric}: {self.best_model_id} ({self.best_model_type}) - Test RMSE: {best_model['test_RMSE']}")


    # ============================================
    #            PREDICTIONS
    # ============================================
   
    def _predict_expected_wins(self, model, model_type, metrics_df):

        is_zip_model = model_type in ['zip', 'zinb']
        predictors = self.select_predictors(metrics_df, model_type)

        if is_zip_model:
            X_count, X_inflate = prepare_features(
                metrics_df, 
                predictors['count_predictors'], 
                predictors['inflation_predictors'], 
                logger=self.logger
            )

            self.scaler.fit(X_count)
            X_count_scaled = self.scaler.transform(X_count)
            X_count_const = sm.add_constant(X_count_scaled, has_constant='add')

            self.scaler.fit(X_inflate)
            X_inflate_scaled = self.scaler.transform(X_inflate)
            X_inflate_const = sm.add_constant(X_inflate_scaled, has_constant='add')

            expected_wins = model.predict(X_count_const, exog_infl=X_inflate_const)

        else:
            X = prepare_features(metrics_df, predictors['all_predictors'], logger=self.logger)

            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            X_const = sm.add_constant(X_scaled, has_constant='add')

            if model_type in ['xgboost', 'gbm']:
                expected_wins = model.predict(X)
            else:
                expected_wins = model.predict(X_const)

        results_df = pd.DataFrame({
            'race_season': metrics_df['race_season'],
            'driver_id': metrics_df['driver_id'],   
            'driver_fullname': metrics_df['driver_fullname'],
            'wins': metrics_df['wins'],
            'expected_wins': expected_wins
        })

        results_df['win_diff'] = results_df['wins'] - results_df['expected_wins']
        results_df.sort_values(by=['race_season', 'expected_wins'], ascending=[True, False], inplace=True)

        return results_df

    def generate_predictions(self, metrics_df, model=None, model_id=None, model_type=None, manual=False):
        """
        Generates predictions from a specified or top model.
        
        Args:
            metrics_df (pd.DataFrame): Preprocessed data.
            model (object, optional): Preloaded model. If not provided, loads using model_id.
            model_id (str, optional): Model identifier. Required if model is None.
            model_type (str, optional): Model type. Required if model is None.
            manual (bool): Whether this is a manual model (for logging).
        
        Returns:
            pd.DataFrame: Predictions with expected_wins and win_diff.
        """
        # If model isn't passed, we need to load it
        if model is None:
            model_id = model_id or self.best_model_id
            model_type = model_type or self.best_model_type

            model = load_model(model_id, logger=self.logger)
            self.logger.info(f"Loaded model from saved file: {model_id}")
        else:
            # If a manual model is provided but no type or id, use manual ones
            model_id = model_id or self.manual_model_id
            model_type = model_type or self.model_type

        # Run the prediction
        results_df = self._predict_expected_wins(model, model_type, metrics_df)

        # Log depending on whether it's manual or top model
        if manual:
            self.logger.info(f"Generated predictions using manual model_id: {model_id}")
        else:
            self.logger.info(f"Generated predictions using best model_id: {model_id}")

        return results_df
    
    # ============================================
    #            RUN PIPELINE
    # ============================================

    def run_pipeline(self):
        """
        Runs the complete pipeline: loads data, preprocesses, fits models, evaluates, and saves leaderboard.
        """
        self.logger.info("Starting Racing ML Pipeline...")

        # 1. Load and preprocess data
        df = load_data(self.config)

        # 3. Load leaderboard file path (if it exists)
        leaderboard_file = os.path.join('models', self.config['output']['leaderboard_csv'])

    
        # # 2. Save aggregated driver data for inspection
        # aggregated_data_path = os.path.join('data', 'aggregated_driver_data.csv')
        # self.logger.info(f"Saving aggregated driver data to {aggregated_data_path}")
        # metrics_df.to_csv(aggregated_data_path, index=False)


        if self.manual_model_id:

            self.logger.info(f"model_id found in config file")
            self.logger.info(f"Manual model selection enabled with model_id: {self.manual_model_id}")

            # 4.1(A). Validate the manual_model_id exists in leaderboard
            if not os.path.exists(leaderboard_file):
                raise FileNotFoundError(f"Leaderboard file not found at {leaderboard_file}")

            leaderboard_df = pd.read_csv(leaderboard_file)

            if self.manual_model_id not in leaderboard_df['model_id'].values:
                raise ValueError(f"Manual model_id {self.manual_model_id} not found in leaderboard.")
            
            # 4.1(B). Load the specified model
            model = load_model(self.manual_model_id, logger = self.logger)

            # 4.1(C). Save info about selected model type for downstream use
            model_row = leaderboard_df[leaderboard_df['model_id'] == self.manual_model_id].iloc[0]
            self.model_id = self.manual_model_id
            self.model_type = model_row['model_type']
            # self.best_model_type = model_row['model_type']
            self.logger.info(f"Manual model {self.manual_model_id} of type {self.model_type} loaded successfully.")
            
            # 4.1(D) Check for preprocessed data in model folder
            metrics_df_path = os.path.join('data', 'aggregated_driver_data.csv')
            if os.path.exists(metrics_df_path):
                self.logger.info(f"Loading processed data from {metrics_df_path}")
                metrics_df = pd.read_csv(metrics_df_path)
            else:
                self.logger.info("Processed data not found; generating new data")
                metrics_df = self.preprocess_data(df)

                # Save for future use
                metrics_df.to_csv(metrics_df_path, index=False)
                self.logger.info(f"Saved processed metrics to {metrics_df_path}")

            self.logger.info('Driver-Aggregated Data (first 5 rows):')
            self.logger.info(metrics_df.head().to_string(index=False))

            # 4.1(E). Run predictions with the loaded model
            manual_model_results = self.generate_predictions(metrics_df, model)

            # 4.1(F). Championship Evaluation for manual model
            from championship_evaluator import ChampionshipEvaluator
            evaluator = ChampionshipEvaluator(self.config, logger=self.logger)
            
            driver_comparison_df = evaluator.evaluate_championship_predictions(
                metrics_df,
                manual_model_results,
                entity='driver',
                model_type=self.model_type,
                model_id=self.manual_model_id
            )

            team_comparison_df = evaluator.evaluate_championship_predictions(
                metrics_df,
                manual_model_results,
                entity='team',
                model_type=self.model_type,
                model_id=self.manual_model_id
            )

            self.logger.info("Championship evaluations completed for manual model.")

        else:

            # Run modeling process

            # 4.2(A) Process or load Data    
            aggregated_data_path = os.path.join('data', 'aggregated_driver_data.csv')

            if not self.config['generate_grouped_data'] and not os.path.exists(aggregated_data_path):
                raise FileNotFoundError(
                    "'generate_grouped_data' in config is set to False, but no aggregated_driver_data.csv file was found.\n"
                    "Set 'generate_grouped_data' to True to re-generate the data."
                )

            if self.config['generate_grouped_data']:
                metrics_df = self.preprocess_data(df)
            else:
                metrics_df = pd.read_csv(aggregated_data_path)

            self.logger.info('Driver-Aggregated Data (first 5 rows):')
            self.logger.info(metrics_df.head().to_string(index=False))

            # # 4.2(B). Determine which models to run
            # model_types = get_enabled_model_types(self.config)
            # self.logger.info(f"Model types to run: {model_types}")
            
            # 4.2(C). Cross-validation setup
            # cv_folds = self.config.get('cross_validation', {}).get('folds')
            # if cv_folds:
            #     self.logger.info(f"Cross-validation enabled: {cv_folds} folds")
            # else:
            #     self.logger.info("Cross-validation disabled")

            # 4.2(B) Run the model fitting process
            self.fit_models(metrics_df)

            # # Championship Evaluation
            # self.logger.info("Generating predictions...")
            # best_model_results = self.generate_predictions(metrics_df)

            # # 4.1(F). Championship Evaluation for manual model
            # from championship_evaluator import ChampionshipEvaluator
            # evaluator = ChampionshipEvaluator(self.config, logger=self.logger)
            
            # driver_comparison_df = evaluator.evaluate_championship_predictions(
            #     metrics_df,
            #     best_model_results,
            #     entity='driver',
            #     model_type=self.model_type,
            #     model_id=self.best_model_id
            # )

            # team_comparison_df = evaluator.evaluate_championship_predictions(
            #     metrics_df,
            #     best_model_results,
            #     entity='team',
            #     model_type=self.model_type,
            #     model_id=self.best_model_id
            # )

            # self.logger.info("Championship evaluations completed for all fitted models.")


if __name__ == "__main__":

    tmp = RacingMLPipeline('config.yaml')
    tmp.run_pipeline()















