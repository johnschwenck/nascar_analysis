import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.discrete import discrete_model
import statsmodels.api as sm
import warnings

# Suppress the specific sklearn warning about ylims
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.inspection._plot.partial_dependence")

def plot_over_under_performance(metrics_df, predictions, output_dir):
    """
    Plots over/under performance for drivers.
    """
    df = metrics_df.copy()
    df['expected_wins'] = predictions
    df['over_under'] = df['wins'] - df['expected_wins']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='driver_fullname', y='over_under', hue='race_season')
    plt.title('Driver Over/Under Performance')
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/over_under_wins.png")
    plt.close()

def plot_champion_accuracy(champion_df, output_dir):
    """
    Plots whether the model-predicted driver matched the actual champion.
    """
    fig = px.bar(champion_df, x='race_season', y='match', title='Champion Prediction Accuracy (Per Season)')
    fig.write_image(f"{output_dir}/championship_accuracy.png")

def plot_feature_distributions(df, save_path=None, exclude_columns=[]):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns)
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "feature_distributions.png"))
    plt.close()

def plot_correlation_heatmap(df, save_path=None, exclude_columns=[]):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns)
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')

    if save_path:
        plt.savefig(os.path.join(save_path, "correlation_heatmap.png"))
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title("Top Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")

        if save_path:
            plt.savefig(os.path.join(save_path, "feature_importance.png"))
        plt.close()
    else:
        print("Model does not support feature_importances_.")

def plot_residuals(y_true, y_pred, save_path=None):
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")

    if save_path:
        plt.savefig(os.path.join(save_path, "residuals_plot.png"))
    plt.close()

#######################

def detect_model_family(model):
    """
    Detect if the model is from sklearn or statsmodels.
    Returns: 'sklearn', 'statsmodels', or 'unknown'
    """

    # Statsmodels classes to check
    statsmodels_wrappers = (
        RegressionResultsWrapper,
        GLMResultsWrapper,
        discrete_model.BinaryResultsWrapper,
        discrete_model.MultinomialResultsWrapper
    )

    # Check for statsmodels
    if isinstance(model, statsmodels_wrappers):
        return 'statsmodels'
    # Check for scikit-learn
    if isinstance(model, BaseEstimator):
        return 'sklearn'
    return 'unknown'

def PDP_wrapper(model, X, predictors, save_path=None, features=None, logger=None):
    """
    Wrapper function to generate partial dependence plots for both sklearn and statsmodels models.
    
    Args:
        model: Trained model (either sklearn or statsmodels)
        X: DataFrame of features used for prediction
        predictors: Dictionary containing 'all_predictors' (and optionally 'count_predictors', 'inflation_predictors')
        save_path: Directory to save the plots
        features: List of feature names to generate PDPs for (optional: defaults to first 10 predictors)
        logger: Optional logger
    """
    if save_path:
        pdp_save_path = os.path.join(save_path, 'PDP')
        os.makedirs(pdp_save_path, exist_ok=True)
    else:
        pdp_save_path = None
    
    # Detect model type (sklearn or statsmodels)
    model_family = detect_model_family(model)

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
        else:
            if logger:
                logger.error("X must be a pandas DataFrame or numpy array")
            return
    
    # Determine available features based on X columns (after encoding)
    available_features = X.columns.tolist()

    # Select features for PDP
    all_predictors = predictors['all_predictors']
    if features is None:
        features = all_predictors[:min(10, len(all_predictors))]
    else:
        features = [f for f in features if f in all_predictors]
        if not features:
            if logger:
                logger.warning("No valid features specified for PDP. Using default features.")
            features = all_predictors[:min(10, len(all_predictors))]


    # For sklearn models, adjust features to match encoded columns
    if model_family == 'sklearn':
        # Identify categorical features that would have been encoded
        original_df = pd.DataFrame(columns=all_predictors)
        for col in all_predictors:
            original_df[col] = pd.Series(dtype='object' if col in ['race_season', 'most_likely_dnf_cause'] else 'float')
        cat_cols = original_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        pdp_features = []
        categorical_groups = {}  # To group encoded features by original name
        for feature in features:
            if feature in cat_cols:
                encoded_cols = [col for col in available_features if col.startswith(f"{feature}_")]
                if encoded_cols:
                    categorical_groups[feature] = encoded_cols
                    pdp_features.extend(encoded_cols)  # Keep individual encoded columns for now
            else:
                if feature in available_features:
                    pdp_features.append(feature)
        if not pdp_features:
            if logger:
                logger.error("No valid features found in X after encoding adjustments")
            return
    else:
        pdp_features = features  # Statsmodels uses original features


    if logger:
        logger.info(f"Detected model family: {model_family}. Generating PDPs for features: {features}")

    try:
        if model_family == 'sklearn':
            for feature in pdp_features:
                fig, ax = plt.subplots(figsize=(8, 6))
                # Generate PDP for the single feature
                disp = PartialDependenceDisplay.from_estimator(model, X, [feature], kind='average', ax=ax)
                plt.title(f"Partial Dependence of {feature}")
                
                # Check if the PDP has meaningful variation
                pd_values = disp.pd_results_[0].average
                if np.allclose(pd_values, pd_values[0], atol=1e-3):  # Check for near-constant values
                    if logger:
                        logger.warning(f"Skipping PDP for {feature} due to near-constant values")
                    plt.close(fig)
                    continue
                
                plt.tight_layout()
                if pdp_save_path:
                    plot_file = os.path.join(pdp_save_path, f'pdp_{feature}.png')
                    plt.savefig(plot_file)
                    if logger:
                        logger.info(f"Saved PDP plot for {feature} to {plot_file}")
                plt.close(fig)

            # Optional: Aggregate PDPs for categorical variables
            for orig_feature, encoded_cols in categorical_groups.items():
                if len(encoded_cols) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    all_pds = []
                    for enc_col in encoded_cols:
                        disp = PartialDependenceDisplay.from_estimator(model, X, [enc_col], kind='average')
                        all_pds.append(disp.pd_results_[0].average)
                    avg_pd = np.mean(all_pds, axis=0)
                    grid = np.linspace(0, 1, len(avg_pd))  # Approximate grid
                    ax.plot(grid, avg_pd, label=f"Average PDP for {orig_feature}")
                    ax.set_title(f"Average Partial Dependence of {orig_feature}")
                    ax.set_xlabel(orig_feature)
                    ax.set_ylabel("Average Prediction")
                    ax.legend()
                    if pdp_save_path:
                        plot_file = os.path.join(pdp_save_path, f'pdp_avg_{orig_feature}.png')
                        plt.savefig(plot_file)
                        if logger:
                            logger.info(f"Saved averaged PDP plot for {orig_feature} to {plot_file}")
                    plt.close(fig)

        elif model_family == 'statsmodels':
            for feature in pdp_features:
                manual_PDP(model, X, feature, pdp_save_path, logger)

        else:
            if logger:
                logger.warning(f"No PDP implementation for model family: {model_family}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to generate PDP: {str(e)}")

    # if model_family == 'sklearn':
    #     # Use sklearn PartialDependenceDisplay for tree-based models
    #     try:
    #         disp = PartialDependenceDisplay.from_estimator(model, X, features, kind='average')
    #         plt.tight_layout()
            
    #         if save_path:
    #             plot_file = os.path.join(save_path, f'pdp_sklearn.png')
    #             plt.savefig(plot_file)
    #             logger_msg = f"Saved PDP plot to {plot_file}"
    #             print(logger_msg) if logger is None else logger.info(logger_msg)

    #         plt.close()

    #     except Exception as e:
    #         logger_msg = f"Failed to generate PDP for sklearn model: {str(e)}"
    #         print(logger_msg) if logger is None else logger.error(logger_msg)

    # elif model_family == 'statsmodels':
    #     for feature in features:
    #         manual_PDP(model, X, feature, save_path=save_path, logger=logger)
    
    # else:
    #     logger_msg = f"No PDP implementation for model type: {model_family}"
    #     print(logger_msg) if logger is None else logger.warning(logger_msg)


def manual_PDP(model, X, feature, save_path=None, logger=None):
    """
    Creates a simple partial dependence plot for a single feature in statsmodels models.

    Args:
        model: statsmodels fitted model
        X: Original dataset (DataFrame)
        feature: The feature column name to plot PDP for
        save_path: Optional folder to save plot
        logger: Optional logger
    """
    if logger:
        logger.info(f"Generating manual PDP for {feature}")

    X_copy = X.copy()

    # Values to test in PDP: linear space of the feature
    grid = np.linspace(X_copy[feature].min(), X_copy[feature].max(), 50)
    avg_prediction = []

    # Prepare the full feature set with scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_copy)
    X_const = sm.add_constant(X_scaled, has_constant='add')
    feature_idx = X_copy.columns.get_loc(feature)

    for val in grid:

        # Replace the scaled value for the feature
        X_temp = X_scaled.copy()
        original_vals = X_copy[feature].values
        scaled_val = (val - original_vals.mean()) / original_vals.std()  # Scale the grid value
        X_temp[:, feature_idx] = scaled_val
        
        X_temp_const = sm.add_constant(X_temp, has_constant='add')
        preds = model.predict(X_temp_const)
        avg_prediction.append(np.mean(preds))

    # Plot PDP
    plt.figure(figsize=(8, 6))
    plt.plot(grid, avg_prediction, label=f"PDP for {feature}")
    plt.xlabel(f"{feature}")
    plt.ylabel("Average Prediction")
    plt.title(f"Partial Dependence of {feature} (statsmodels)")
    plt.legend()

    if save_path:
        plot_file = os.path.join(save_path, f"manual_pdp_{feature}.png")
        plt.savefig(plot_file)
        if logger:
            logger.info(f"Saved manual PDP for {feature} to {plot_file}")

    plt.close()
