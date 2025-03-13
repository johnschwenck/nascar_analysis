## High-Level Workflow for NASCAR Race Results ML Pipeline
The purpose of this code is to analyze historical NASCAR race results to predict how many races a driver should win each season and compare that prediction to actual results. It also identifies patterns to evaluate team and driver performance, helping teams understand who exceeded expectations (over-performed) or fell short (under-performed). This pipeline systematically processes NASCAR race data to predict expected driver performance, analyze actual versus expected outcomes, and evaluate championship results using advanced statistical modeling.

### Workflow Steps
Here's how the workflow moves step-by-step at a high level:

1. **Configuration Setup (`config.yaml`)**

> ***Function: `load_config()`***
     Define file paths, model settings and hyperparameters, and analysis settings.

2. **Data Loading**

> ***Function: `load_data()`***
    - Load NASCAR race results data and championship data based on the configuration YAML file.

3. **Data Preprocessing**

> Function: `preprocess_data()`
    - Aggregate and summarize raw race results data to calculate key metrics (wins, finishes, laps led) by driver and season, calculating statistics like total wins, top finishes, laps led, and accidents per driver each year.

4. **Feature Selection**

> Function: `select_predictors()`
    - Dynamically choose variables that effectively predict driver wins, removing redundant or irrelevant data in order to strengthen the signal into the model.

5. **Data Splitting**

> Function: `data_partitioning()`
    - Split dataset into training and testing sets, maintaining season-based groupings, to evaluate robustness, over-fitting, and out-of-sample performance.

6. **Model Training & Evaluation**

> Function: `fit_models()`
- Train machine learning models (XGBoost, GBM) as well as statistical models (OLS, Poisson, Negative Binomial, etc) using historical data to predict expected driver wins. It uses the selected variables from step 5, and built with the training data from step 6. After training, the model's performance is evaluated to understand accuracy—how well predictions match actual outcomes.

7. **Performance Metrics & Analysis**

> Function: `compute_metrics()`
   - Measure prediction performance using RMSE, MAE, and R² metrics.
   - Calculates how well the model did by comparing predicted wins to actual wins. Key metrics include RMSE (how off predictions are), MAE (average prediction error), and R² (how well the predictions fit the actual results).

8. **Model Saving and Loading**

> Function: `save_model()` & `load_model()`
    - Once trained, models are saved for future use. This allows us to revisit previous analyses easily without having to retrain models every time you analyze data.

9. **Visualizations (`visualization.py`)**

> Functions: `plot_over_under_performance()`, `plot_feature_distributions()`, `plot_correlation_heatmap()`, `plot_feature_importance()`, `plot_residuals()`
   - Generate visual insights, such as performance distributions, correlation heatmaps, and feature importance, to understand which drivers performed above or below expectations, how different factors influenced wins, and any relationships / patterns in the data.

10. **Championship Evaluation (`ChampionshipEvaluator`)**

> Function: `evaluate_championship_predictions()`
   - Compare predicted champions (drivers or teams) against actual champions to assess model effectiveness.
   - Checks if the model's predicted "best" driver or team actually won the championship. It summarizes how often predictions match reality, highlighting accuracy and the effectiveness of the predictive models.

11. **Logging and Reporting**

Function: `setup_logger()`
   - Track and document the entire analysis process for transparency and ease of debugging.

12. **Running the Entire Workflow**  

   > **Function: `run_pipeline()`**  
   - Orchestrates all steps in a single call—from loading the configuration file to preprocessing data, training models, evaluating performance, generating visualizations, and saving results. This function allows the entire workflow to be executed consistently with just one line of code.


### Stand-alone Utility Functions (Helpers)
Several helper functions perform important tasks independently, assisting the main functions above:

`prepare_features()`
- Prepares data for predictive modeling by handling categorical variables (turning categories into usable numerical values) and filling in missing values.

`scale_and_add_constant()`
- Adjusts the numerical data to be on the same scale, making comparisons and predictions more accurate.

`compute_metrics()`
- Calculates performance metrics to evaluate the accuracy of predictions clearly and objectively.

`fit_ml_model()`
- Trains models efficiently and chooses the best settings (hyperparameters) for prediction accuracy.

`cross_validate()`
- Checks the stability of models by running multiple trials on different subsets of data, ensuring predictions are reliable.
