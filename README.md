```
# ===============================
# NASCAR Race Results ML Pipeline
# ===============================
#
# Directory Structure:
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
```
## Background

The list of high-level metrics that JGR typically uses to characterize race outcomes consists of: 
 - wins
 - top 5 finishes
 - top 10 finishes
 - lead lap finishes
 - laps led*
 
NOTE: Laps led can be a deceiving metric, however, since track lengths vary and the number of laps run at each race therefore also varies. To normalize the laps led metric, we typically convert laps led to percentage of total race laps led.

## Project Description

Calculate the above metrics (and any others you feel are interesting) from the cup_race_results_2014_2024.csv dataset and use them to answer the following questions:

JGR keeps an internal scorecard of season-over-season performance to guide organizational development.  Our most important KPIs are championships won and race wins per season.  However, both championships and wins are relatively rare occurrences.  Wins, for example, are rare because only one competitor can score a win at each race, so evaluating solely on observed wins may not be the best way to capture overall performance or performance potential.

Therefore, we ask you to use the metrics derived above to develop a model that predicts an 'expected' number of wins per season based on those accompanying metrics.  Use those model results to highlight which drivers over-performed (more wins than expected) and under-performed (less wins than expected) across seasons.

Also locate the Cup Series champions for each season 2014-2024 with a simple web search.  Answer the question of how often does the statistically best team for a given season win the championship?

The results dataset consists of rows of data that represent a race finishing result for each entered car at each race from 2014 through 2024.  Basic field descriptions are outlined below:

- result_id: int; unique identifier for each result record
- series_id: int; series identifier (1 = Cup)
- race_season int; race season (year)
- race_id int; id for the race event
- race_date str; scheduled date of the race event
- race_name str; NASCAR official name of the race event
- track_id int; id of the track the race is being held at
- track_name str; name of the track the race is being held at
- actual_laps int; actual race distance of the event (may be different than scheduled if rain-shortened, etc.)
- car_number str; competitor car number
- driver_id int; id of the driver competing with that car number
- driver_fullname str; name of the driver competing with that car number
- team_id int; id of the race organization fielding the car with that car number
- team_name str; name of the race organization fielding the car with that car number
- finishing_position int; finishing position
- diff_laps int; number of laps behind the leader at race conclusion
- diff_time numeric; time behind leader at race conclusion
- finishing_status str; NASCAR-labeled status of the car at race conclusion
- laps_completed int; number of laps completed by the competitor
- laps_led int; number of laps led by the competitor
- starting_position int; race starting position


## CODE DOCUMNETATION

Used Acronyms:
- Joe Gibbs Racing (JGR)	
- Hendrick Motorsports (HMS)	
- Stewart-Haas Racing (SHR)	
- Furniture Row Racing (FRR)


## Current Model Leaderboard

<!-- LEADERBOARD_START -->

**Leaderboard (Top 8 Models)**  
_Last updated: 2025-03-12 14:10:04_

| model_id              | model_type   |     timestamp |   train_RMSE |   test_RMSE |   train_MAE |   test_MAE |   train_R2 |   test_R2 |   CV_RMSE_Mean |   CV_RMSE_Std |   train_test_diff |
|:----------------------|:-------------|--------------:|-------------:|------------:|------------:|-----------:|-----------:|----------:|---------------:|--------------:|------------------:|
| gbm_20250312_1325     | gbm          | 20250312_1325 |     0.233236 |    0.757774 |    0.116723 |   0.372363 |   0.967888 |  0.733576 |       0.736935 |      0.042956 |         -0.524538 |
| gbm_20250312_1409     | gbm          | 20250312_1409 |     0.324494 |    0.774453 |    0.161079 |   0.382465 |   0.937844 |  0.721719 |       0.743694 |      0.042283 |         -0.449959 |
| xgboost_20250312_1325 | xgboost      | 20250312_1325 |     0.007882 |    0.797033 |    0.004325 |   0.399319 |   0.999963 |  0.705255 |       0.765778 |      0.075304 |         -0.789151 |
| ols_20250312_1409     | ols          | 20250312_1409 |     0.621204 |    0.80262  |    0.301909 |   0.400279 |   0.772208 |  0.701109 |       0.693959 |      0.105832 |         -0.181416 |
| xgboost_20250312_1409 | xgboost      | 20250312_1409 |     0.045033 |    0.829678 |    0.02348  |   0.41044  |   0.998803 |  0.680616 |       0.691927 |      0.03426  |         -0.784645 |
| poisson_20250312_1409 | poisson      | 20250312_1409 |     0.717531 |    0.891409 |    0.310055 |   0.41198  |   0.696086 |  0.631322 |       0.693959 |      0.105832 |         -0.173878 |
| zip_20250312_1409     | zip          | 20250312_1409 |     0.675794 |    1.26727  |    0.321196 |   0.500946 |   0.730413 |  0.254866 |     nan        |    nan        |         -0.591479 |
| negbin_20250312_1409  | negbin       | 20250312_1409 |     1.22236  |    1.88578  |    0.380379 |   0.51644  |   0.118005 | -0.649968 |       0.693959 |      0.105832 |         -0.663422 |

<!-- LEADERBOARD_END -->