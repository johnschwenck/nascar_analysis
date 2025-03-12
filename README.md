## JGR Race Results Analysis and Modeling Project

# ===============================
# NASCAR Race Results ML Pipeline
# ===============================
#
# Directory Structure:
# ├── config.yaml
# ├── README.md
# ├── requirements.txt
# ├── src/
# │   ├── __init__.py
# │   ├── nascar_model_pipeline.py
# │   └── utils.py
# ├── data/
# │   └── cup_race_results_2014_2024.csv
# ├── outputs/
# │   ├── leaderboard.csv
# │   └── plots/
# │       └── wins_distribution.png
# └── notebooks/
# ===============================

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
Joe Gibbs Racing (JGR)	
Hendrick Motorsports (HMS)	
Stewart-Haas Racing (SHR)	
Furniture Row Racing (FRR)
