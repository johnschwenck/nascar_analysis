import os
import pandas as pd
import logging


def safe_read_csv(file_path):
    """
    Attempts to read a CSV with different encodings.
    """
    encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'unicode_escape']
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"File loaded successfully with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            print(f"Encoding {enc} failed... trying next.")

    raise ValueError(f"Failed to load CSV: {file_path}. Tried encodings: {encodings_to_try}")


class ChampionshipEvaluator:
    """
    Evaluates model predictions for NASCAR Cup Series championships by comparing expected
    performance with actual outcomes. It highlights driver over/under-performance and
    compares the statistically best drivers to actual champions.

    Outputs include:
        - Over/Under-performance reports
        - Championship comparison reports
    """

    def __init__(self, config, logger=None):
        """
        Initializes the evaluator.

        Args:
            config (dict): Pipeline configuration.
            logger (Logger, optional): Logger instance. If None, creates default logger.
        """
        self.config = config
        self.logger = logger or self._setup_default_logger()

        champions_path = self.config.get('champions_path')
        if not champions_path or not os.path.exists(champions_path):
            raise FileNotFoundError(f"Champions file not found at: {champions_path}")

        self.champions_df = safe_read_csv(champions_path)


    def _setup_default_logger(self):
        """
        Sets up a default logger if one isn't provided.

        Returns:
            Logger: Configured logger instance.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger("ChampionshipEvaluator")

    def evaluate_championship_predictions(
        self,
        metrics_df,
        model_results,
        entity='driver',  # 'driver' or 'team'
        model_type=None,
        model_id=None
    ):
        """
        Evaluates championship predictions at either the driver or team level.
        
        Args:
            metrics_df (pd.DataFrame): Aggregated metrics (drivers or teams).
            model_results (pd.DataFrame): Expected wins dataframe.
            entity (str): 'driver' or 'team'.
            model_type (str): The type of model used (e.g., 'ols', 'poisson', 'gbm').
            model_id (str): Unique model identifier (e.g., 'poisson_20250311_153426').

        Returns:
            pd.DataFrame: Comparison results (driver/team level).
        """

        assert entity in ['driver', 'team'], "Entity must be 'driver' or 'team'."

        model_desc = f"{model_type} ({model_id})" if model_type and model_id else "Unknown Model"

        self.logger.info(f"\n\n========== Starting {entity.capitalize()}-Level Championship Prediction Evaluation ==========\n")
        self.logger.info(f"Model used for expected wins prediction: {model_desc}\n\n")

        # Define model-specific output directory
        if model_id and model_type:
            output_dir = os.path.join('models', model_type, model_id)
        else:
            output_dir = os.path.join('models', 'error_iters')  # Fallback to error_iters folder if no model_id or model_type
            self.logger.warning("No model_id or model_type provided. Saving outputs to default 'models/error_iters' folder.")

        # Create the output directory if it doesn’t exist
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Using output directory: {output_dir}")

        # Merge actual metrics with expected predictions
        self.logger.info(f"Merging actual {entity} metrics with expected wins predictions...")
        combined_df = self._merge_metrics_and_predictions(metrics_df, model_results)

        # Group and aggregate actual and expected wins
        group_cols = ['race_season', 'driver_fullname'] if entity == 'driver' else ['race_season', 'team_name']
        id_col = 'driver_fullname' if entity == 'driver' else 'team_name'

        self.logger.info(f"Aggregating actual and expected wins at the {entity} level...")
        agg_df = combined_df.groupby(group_cols).agg(
            actual_wins=('wins', 'sum'),
            expected_wins=('expected_wins', 'sum')
        ).reset_index()
        
        # Add model info
        agg_df['model_type'] = model_type
        agg_df['model_id'] = model_id

        # Add win_diff column
        agg_df['win_diff'] = agg_df['actual_wins'] - agg_df['expected_wins']

        # Check for negative expected wins and warn
        if (agg_df['expected_wins'] < 0).any():
            self.logger.warning(f"Negative expected wins detected in {model_desc}. "
                                f"Consider reviewing the model outputs or switching models.")
        
        self.logger.info(f"Aggregated DataFrame shape: {agg_df.shape}")
        self.logger.info(f"Aggregated data:\n{agg_df}")

        # Over/under-performance
        self.logger.info("Calculating over/under-performance...")
        agg_df['performance_diff'] = agg_df['actual_wins'] - agg_df['expected_wins']
        agg_df['over_under'] = agg_df['performance_diff'].apply(
            lambda x: 'Over' if x > 0 else 'Under' if x < 0 else 'Expected'
        )

        # Sorting logic from config
        sort_by = self.config['output'].get(f'OU_sort_column', 'win_diff')
        ascending = self.config['output'].get(f'OU_sort_ascending', False)

        if sort_by in agg_df.columns:
            agg_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
            self.logger.info(f"Sorted {entity}-level results by '{sort_by}' (ascending={ascending})")
        else:
            self.logger.warning(f"Sort column '{sort_by}' not found. Skipping sort.")

        # Save over/under-performance report to model-specific folder
        output_file = os.path.join(
            output_dir, 'analysis',
            f"{entity}_over_under_performance.csv"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        agg_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {entity}-level over/under-performance report to: {output_file}")

        # Find statistical champion (highest expected wins)
        self.logger.info("Identifying top expected performer (statistical champion)...")
        top_expected_df = agg_df.groupby('race_season').apply(
            lambda group: group.sort_values(by='expected_wins', ascending=False).iloc[0]
        ).reset_index(drop=True)

        top_expected_col = f'expected_champion_{entity}'
        top_expected_df.rename(columns={id_col: top_expected_col}, inplace=True)

        # self.logger.info(f"Top expected {entity}s:\n{top_expected_df[['race_season', top_expected_col, 'expected_wins']]}")

        # Load champions data from champions_df (already read during init)
        champions_df = self.champions_df.copy()

        # Ensure race_season is int for both DataFrames
        self._ensure_column_type(top_expected_df, 'race_season', int)
        self._ensure_column_type(champions_df, 'race_season', int)

        # Define actual columns for merge
        if entity == 'driver':
            actual_cols = ['race_season', 'actual_champion']
        else:
            actual_cols = ['race_season', 'actual_champion_team']

        # Clean columns for consistent comparison
        self.logger.info(f"Stripping whitespace and ensuring consistency in {actual_cols}...")
        top_expected_df[top_expected_col] = top_expected_df[top_expected_col].astype(str).str.strip()
        for col in actual_cols[1:]:
            champions_df[col] = champions_df[col].astype(str).str.strip()

        # Merge top expected with actual champions
        # self.logger.info(f"Expected Winning {entity}s vs Actual Winning {entity}s...")
        comparison_df = pd.merge(
            top_expected_df[['race_season', top_expected_col, 'expected_wins']],
            champions_df[actual_cols],
            on='race_season',
            how='left'
        ).reset_index(drop=True)

        self.logger.info(f"\n\nExpected Winning {entity}s vs Actual Winning {entity}s by Year:\n{comparison_df}")

        # Match flag
        match_col = 'actual_champion' if entity == 'driver' else 'actual_champion_team'
        comparison_df['match'] = comparison_df[top_expected_col] == comparison_df[match_col]

        # Log match stats
        success_count = comparison_df['match'].sum()
        total_seasons = len(comparison_df)
        success_rate = (success_count / total_seasons) * 100
        self.logger.info(
            f"{entity.capitalize()}-level success rate: {success_rate:.2f}% "
            f"({success_count} out of {total_seasons} seasons)"
        )

        # Save championship comparison report to model-specific folder
        champion_comp_file = os.path.join(
            output_dir, 'analysis',
            f"{entity}_championship_comparison.csv"
        )
        comparison_df.to_csv(champion_comp_file, index=False)
        self.logger.info(f"\nSaved {entity}-level championship comparison report to: {champion_comp_file}")

        self.logger.info(f"\n\n========== Completed {entity.capitalize()}-Level Championship Prediction Evaluation ==========\n")

        return comparison_df


    def _merge_metrics_and_predictions(self, metrics_df, model_results):
        """
        Merges actual metrics with model predictions for expected wins.

        Args:
            metrics_df (pd.DataFrame): Aggregated driver metrics.
            model_results (pd.DataFrame): Expected wins per driver.

        Returns:
            pd.DataFrame: Merged DataFrame with actual and expected wins.
        """
        merged_df = pd.merge(
            # metrics_df[['race_season', 'driver_id', 'driver_fullname', 'team_name', 'wins']],
            metrics_df,
            model_results[['race_season', 'driver_id', 'expected_wins']],
            on=['race_season', 'driver_id'],
            how='left'
        )

        # Ensure no missing expected_wins
        merged_df['expected_wins'] = merged_df['expected_wins'].fillna(0)

        self.logger.info("Merged actual metrics with expected wins.")
        return merged_df

    def _find_top_expected_drivers(self, combined_df):
        """
        Identifies the driver with the highest expected wins for each season.

        Args:
            combined_df (pd.DataFrame): Merged actual and expected wins DataFrame.

        Returns:
            pd.DataFrame: DataFrame of top expected drivers per season.
        """
        top_expected = combined_df.groupby('race_season').apply(
            lambda group: group.sort_values(by='expected_wins', ascending=False).iloc[0]
        ).reset_index(drop=True)

        top_expected.rename(columns={'driver_fullname': 'expected_champion'}, inplace=True)

        self.logger.info("Identified top expected performers (statistical champions) for each season.")
        return top_expected

    def evaluate_team_championship_predictions(self, metrics_df, model_results):
        """
        Evaluates championship predictions at the team level by comparing predicted team performance
        to actual outcomes. Highlights team over/under-performance and identifies top teams by expected wins.

        Args:
            metrics_df (pd.DataFrame): Aggregated driver metrics from preprocess_data(), including actual wins.
            model_results (pd.DataFrame): DataFrame containing expected wins for each driver.
                                        Must include: ['race_season', 'driver_id', 'expected_wins']

        Returns:
            pd.DataFrame: DataFrame with team-level championship comparison results.
        """
        self.logger.info("Starting team-level championship prediction evaluation...")

        # Merge actual metrics with model predictions
        combined_df = self._merge_metrics_and_predictions(metrics_df, model_results)

        # TEAM-LEVEL AGGREGATION
        team_agg_df = combined_df.groupby(['race_season', 'team_name']).agg(
            actual_wins=('wins', 'sum'),
            expected_wins=('expected_wins', 'sum')
        ).reset_index()

        # Over/under-performance by team
        team_agg_df['performance_diff'] = team_agg_df['actual_wins'] - team_agg_df['expected_wins']
        team_agg_df['over_under'] = team_agg_df['performance_diff'].apply(
            lambda x: 'Over' if x > 0 else 'Under' if x < 0 else 'Expected'
        )

        # Save over/under-performance results for teams
        team_over_under_output = self.config['output'].get(
            'team_over_under_performance_csv',
            'outputs/team_over_under_performance.csv'
        )
        os.makedirs(os.path.dirname(team_over_under_output), exist_ok=True)
        team_agg_df.to_csv(team_over_under_output, index=False)
        self.logger.info(f"Team Over/Under performance report saved to {team_over_under_output}")

        # Find the "statistical champion team" (team with highest expected wins) per season
        top_expected_teams_df = team_agg_df.groupby('race_season').apply(
            lambda group: group.sort_values(by='expected_wins', ascending=False).iloc[0]
        ).reset_index(drop=True)

        top_expected_teams_df.rename(columns={'team_name': 'expected_champion_team'}, inplace=True)

        # Load actual Cup champions for comparison (if you have team data)
        champions_path = self.config.get('champions_path')
        if not champions_path or not os.path.exists(champions_path):
            raise FileNotFoundError(f"Champions file not found at: {champions_path}")

        champions_df = self.champions_df.copy()

        # Merge top teams with actual champion teams
        comparison_df = pd.merge(
            top_expected_teams_df[['race_season', 'expected_champion_team', 'expected_wins']],
            champions_df[['race_season', 'actual_champion', 'actual_champion_team']],
            on='race_season',
            how='left'
        ).reset_index(drop = True)

        # Clean up string fields for accurate comparison
        comparison_df['expected_champion_team'] = comparison_df['expected_champion_team'].astype(str).str.strip()
        comparison_df['actual_champion_team'] = comparison_df['actual_champion_team'].astype(str).str.strip()

        # Determine if the expected team matches the actual champion's team
        comparison_df['match'] = (
            comparison_df['expected_champion_team'].astype(str).str.strip() ==
            comparison_df['actual_champion_team'].astype(str).str.strip()
        )

        self.logger.info("Expected Teams vs Actual Teams:\n{}".format(comparison_df[['race_season', 'actual_champion_team', 'expected_champion_team', 'match']]))

        # Success rate calculation
        success_rate = comparison_df['match'].mean() * 100
        self.logger.info(f"Team-level success rate: {success_rate:.2f}% ({comparison_df['match'].sum()} out of {len(comparison_df)} seasons)")


        # Save championship comparison report for teams
        team_champ_comparison_output = self.config['output'].get(
            'team_championship_comparison_csv',
            'outputs/team_championship_comparison.csv'
        )
        comparison_df.to_csv(team_champ_comparison_output, index=False)
        self.logger.info(f"Team Championship comparison report saved to {team_champ_comparison_output}")

        return comparison_df

    def _ensure_column_type(self, df, column, dtype=int):
        """
        Ensures that a DataFrame column has the specified data type.
        Args:
            df (pd.DataFrame): The DataFrame to modify.
            column (str): The column name.
            dtype (type): The target data type (default: int).
        """
        if column not in df.columns:
            self.logger.warning(f"Column '{column}' not found in DataFrame.")
            return

        current_dtype = df[column].dtype

        # Clean non-numeric entries if converting to int
        if dtype == int:
            # Log unique problematic values
            invalid_rows = df[~df[column].astype(str).str.strip().str.match(r'^-?\d+$')]
            if not invalid_rows.empty:
                self.logger.warning(f"Found non-numeric entries in column '{column}':")
                self.logger.warning(f"{invalid_rows[[column]]}")

            # Convert non-numeric to NaN, then fill or drop
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df.dropna(subset=[column], inplace=True)  # Drop rows with invalid values
            df[column] = df[column].astype(dtype)
            self.logger.info(f"Cleaned and converted column '{column}' to {dtype}.")
        else:
            # Try direct conversion for non-int types
            try:
                df[column] = df[column].astype(dtype)
                self.logger.info(f"Column '{column}' converted from {current_dtype} to {dtype}.")
            except Exception as e:
                self.logger.error(f"Failed to convert column '{column}' to {dtype}: {e}")
