import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

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
