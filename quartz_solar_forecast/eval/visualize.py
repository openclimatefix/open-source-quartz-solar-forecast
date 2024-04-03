import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import re

def visualize_results(results_df):
    '''
    Visualize Eval Results

    results_df dataframe with the following columns
    - timestamp
    - pv_id
    - horizon_hours
    - forecast_power
    - generation_power
    '''

    # Ensure results folder exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Prediction vs Actual values for each PV ID
    pv_ids = results_df['pv_id'].unique()
    for pv_id in pv_ids:
        df_subset = results_df[results_df['pv_id'] == pv_id]
        num_plots = len(df_subset) // 48  # Find out how many full 48-hour plots we can make

        for i in range(num_plots):
            # Extract data for the ith 48-hour segment
            df_segment = df_subset.iloc[i*48:(i+1)*48].copy()
            
            start_timestamp = df_segment.iloc[0]['timestamp']
            start_timestamp_str = re.sub(r"[\s\-:]", "_", start_timestamp)
            df_segment['timestamp'] = pd.to_datetime(df_segment['timestamp'])

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_segment, x='timestamp', y='generation_power', label='Actual', marker='o')
            sns.lineplot(data=df_segment, x='timestamp', y='forecast_power', label='Predicted', marker='o')

            # Setting the x-ticks and labels
            ax = plt.gca()  # Get the current Axes instance
            x_ticks = pd.date_range(start=df_segment['timestamp'].iloc[0], periods=len(df_segment), freq='h')
            ax.set_xticks(x_ticks)  # Set x-ticks to every hour

            # Format the x-tick labels to show every 5 hours
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=5))  # Show label every 5 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format the datetime
            ax.xaxis.set_minor_locator(mdates.HourLocator())  # Keep a tick for every hour

            # Plot title, labels, legend, and tight layout
            plt.title(f'48HRs Predicted vs. Actual Solar Power Output for {pv_id} (Starting from {start_timestamp})')
            plt.xlabel('Timestamp')
            plt.ylabel('Power Output (kW)')
            plt.legend()
            plt.tight_layout()

            # Save the figure
            plt.savefig(f'results/pred_vs_actual_{pv_id}_{start_timestamp_str}.png')
            plt.close()

    # Distribution of Errors across all data points
    results_df['error'] = results_df['forecast_power'] - results_df['generation_power']
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['error'], kde=True, bins=30)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (kW)')
    plt.ylabel('Frequency')
    plt.savefig('results/error_distribution.png')
    plt.close()