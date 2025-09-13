import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import argparse
import os

def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    return df

def get_measurement_columns(df, measurement_index):
    normal_col = f'z_normal_{measurement_index}'
    attacked_col = f'z_attacked_{measurement_index}'
    
    available_cols = []
    if normal_col in df.columns:
        available_cols.append(normal_col)
    if attacked_col in df.columns:
        available_cols.append(attacked_col)
    
    return available_cols

def create_time_index(df, start_time="2014-04-01 00:00:00", time_interval_hours=1):
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    time_index = pd.date_range(
        start=start_dt, 
        periods=len(df), 
        freq=f"{time_interval_hours}h"
    )
    return time_index

def plot_measurement_data(df, measurement_index, output_path=None, day_start=0, num_days=1):
    measurement_cols = get_measurement_columns(df, measurement_index)
    
    time_index = create_time_index(df)
    
    points_per_day = 24
    start_idx = day_start * points_per_day
    end_idx = start_idx + (num_days * points_per_day)
    
    end_idx = min(end_idx, len(df))
    
    if start_idx >= len(df):
        print(f"Error: Day {day_start} exceeds dataset length")
        return
    
    plot_data = df.iloc[start_idx:end_idx]
    plot_time = time_index[start_idx:end_idx]
    
    plt.figure(figsize=(15, 6))
    
    colors = ['blue', 'red']
    
    for i, col in enumerate(measurement_cols):
        measurement_data = plot_data[col]
        color = colors[i % len(colors)]
        
        if 'normal' in col:
            plt.plot(plot_time, measurement_data, linewidth=0.8, alpha=0.8, color=color, label='Normal')
        else:
            plt.plot(plot_time, measurement_data, linewidth=0.8, alpha=0.8, color=color, label='Attacked', linestyle='--')
    
    if 'fdia_label' in df.columns:
        attack_periods = plot_data['fdia_label'] == 1
        if attack_periods.any():
            for i in range(len(plot_time)):
                if attack_periods.iloc[i]:
                    plt.axvspan(plot_time[i], plot_time[i] + timedelta(hours=1), 
                              alpha=0.2, color='yellow', zorder=0)
    
    plt.title(f'Power System Measurement {measurement_index} - Day {day_start + 1}', fontsize=14, fontweight='bold')
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Measurement Value (p.u.)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H'))
    plt.xticks(rotation=0)
    
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot measurement data from FDIA dataset')
    parser.add_argument('--dataset', '-d', 
                       default='data/processed/ieee13_fdia_dataset.csv',
                       help='Path to the FDIA dataset CSV file')
    parser.add_argument('--measurement', '-m', type=int, default=0,
                       help='Measurement index to plot (default: 0)')
    parser.add_argument('--day', type=int, default=0,
                       help='Day to start plotting (0-based, default: 0)')
    parser.add_argument('--days', type=int, default=1,
                       help='Number of days to plot (default: 1)')
    parser.add_argument('--output', '-o',
                       help='Output path for the plot (optional)')
    
    args = parser.parse_args()
    
    df = load_dataset(args.dataset)
    
    output_path = args.output or f'outputs/measurement_{args.measurement}_day_{args.day}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plot_measurement_data(df, args.measurement, output_path, args.day, args.days)

if __name__ == "__main__":
    main()
