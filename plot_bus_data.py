#!/usr/bin/env python3
"""
Plot voltage measurements for a specific bus over a day from the generated FDIA dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os

def load_dataset(dataset_path):
    """Load the FDIA dataset"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def get_measurement_columns(df, measurement_index):
    """Get measurement columns for a specific measurement index"""
    # Look for z_normal and z_attacked columns for the given measurement index
    normal_col = f'z_normal_{measurement_index}'
    attacked_col = f'z_attacked_{measurement_index}'
    
    available_cols = []
    if normal_col in df.columns:
        available_cols.append(normal_col)
    if attacked_col in df.columns:
        available_cols.append(attacked_col)
    
    if not available_cols:
        # Show available measurement columns
        measurement_cols = [col for col in df.columns if col.startswith('z_normal_') or col.startswith('z_attacked_')]
        max_measurements = len([col for col in df.columns if col.startswith('z_normal_')])
        print(f"No measurement columns found for index {measurement_index}")
        print(f"Available measurement indices: 0 to {max_measurements-1}")
        return []
    
    print(f"Found measurement columns for index {measurement_index}: {available_cols}")
    return available_cols

def create_time_index(df, start_time="2014-04-01 00:00:00", time_interval_hours=1):
    """Create a datetime index for the dataset"""
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    time_index = pd.date_range(
        start=start_dt, 
        periods=len(df), 
        freq=f"{time_interval_hours}h"  # Use lowercase 'h' for hours
    )
    return time_index

def plot_measurement_data(df, measurement_index, output_path=None, day_start=0, num_days=1):
    """
    Plot measurement data for a specific measurement index over specified days
    
    Parameters:
    - df: DataFrame with the dataset
    - measurement_index: Measurement index to plot (0-based)
    - output_path: Path to save the plot (optional)
    - day_start: Starting day (0-based index)
    - num_days: Number of days to plot
    """
    
    # Get measurement columns for the specified index
    measurement_cols = get_measurement_columns(df, measurement_index)
    if not measurement_cols:
        return
    
    # Create time index (assuming 1-hour intervals)
    time_index = create_time_index(df)
    
    # Calculate data points per day (24 hours per day)
    points_per_day = 24
    start_idx = day_start * points_per_day
    end_idx = start_idx + (num_days * points_per_day)
    
    # Ensure we don't exceed dataset bounds
    end_idx = min(end_idx, len(df))
    
    if start_idx >= len(df):
        print(f"Error: Day {day_start} exceeds dataset length")
        return
    
    # Subset the data
    plot_data = df.iloc[start_idx:end_idx]
    plot_time = time_index[start_idx:end_idx]
    fdia_labels = plot_data.get('fdia_label', [0] * len(plot_data))
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    
    # Plot each measurement (normal and attacked if available)
    colors = ['blue', 'red']
    labels = []
    
    for i, col in enumerate(measurement_cols):
        measurement_data = plot_data[col]
        color = colors[i % len(colors)]
        
        if 'normal' in col:
            plt.plot(plot_time, measurement_data, linewidth=0.8, alpha=0.8, color=color, label='Normal')
        else:
            plt.plot(plot_time, measurement_data, linewidth=0.8, alpha=0.8, color=color, label='Attacked', linestyle='--')
    
    # Highlight attack periods if available
    if 'fdia_label' in df.columns:
        attack_periods = plot_data['fdia_label'] == 1
        if attack_periods.any():
            # Add background shading for attack periods
            for i in range(len(plot_time)):
                if attack_periods.iloc[i]:
                    plt.axvspan(plot_time[i], plot_time[i] + timedelta(hours=1), 
                              alpha=0.2, color='yellow', zorder=0)
    
    plt.title(f'Power System Measurement {measurement_index} - Day {day_start + 1}', fontsize=14, fontweight='bold')
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Measurement Value (p.u.)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show only hours
    from matplotlib.dates import DateFormatter
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H'))
    plt.xticks(rotation=0)  # No rotation needed for just hours
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save or show the plot
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
    
    try:
        # Load the dataset
        df = load_dataset(args.dataset)
        
        # Create the plot
        output_path = args.output or f'outputs/measurement_{args.measurement}_day_{args.day}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plot_measurement_data(df, args.measurement, output_path, args.day, args.days)
        
        # Print dataset info
        print(f"\nDataset Info:")
        print(f"  Total samples: {len(df)}")
        print(f"  Total days: {len(df) / 24:.1f}")  # 24 hours per day
        print(f"  Attack samples: {df['fdia_label'].sum() if 'fdia_label' in df.columns else 'N/A'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
