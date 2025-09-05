import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import os
import requests
import zipfile
import io
import pyarrow.feather as feather     
# from io import StringIO
# Load the datasets (latest from GBB)
# Data sources: 
# https://nemweb.com.au/Reports/Current/GBB/GasBBActualFlowStorage.zip 
# https://nemweb.com.au/Reports/Current/GBB/GasBBShortTermCapacityOutlook.CSV
# See read_port_pirie_weather_data.py for Port Pirie weather data
# https://data.sa.gov.au/data/dataset/port-pirie-oliver-st-air-quality-monitoring-station-meteorology-data


# 1. Fetch and unzip the first file into 'flow_df'
zip_url = "https://nemweb.com.au/Reports/Current/GBB/GasBBActualFlowStorage.zip"
response = requests.get(zip_url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # Identify CSV files in the archive
    csv_files = [name for name in z.namelist() if name.lower().endswith('.csv')]
    # Read and concatenate all CSVs (if multiple)
    flow_dfs = []
    for fname in csv_files:
        with z.open(fname) as f:
            df = pd.read_csv(
                f,
                parse_dates=['GasDate', 'LastUpdated'],
                dayfirst=True,
            )
            flow_dfs.append(df)
    raw_flow_df = pd.concat(flow_dfs, ignore_index=True)

# 2. Fetch the capacity data CSV directly into 'cap_df'
cap_url = "https://nemweb.com.au/Reports/Current/GBB/GasBBShortTermCapacityOutlook.CSV"
raw_cap_df = pd.read_csv(
    cap_url,
    parse_dates=['GasDate', 'LastUpdated'],
    dayfirst=True,
)

# 3. Inspect the data
print("flow_df preview:")
print(raw_flow_df.head(), "\n")
print("cap_df preview:")
print(raw_cap_df.head())


#---------------------------------------------------
# MAPS Pipeline Utilisation
#---------------------------------------------------

# Data Wrangling for MAPS pipeline utilisation
# Filter MAPS pipeline data
fac, loc = 'MAPS', 'Moomba Hub'
flow_df = raw_flow_df.query(
    "FacilityName == @fac and LocationName == @loc"
)
# flow_df = flow_df.query("GasDate >= '2019-01-01' and GasDate < '2025-01-01'") 

# Calculate flow by summing Supply and TransferIn
flow_df['Flow'] = flow_df['Supply'] + flow_df['TransferIn']

flow_df

# Filter capacity data for MAPS relevant data
recloc = 'Moomba Injection'
cap_df = raw_cap_df.query(
    "FacilityName == @fac and ReceiptLocationName == @recloc"
)

# Join flow data with capacity data
combined_df = flow_df.merge(cap_df[['GasDate', 'FacilityId', 'OutlookQuantity']],
                            on=['GasDate', 'FacilityId'],
                            how='left')

# Calculate utilisation by dividing Flow by OutlookQuantity
combined_df['Utilisation'] = combined_df['Flow'] / combined_df['OutlookQuantity']

# Join weather data
weather_df = pd.read_pickle('data/port_pirie_daily_weather_data.pkl')

combined_df = combined_df.merge(weather_df,
                                left_on='GasDate',
                                right_on='day',
                                how='left')

# Join SA mean daily RRP data
# feather file exported from from R script (delta_prices.R)

prices = feather.read_feather("data/sa_daily_mean_rrp.feather")
prices['date'] = prices['date'].dt.tz_localize(None)
combined_df = combined_df.merge(prices,
                  left_on='GasDate',
                  right_on='date',
                  how='left') \
                      .drop(columns='date')

# Matplotlib plot - reconciles to Pipelines view of the GBB interactive map
plt.figure(figsize=(12, 6))
plt.plot(combined_df['GasDate'], combined_df['Utilisation'], marker='o')
plt.xlabel('GasDate')
plt.ylabel('Utilisation (%)')
plt.title('MAPS Pipeline Utilisation Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True)
plt.tight_layout()
plt.text(
    0.98, 0.02,
    "Reconciles to Pipelines Utilisation column of the GBB interactive map",
    ha='right', va='bottom',
    fontsize=10, fontstyle='italic', color='gray',
    transform=plt.gca().transAxes
)
plt.show()



#---------------------------------------------------
# MAPS Pipeline Utilisation dual y-axis plot with Moving Avg Wind Speed
#---------------------------------------------------

# Start timeseris from 2019-01-01 to 2024-12-31
prepared_df = combined_df.query("GasDate >= '2020-01-01' and GasDate < '2025-12-31'")

# Drop rows with NaNs in either 'day' or 'util_5d_ma' before rolling mean
prepared_df = prepared_df.dropna(subset=['day'])



# Function to create stacked pipeline utilisation and barometric pressure animated GIF

def create_animated_gas_pipeline_gif(prepared_df, output_filename='gifs/gas_pipeline_animation.gif'):
    """
    Creates an animated GIF showing gas pipeline utilisation vs barometric pressure 
    and temperature over time, with data progressively revealing by date.
    
    Parameters:
    prepared_df: DataFrame with columns ['GasDate', 'Utilisation', 'avg_barometric_pressure_hpa', 'avg_temperature_deg_c']
    output_filename: Name of the output GIF file
    """
    

    # --- playback/saving controls ---
    interval_ms = 200        # interactive interval between frames (ms)
    fps = 5                  # GIF frame rate (frames per second)
    repeat_delay_ms = 3000   # desired pause at end of loop (ms)  ### CHANGED

    # 1) Compute rolling means for all variables
    prepared_df = prepared_df.copy()  # Avoid modifying original
    prepared_df['bp_5d_ma'] = (
        prepared_df['avg_barometric_pressure_hpa']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['temp_5d_ma'] = (
        prepared_df['avg_temperature_deg_c']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['util_5d_ma'] = (
        prepared_df['Utilisation']
        .rolling(window=5, min_periods=1)
        .mean()
    )

    # Sort by date to ensure proper animation sequence
    prepared_df = prepared_df.sort_values('GasDate').reset_index(drop=True)

    # 2) Set up the figure with subplots for both charts
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 8))

    # Create twin axes
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    # Initialize empty line objects
    line1, = ax1.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line2, = ax2.plot([], [], linestyle='-', color='tab:green', alpha=0.8, label='5-Day MA Barometric Pressure (hPa)')
    line3, = ax3.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line4, = ax4.plot([], [], linestyle='-', color='tab:purple', alpha=0.8, label='5-Day MA Temperature (°C)')

    # Set up axes properties for top chart (Barometric Pressure) - NO DATE LABEL
    ax1.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Barometric Pressure at Port Pirie')

    ax2.set_ylabel('Barometric Pressure (hPa)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Set up axes properties for bottom chart (Temperature) - DATE LABEL ONLY HERE
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3.grid(True)
    ax3.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Temperature at Port Pirie')

    ax4.set_ylabel('Temperature (°C)', color='tab:purple')
    ax4.tick_params(axis='y', labelcolor='tab:purple')

    # Set axis limits (use full data range)
    ax1.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())
    ax3.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())

    ax1.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)
    ax3.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)

    # Tighter y-axis range for barometric pressure to zoom in
    bp_range = prepared_df['bp_5d_ma'].max() - prepared_df['bp_5d_ma'].min()
    bp_padding = bp_range * 0.1  # 10% padding
    if bp_range == 0:  # avoid identical min/max
        bp_padding = 1.0
    ax2.set_ylim(prepared_df['bp_5d_ma'].min() - bp_padding, prepared_df['bp_5d_ma'].max() + bp_padding)

    ax4.set_ylim(prepared_df['temp_5d_ma'].min() * 0.95, prepared_df['temp_5d_ma'].max() * 1.05)

    # Calculate and add correlations for display
    bp_corr = prepared_df['util_5d_ma'].corr(prepared_df['bp_5d_ma'], method='pearson')
    temp_corr = prepared_df['util_5d_ma'].corr(prepared_df['temp_5d_ma'], method='pearson')

    # Correlation text in top-right of each axes  ### CHANGED
    corr_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax1.text(0.98, 0.98, f'r = {bp_corr:.3f}', transform=ax1.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)
    ax3.text(0.98, 0.98, f'r = {temp_corr:.3f}', transform=ax3.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.25)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=8.25)

    # Add caption
    fig.text(
        0.98, 0.01,
        "Source: AEMO GBB, https://data.sa.gov.au/data/dataset/port-pirie-oliver-st-air-quality-monitoring-station-meteorology-data",
        ha='right', va='bottom',
        fontsize=8,
        fontstyle='italic',
        color='gray'
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.23)

    # Animation function
    def animate(frame):
        # frame is an index into prepared_df (or a repeated last index for the hold)
        end_idx = min(int(frame) + 1, len(prepared_df))

        if end_idx > 0:
            x_data = prepared_df['GasDate'][:end_idx]
            util_data = prepared_df['util_5d_ma'][:end_idx]
            bp_data = prepared_df['bp_5d_ma'][:end_idx]
            temp_data = prepared_df['temp_5d_ma'][:end_idx]

            line1.set_data(x_data, util_data)
            line2.set_data(x_data, bp_data)
            line3.set_data(x_data, util_data)
            line4.set_data(x_data, temp_data)

            current_date = prepared_df['GasDate'].iloc[end_idx - 1].strftime('%Y-%m-%d')
            ax1.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Barometric Pressure at Port Pirie\nCurrent Date: {current_date}')
            ax3.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Temperature at Port Pirie\nCurrent Date: {current_date}')

        return line1, line2, line3, line4

    # Build frame indices
    n_frames = min(len(prepared_df), 100)  # base frames
    base_indices = np.linspace(0, len(prepared_df) - 1, n_frames).astype(int)

    # --- simulate repeat_delay for GIF by padding last frame ---  ### CHANGED
    frame_duration_ms = 1000.0 / fps
    hold_frames = max(1, int(round(repeat_delay_ms / frame_duration_ms)))
    padded_hold = np.full(hold_frames, base_indices[-1], dtype=int)
    frame_indices = np.concatenate([base_indices, padded_hold])

    anim = FuncAnimation(
        fig,
        animate,
        frames=frame_indices,
        interval=interval_ms,     # interactive only
        blit=False,
        repeat=True,
        repeat_delay=repeat_delay_ms  # interactive/HTML only; GIF pause handled by padding
    )

    # Create output directory if it doesn't exist
    outdir = os.path.dirname(output_filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Save as GIF
    print(f"Creating animated GIF: {output_filename}")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer, dpi=150)
    print(f"Animation saved as: {output_filename}")

    return anim


# Example usage:

# Assuming you have your prepared_df ready
anim1 = create_animated_gas_pipeline_gif(prepared_df, 'gifs/gas_pipeline_stacked_animation.gif')

#---------------------------------------------------
# Stacked pipeline utilisation and wind speed animated GIF
#---------------------------------------------------
def create_animated_gas_pipeline_wind_gif(prepared_df, output_filename='gifs/gas_pipeline_wind_animation.gif'):
    """
    Creates an animated GIF showing gas pipeline utilisation vs barometric pressure 
    and wind speed over time, with data progressively revealing by date.
    
    Parameters:
    prepared_df: DataFrame with columns ['GasDate', 'Utilisation', 'avg_barometric_pressure_hpa', 'avg_wind_speed_ms']
    output_filename: Name of the output GIF file
    """
    

    # --- playback/saving controls ---
    interval_ms = 200        # interactive interval between frames (ms)
    fps = 5                  # GIF frame rate (frames per second)
    repeat_delay_ms = 3000   # desired pause at end of loop (ms)

    # 1) Compute rolling means for all variables
    prepared_df = prepared_df.copy()  # Avoid modifying original
    prepared_df['bp_5d_ma'] = (
        prepared_df['avg_barometric_pressure_hpa']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['wind_5d_ma'] = (
        prepared_df['avg_wind_speed_ms']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['util_5d_ma'] = (
        prepared_df['Utilisation']
        .rolling(window=5, min_periods=1)
        .mean()
    )

    # Sort by date to ensure proper animation sequence
    prepared_df = prepared_df.sort_values('GasDate').reset_index(drop=True)

    # 2) Set up the figure with subplots for both charts
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 8))

    # Create twin axes
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    # Initialize empty line objects
    line1, = ax1.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line2, = ax2.plot([], [], linestyle='-', color='tab:green', alpha=0.8, label='5-Day MA Barometric Pressure (hPa)')
    line3, = ax3.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line4, = ax4.plot([], [], linestyle='-', color='tab:orange', alpha=0.8, label='5-Day MA Wind Speed (m/s)')

    # Set up axes properties for top chart (Barometric Pressure) - NO DATE LABEL
    ax1.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Barometric Pressure at Port Pirie')

    ax2.set_ylabel('Barometric Pressure (hPa)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Set up axes properties for bottom chart (Wind Speed) - DATE LABEL ONLY HERE
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3.grid(True)
    ax3.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Wind Speed at Port Pirie')

    ax4.set_ylabel('Wind Speed (m/s)', color='tab:orange')
    ax4.tick_params(axis='y', labelcolor='tab:orange')

    # Set axis limits (use full data range)
    ax1.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())
    ax3.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())

    ax1.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)
    ax3.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)

    # Tighter y-axis range for barometric pressure to zoom in
    bp_range = prepared_df['bp_5d_ma'].max() - prepared_df['bp_5d_ma'].min()
    bp_padding = bp_range * 0.1  # 10% padding
    if bp_range == 0:  # avoid identical min/max
        bp_padding = 1.0
    ax2.set_ylim(prepared_df['bp_5d_ma'].min() - bp_padding, prepared_df['bp_5d_ma'].max() + bp_padding)

    ax4.set_ylim(prepared_df['wind_5d_ma'].min() * 0.95, prepared_df['wind_5d_ma'].max() * 1.05)

    # Calculate and add correlations for display
    bp_corr = prepared_df['util_5d_ma'].corr(prepared_df['bp_5d_ma'], method='pearson')
    wind_corr = prepared_df['util_5d_ma'].corr(prepared_df['wind_5d_ma'], method='pearson')

    # Correlation text in top-right of each axes
    corr_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax1.text(0.98, 0.98, f'r = {bp_corr:.3f}', transform=ax1.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)
    ax3.text(0.98, 0.98, f'r = {wind_corr:.3f}', transform=ax3.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.25)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=8.25)

    # Add caption
    fig.text(
        0.98, 0.01,
        "Source: AEMO GBB, https://data.sa.gov.au/data/dataset/port-pirie-oliver-st-air-quality-monitoring-station-meteorology-data",
        ha='right', va='bottom',
        fontsize=8,
        fontstyle='italic',
        color='gray'
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.23)

    # Animation function
    def animate(frame):
        # frame is an index into prepared_df (or a repeated last index for the hold)
        end_idx = min(int(frame) + 1, len(prepared_df))

        if end_idx > 0:
            x_data = prepared_df['GasDate'][:end_idx]
            util_data = prepared_df['util_5d_ma'][:end_idx]
            bp_data = prepared_df['bp_5d_ma'][:end_idx]
            wind_data = prepared_df['wind_5d_ma'][:end_idx]

            line1.set_data(x_data, util_data)
            line2.set_data(x_data, bp_data)
            line3.set_data(x_data, util_data)
            line4.set_data(x_data, wind_data)

            current_date = prepared_df['GasDate'].iloc[end_idx - 1].strftime('%Y-%m-%d')
            ax1.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Barometric Pressure at Port Pirie\nCurrent Date: {current_date}')
            ax3.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Wind Speed at Port Pirie\nCurrent Date: {current_date}')

        return line1, line2, line3, line4

    # Build frame indices
    n_frames = min(len(prepared_df), 100)  # base frames
    base_indices = np.linspace(0, len(prepared_df) - 1, n_frames).astype(int)

    # --- simulate repeat_delay for GIF by padding last frame ---
    frame_duration_ms = 1000.0 / fps
    hold_frames = max(1, int(round(repeat_delay_ms / frame_duration_ms)))
    padded_hold = np.full(hold_frames, base_indices[-1], dtype=int)
    frame_indices = np.concatenate([base_indices, padded_hold])

    anim = FuncAnimation(
        fig,
        animate,
        frames=frame_indices,
        interval=interval_ms,     # interactive only
        blit=False,
        repeat=True,
        repeat_delay=repeat_delay_ms  # interactive/HTML only; GIF pause handled by padding
    )

    # Create output directory if it doesn't exist
    outdir = os.path.dirname(output_filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Save as GIF
    print(f"Creating animated GIF: {output_filename}")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer, dpi=150)
    print(f"Animation saved as: {output_filename}")

    return anim


# Assuming you have your prepared_df ready
anim2 = create_animated_gas_pipeline_wind_gif(prepared_df, 'gifs/gas_pipeline_wind_animation.gif')


#---------------------------------------------------
# Stacked pipeline utilisation and all weather variables animated GIF   
#---------------------------------------------------

def create_animated_gas_pipeline_all_weather_gif(prepared_df, output_filename='gifs/gas_pipeline_all_weather_animation.gif'):
    """
    Creates an animated GIF showing gas pipeline utilisation vs barometric pressure,
    wind speed, and temperature over time, with data progressively revealing by date.
    
    Parameters:
    prepared_df: DataFrame with columns ['GasDate', 'Utilisation', 'avg_barometric_pressure_hpa', 'avg_wind_speed_ms', 'avg_temperature_deg_c']
    output_filename: Name of the output GIF file
    """
    

    # --- playback/saving controls ---
    interval_ms = 200        # interactive interval between frames (ms)
    fps = 5                  # GIF frame rate (frames per second)
    repeat_delay_ms = 4000   # desired pause at end of loop (ms)

    # 1) Compute rolling means for all variables
    prepared_df = prepared_df.copy()  # Avoid modifying original
    prepared_df['bp_5d_ma'] = (
        prepared_df['avg_barometric_pressure_hpa']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['wind_5d_ma'] = (
        prepared_df['avg_wind_speed_ms']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['temp_5d_ma'] = (
        prepared_df['avg_temperature_deg_c']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    prepared_df['util_5d_ma'] = (
        prepared_df['Utilisation']
        .rolling(window=5, min_periods=1)
        .mean()
    )

    # Sort by date to ensure proper animation sequence
    prepared_df = prepared_df.sort_values('GasDate').reset_index(drop=True)

    # 2) Set up the figure with subplots for all three charts
    fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(8, 12))

    # Create twin axes
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax6 = ax5.twinx()

    # Initialize empty line objects
    line1, = ax1.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line2, = ax2.plot([], [], linestyle='-', color='tab:green', alpha=0.8, label='5-Day MA Barometric Pressure (hPa)')
    line3, = ax3.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line4, = ax4.plot([], [], linestyle='-', color='tab:orange', alpha=0.8, label='5-Day MA Wind Speed (m/s)')
    line5, = ax5.plot([], [], linestyle='-', color='tab:blue', label='5-Day MA Utilisation')
    line6, = ax6.plot([], [], linestyle='-', color='tab:purple', alpha=0.8, label='5-Day MA Temperature (°C)')

    # Set up axes properties for top chart (Barometric Pressure) - NO DATE LABEL
    ax1.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Barometric Pressure at Port Pirie')

    ax2.set_ylabel('Barometric Pressure (hPa)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Set up axes properties for middle chart (Wind Speed) - NO DATE LABEL
    ax3.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3.grid(True)
    ax3.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Wind Speed at Port Pirie')

    ax4.set_ylabel('Wind Speed (m/s)', color='tab:orange')
    ax4.tick_params(axis='y', labelcolor='tab:orange')

    # Set up axes properties for bottom chart (Temperature) - DATE LABEL ONLY HERE
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Pipeline Utilisation (%)', color='tab:blue')
    ax5.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax5.tick_params(axis='y', labelcolor='tab:blue')
    ax5.grid(True)
    ax5.set_title('Moomba to Adelaide Gas Pipeline Utilisation & Temperature at Port Pirie')

    ax6.set_ylabel('Temperature (°C)', color='tab:purple')
    ax6.tick_params(axis='y', labelcolor='tab:purple')

    # Set axis limits (use full data range)
    ax1.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())
    ax3.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())
    ax5.set_xlim(prepared_df['GasDate'].min(), prepared_df['GasDate'].max())

    ax1.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)
    ax3.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)
    ax5.set_ylim(prepared_df['util_5d_ma'].min() * 0.95, prepared_df['util_5d_ma'].max() * 1.05)

    # Tighter y-axis range for barometric pressure to zoom in
    bp_range = prepared_df['bp_5d_ma'].max() - prepared_df['bp_5d_ma'].min()
    bp_padding = bp_range * 0.1  # 10% padding
    if bp_range == 0:  # avoid identical min/max
        bp_padding = 1.0
    ax2.set_ylim(prepared_df['bp_5d_ma'].min() - bp_padding, prepared_df['bp_5d_ma'].max() + bp_padding)

    ax4.set_ylim(prepared_df['wind_5d_ma'].min() * 0.95, prepared_df['wind_5d_ma'].max() * 1.05)
    ax6.set_ylim(prepared_df['temp_5d_ma'].min() * 0.95, prepared_df['temp_5d_ma'].max() * 1.05)

    # Calculate and add correlations for display
    bp_corr = prepared_df['util_5d_ma'].corr(prepared_df['bp_5d_ma'], method='pearson')
    wind_corr = prepared_df['util_5d_ma'].corr(prepared_df['wind_5d_ma'], method='pearson')
    temp_corr = prepared_df['util_5d_ma'].corr(prepared_df['temp_5d_ma'], method='pearson')

    # Correlation text in top-right of each axes
    corr_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax1.text(0.98, 0.98, f'r = {bp_corr:.3f}', transform=ax1.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)
    ax3.text(0.98, 0.98, f'r = {wind_corr:.3f}', transform=ax3.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)
    ax5.text(0.98, 0.98, f'r = {temp_corr:.3f}', transform=ax5.transAxes,
             fontsize=8.25, fontstyle='italic', color='gray', zorder=10,
             ha='right', va='top', bbox=corr_bbox)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.25)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=8.25)

    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    ax5.legend(lines5 + lines6, labels5 + labels6, loc='upper left', fontsize=8.25)

    # Add caption
    fig.text(
        0.98, 0.01,
        "Source: AEMO GBB, https://data.sa.gov.au/data/dataset/port-pirie-oliver-st-air-quality-monitoring-station-meteorology-data",
        ha='right', va='bottom',
        fontsize=8,
        fontstyle='italic',
        color='gray'
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06, hspace=0.30)

    # Animation function
    def animate(frame):
        # frame is an index into prepared_df (or a repeated last index for the hold)
        end_idx = min(int(frame) + 1, len(prepared_df))

        if end_idx > 0:
            x_data = prepared_df['GasDate'][:end_idx]
            util_data = prepared_df['util_5d_ma'][:end_idx]
            bp_data = prepared_df['bp_5d_ma'][:end_idx]
            wind_data = prepared_df['wind_5d_ma'][:end_idx]
            temp_data = prepared_df['temp_5d_ma'][:end_idx]

            line1.set_data(x_data, util_data)
            line2.set_data(x_data, bp_data)
            line3.set_data(x_data, util_data)
            line4.set_data(x_data, wind_data)
            line5.set_data(x_data, util_data)
            line6.set_data(x_data, temp_data)

            current_date = prepared_df['GasDate'].iloc[end_idx - 1].strftime('%Y-%m-%d')
            ax1.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Barometric Pressure at Port Pirie\nCurrent Date: {current_date}')
            ax3.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Wind Speed at Port Pirie\nCurrent Date: {current_date}')
            ax5.set_title(f'Moomba to Adelaide Gas Pipeline Utilisation & Temperature at Port Pirie\nCurrent Date: {current_date}')

        return line1, line2, line3, line4, line5, line6

    # Build frame indices
    n_frames = min(len(prepared_df), 100)  # base frames
    base_indices = np.linspace(0, len(prepared_df) - 1, n_frames).astype(int)

    # --- simulate repeat_delay for GIF by padding last frame ---
    frame_duration_ms = 1000.0 / fps
    hold_frames = max(1, int(round(repeat_delay_ms / frame_duration_ms)))
    padded_hold = np.full(hold_frames, base_indices[-1], dtype=int)
    frame_indices = np.concatenate([base_indices, padded_hold])

    anim = FuncAnimation(
        fig,
        animate,
        frames=frame_indices,
        interval=interval_ms,     # interactive only
        blit=False,
        repeat=True,
        repeat_delay=repeat_delay_ms  # interactive/HTML only; GIF pause handled by padding
    )

    # Create output directory if it doesn't exist
    outdir = os.path.dirname(output_filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Save as GIF
    print(f"Creating animated GIF: {output_filename}")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer, dpi=150)
    print(f"Animation saved as: {output_filename}")

    return anim


# Assuming you have your prepared_df ready
anim3 = create_animated_gas_pipeline_all_weather_gif(prepared_df, 'gifs/gas_pipeline_all_weather_animation.gif')