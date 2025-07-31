# Source: https://github.com/UNSW-CEEM/NEMOSIS/tree/master
# Can try and use similar approach for DISPATCHLOAD
# Data will be much larger as 5-min by DUIDs
# Import rooftop solar data via nemosis and parse for PyPSA 
import pandas as pd
import nemosis
import os
from datetime import datetime

# Review available tables
from nemosis import defaults
print(defaults.dynamic_tables)

# Define start and end time for 2024
start_time = '2024/01/01 00:00:00'
end_time = '2024/12/31 23:30:00'

# Define data directory and ensure it exists
data_folder = 'data/nemweb/clean'
raw_data_folder = 'data/nemweb/raw'
os.makedirs(data_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

# Define table and fields
table_name = 'ROOFTOP_PV_ACTUAL'
fields = ['SETTLEMENTDATE', 'REGIONID', 'POWER']  # Verify if 'POWER' is correct

try:
    # Retrieve rooftop PV actuals from AEMO via NEMOSIS
    # Needed to see what data looks like, hence using fformat='csv'.
    # Using Feather is another option once columns are determined.
    # Couldn't use select_columns='all' with default feather format.

    rooftop_data  = nemosis.dynamic_data_compiler(
    start_time=start_time,
    end_time=end_time,
    table_name=table_name,
    raw_data_location=raw_data_folder,  # Specify the required raw_data_location
    select_columns='all',
    fformat='csv',
    keep_csv=False  # Set to True to keep raw CSV files
    )

    rooftop_data.head(50)

    # Filter for TYPE = 'MEASUREMENT' (change to 'SATELLITE' if needed)
    filtered_data = rooftop_data[rooftop_data['TYPE'] == 'MEASUREMENT']

    # Filter for specific regions
    filtered_data = filtered_data.query('REGIONID in ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]')
    filtered_data

    # Convert timestamps
    filtered_data['INTERVAL_DATETIME'] = pd.to_datetime(filtered_data['INTERVAL_DATETIME'])

    # Pivot into region x time format
    pivot_df = filtered_data[['INTERVAL_DATETIME', 'REGIONID', 'POWER']] \
        .pivot(index='INTERVAL_DATETIME', columns='REGIONID', values='POWER')

    # Resample to hourly resolution (mean of half-hour values)
    hourly_df = pivot_df.resample('1h').mean().round(3)

    # Save to CSV
    output_path = os.path.join(data_folder, "rooftop_solar_hourly_2024.csv")
    hourly_df.to_csv(output_path)
    print(f"Data saved successfully to {output_path}")

except Exception as e:
    print(f"An error occurred: {str(e)}")



# Load uploaded rooftop solar data
rooftop_df = pd.read_csv(f"{data_folder}/rooftop_solar_hourly_2024.csv", parse_dates=["INTERVAL_DATETIME"])
rooftop_df = rooftop_df.set_index("INTERVAL_DATETIME")

rooftop_df.head(20)

# Normalize to get p_max_pu values per region by dividing each column by its own max and fill any NaNs as it will create optimisation warnings
p_max_pu_df = rooftop_df.div(rooftop_df.max()).fillna(0)


# Rename columns to match generator names like 'NSW1-ROOFTOP-SOLAR'
p_max_pu_df.columns = [f"{col}-ROOFTOP-SOLAR" for col in p_max_pu_df.columns]

# Save p_max_pu time series
p_max_pu_path = "data/generators-p_max_pu.csv"
p_max_pu_df.to_csv(p_max_pu_path)

# Build generator entries
generator_entries = []
for region in rooftop_df.columns:
    p_nom = rooftop_df[region].max()
    generator_entries.append({
        "name": f"{region}-ROOFTOP-SOLAR",
        "bus": region,
        "carrier": "Rooftop Solar",
        "p_nom": round(p_nom, 2),
        "marginal_cost": 0
    })

generators_df = pd.DataFrame(generator_entries)
generators_path = "data/rooftop_generators_partial.csv"
generators_df.to_csv(generators_path, index=False)

p_max_pu_path, generators_path



