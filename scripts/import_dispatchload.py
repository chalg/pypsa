import requests
import zipfile
import io
import pandas as pd
import pytz


# Function to download and extract dispatch load data from NEMWEB archive
def nemweb_archive_dispatchload(datestring):
    year = datestring[:4]
    month = datestring[4:6]

    base_url = (
        "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        f"{year}/MMSDM_{year}_{month}/MMSDM_Historical_Data_SQLLoader/DATA/"
    )

    url_variants = [
        f"PUBLIC_ARCHIVE%23DISPATCHLOAD%23FILE01%23{datestring}0000.zip",
        f"PUBLIC_DVD_DISPATCHLOAD_{datestring}0000.zip"
    ]

    response = None
    for variant in url_variants:
        try:
            url = base_url + variant
            response = requests.get(url)
            response.raise_for_status()
            print(f"Successfully downloaded from {url}")
            break  # exit loop if successful
        except requests.exceptions.HTTPError as e:
            print(f"URL failed: {url} | Error: {e}")
            continue  # try the next variant

    if response is None or response.status_code != 200:
        raise Exception("All URL variants failed. Could not fetch data.")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            data_file = pd.read_csv(f, header=1, dtype=str)

    if data_file.empty:
        print("Warning: DataFrame is empty after loading.")
        return data_file

    data_file = data_file.drop(index=0).reset_index(drop=True)
    data_file['SETTLEMENTDATE'] = data_file['SETTLEMENTDATE'].str.strip('"')
    data_file['LASTCHANGED'] = data_file['LASTCHANGED'].str.strip('"')

    tz = pytz.timezone("Australia/Brisbane")
    data_file['SETTLEMENTDATE'] = pd.to_datetime(data_file['SETTLEMENTDATE'], format="%Y/%m/%d %H:%M:%S").dt.tz_localize(tz)
    data_file['LASTCHANGED'] = pd.to_datetime(data_file['LASTCHANGED'], format="%Y/%m/%d %H:%M:%S").dt.tz_localize(tz)

    numeric_cols = ['INTERVENTION', 'INITIALMW', 'TOTALCLEARED']
    cols_to_convert = [col for col in numeric_cols if col in data_file.columns]
    data_file[cols_to_convert] = data_file[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    idx = data_file.groupby(['DUID', 'SETTLEMENTDATE'])['INTERVENTION'].idxmax()
    data_file = data_file.loc[idx].reset_index(drop=True)

    return data_file[['DUID', 'SETTLEMENTDATE', 'INITIALMW', 'TOTALCLEARED']]



# for loadiing only one month at a time.
datestring = "20240501"  #01 is required at the end for url format

df = nemweb_archive_dispatchload(datestring)

df = df[['DUID', 'SETTLEMENTDATE', 'INITIALMW', 'TOTALCLEARED']]

print(df.head())       # Show the first few rows
print(df.info())       # Summary of data types and non-null counts

# TODO: still need to download and join
# "https://www.aemo.com.au/-/media/Files/Electricity/NEM/Participant_Information/NEM-Registration-and-Exemption-List.xls" to replicate nemwebR_test.R.

def load_dispatchloads_multiple(dates):
    """
    Load and combine dispatch load data for multiple year-month strings.
    
    Parameters:
    - dates: list of strings in 'YYYYMM' format
    
    Returns:
    - combined pandas DataFrame with all data stacked
    """
    dfs = []
    for date in dates:
        try:
            print(f"Loading data for {date}...")
            df = nemweb_archive_dispatchload(date)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load data for {date}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        print("No data loaded.")
        return pd.DataFrame()  # empty dataframe if no data loaded
    
# Usage: 
# Attempt a whole year of data, but this may take a while (51,743,231 rows).
dates = ['20240101', '20240201', '20240301', '20240401',
         '20240501', '20240601', '20240701', '20240801',
         '20240901', '20241001', '20241101', '20241201']

combined_df = load_dispatchloads_multiple(dates)
print(combined_df.info)
print(combined_df.head())
print(combined_df.tail())

# Save raw data to Feather format for later use
combined_df.to_feather('data/nemweb/raw/dispatchload_2024.feather')

# AEMO spreadsheet processing:
import tempfile
import requests
import pandas as pd
import janitor  

# Download the Excel file

url = "https://www.aemo.com.au/-/media/Files/Electricity/NEM/Participant_Information/NEM-Registration-and-Exemption-List.xls"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
                  "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

file_bytes = io.BytesIO(response.content)

generator_data = pd.read_excel(file_bytes, sheet_name="PU and Scheduled Loads")
generator_data_tbl = generator_data.clean_names()
generator_data_tbl.columns = generator_data_tbl.columns.str.rstrip('_')

# Remove duplicate rows (like distinct())
generator_data_tbl = generator_data_tbl.drop_duplicates()

# Example: assuming df is a DataFrame you already have
# Make sure to clean column names for dispatchload too
dispatchload = combined_df.clean_names()

# Data Wrangling equivalent:
# Filter out rows where 'reg_cap_generation_mw' == "-"
generator_data_tbl = generator_data_tbl[generator_data_tbl['reg_cap_generation_mw'] != "-"]

# Convert 'reg_cap_generation_mw' to numeric
generator_data_tbl['reg_cap_generation_mw'] = pd.to_numeric(generator_data_tbl['reg_cap_generation_mw'])

# Group by 'duid' and sum 'reg_cap_generation_mw' as 'nameplate_cap'
generator_data_tbl['nameplate_cap'] = generator_data_tbl.groupby('duid')['reg_cap_generation_mw'].transform('sum')

# Left join dispatchload with generator_data_tbl on common columns (like left_join)
dispatchload_joined = dispatchload.merge(generator_data_tbl, how='left')

# Remove duplicates after join
dispatchload_joined = dispatchload_joined.drop_duplicates()

# To inspect summary (like glimpse)
print(dispatchload_joined.info())
print(dispatchload_joined.head())

# pandas doesn’t have direct "participant:nameplate_cap" slicing, so:
# Find column positions of 'participant' and 'nameplate_cap'
cols = list(dispatchload_joined.columns)
start_idx = cols.index('participant')
end_idx = cols.index('nameplate_cap')

# Build the list of columns to select
selected_cols = ['settlementdate', 'duid', 'initialmw', 'totalcleared'] + cols[start_idx:end_idx+1]

# Subset the DataFrame
dispatchload_joined = dispatchload_joined[selected_cols]

dispatchload_joined['cf'] = dispatchload_joined.apply(
    lambda row: row['initialmw'] / row['nameplate_cap'] if row['nameplate_cap'] else None,
    axis=1
)

# Optional - half-hour timestamp
dispatchload_joined['datetime_30min'] = dispatchload_joined['settlementdate'].dt.floor('30T')


# Extract hour and date from settlementdate
dispatchload_joined['hour'] = dispatchload_joined['settlementdate'].dt.hour
dispatchload_joined['date'] = dispatchload_joined['settlementdate'].dt.date  # datetime.date object

# Group by the specified columns and calculate mean of cf
group_cols = ['duid', 'date', 'hour', 'region', 'station_name', 'participant', 'fuel_source_primary', 'fuel_source_descriptor']
# Optional: half-hourly aggregation
group_cols = ['datetime_30min', 'region', 'fuel_source_descriptor']

dispatchload_prepared = (
    dispatchload_joined
    .query("fuel_source_descriptor in ['Solar', 'Wind', 'Water']")
    .groupby(group_cols, dropna=False)  # dropna=False keeps NaNs in grouping cols if any
    .agg(mean_cf=('cf', 'mean'))
    .reset_index()
)

# Filter out rows where mean_cf is NA
dispatchload_prepared = dispatchload_prepared[dispatchload_prepared['mean_cf'].notna()]

# Convert to PyPSA pattern for generator names
# e.g., 'NSW1-WIND', 'VIC1-SOLAR', etc
dispatchload_prepared['Generator'] = (
    dispatchload_prepared['region'] + "-" + dispatchload_prepared['fuel_source_descriptor']
).str.upper()

# Replace 'WATER' with 'HYDRO' in generator names
dispatchload_prepared['Generator'] = dispatchload_prepared['Generator'].str.replace(r'-WATER$', '-HYDRO', regex=True)

# Gather only the relevant columns to pivot into wide format
re_cf_30mins = dispatchload_prepared[['datetime_30min', 'Generator', 'mean_cf']]

# Review the first 200 rows of the DataFrame
re_cf_30mins.head(200)

# Pivot the DataFrame to have datetime_30min as index and Generator as columns
re_cf_30mins = re_cf_30mins.pivot(index='datetime_30min', columns='Generator', values='mean_cf')

# Remove timezone information if present
re_cf_30mins.index = re_cf_30mins.index.tz_localize(None)

# Drop last row as it is in 2025 (should be 17,568 rows for leap year 2024)
re_cf_30mins = re_cf_30mins.drop(re_cf_30mins.index[-1])

# Save clean data to Feather format for import into PyPSA ready format
re_cf_30mins.to_feather('data/nemweb/clean/re_cf_30mins_2024.feather')

# # Spot check
# filtered_df = dispatchload_prepared[
#     (dispatchload_prepared['duid'] == "AGLHAL") &
#     (dispatchload_prepared['date'] == pd.to_datetime("2025-06-17").date())
# ]



# # Print up to 24 rows - multiple rows reconcile to R script output.
# filtered_df.head(24)


