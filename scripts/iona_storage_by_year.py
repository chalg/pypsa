# Source: https://nemweb.com.au/Reports/Current/GBB/GasBBActualFlowStorage.zip
# Re-importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import zipfile
import requests
import io

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
    df = pd.concat(flow_dfs, ignore_index=True)



df_storage = df[
    (df['FacilityType'] == "STOR") &
    (df['FacilityName'] == "Iona UGS") &
    (df['GasDate'].dt.year >= 2019)
]

# Group and prepare data
df_grouped = df_storage.groupby('GasDate')['HeldInStorage'].sum().reset_index()
df_grouped['Year'] = df_grouped['GasDate'].dt.year
df_grouped['Month-Day'] = df_grouped['GasDate'].dt.strftime('%m-%d')

# Pivot data
df_pivot = df_grouped.pivot(index='Month-Day', columns='Year', values='HeldInStorage')

# Sorting based on the standard year
sorted_month_days = pd.date_range("2021-01-01", "2021-12-31").strftime('%m-%d')
df_pivot = df_pivot.reindex(sorted_month_days)

# Plot the actual dataset
plt.figure(figsize=(8, 5))

for year in df_pivot.columns:
    linewidth = 4 if year in [2023, 2024] else 1.5
    alpha = 1 if year in [2023, 2024] else 0.75
    plt.plot(df_pivot.index,
             df_pivot[year],
             label=f'{year}',
             linewidth=linewidth,
             alpha=alpha)

plt.title('Gas Storage Levels at Iona Under Ground Storage by Year')
# plt.xlabel('Month')
plt.ylabel('Storage (TJ)')
plt.legend(title='Year')
plt.grid(True)

# Add caption
# Replace the plt.text() call with:
plt.figtext(
    0.99, 0.02,
    "Source: AEMO GBB: https://nemweb.com.au/Reports/Current/GBB/GasBBActualFlowStorage.zip",
    ha='right', va='bottom',
    fontsize=8,
    fontstyle='italic',
    color='gray'
)

month_positions = pd.date_range("2021-01-01", periods=12, freq='MS').strftime('%m-%d')
month_labels = pd.date_range("2021-01-01", periods=12, freq='MS').strftime('%b')
plt.xticks(ticks=[df_pivot.index.get_loc(md) for md in month_positions if md in df_pivot.index], labels=month_labels, rotation=0)

plt.tight_layout()
# And adjust the bottom margin:
plt.subplots_adjust(top=0.95, bottom=0.12)

plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()