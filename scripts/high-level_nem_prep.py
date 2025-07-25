# Prepare NEM demand (loads_t) timeseries data for PyPSA network
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pypsa
from pypsa.plot import add_legend_patches     
from datetime import timedelta
from datetime import datetime  
from dateutil.parser import parse
import pyarrow.feather as feather
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
from time import perf_counter


# Read in feather files generated in R
# Load demand (load) data from the Feather file at 5 minute resolution
loads_time_series_df = feather.read_feather('data/nem_demand_2024.feather')

# Pivot so regionid is across the top (columns)
loads_time_series_df = loads_time_series_df.pivot(index='settlementdate', columns='regionid', values='totaldemand')

# Setting temporal resolution, this can be adjusted as needed
freq = '1h'

# Resample the time series data to 1-hourly resolution
loads_time_series_df = loads_time_series_df.resample(freq).mean().round(2)

loads_time_series_df.index = loads_time_series_df.index.tz_localize(None)

# TODO: generate more robust data pipeline using python only so it is more reproducible. Can't find library to automate https://nemweb.com.au/Reports/Current/Public_Prices/ import for regional demand.
# Use nemosis in similar fashion to the import_rooftop_solar script...

# Load renewable energy CF data from the Feather file
renewable_cf_df = feather.read_feather('data/re_cf_2024.feather')

# Pivot so Generator is across the top (columns)
renewable_cf_df = renewable_cf_df.pivot(index='datetime', columns='Generator', values='regional_hrly_cf')

# Resample the time series data to 1-hourly resolution
renewable_cf_df = renewable_cf_df.resample(freq).mean().round(3)

renewable_cf_df.index = renewable_cf_df.index.tz_localize(None)

# JOIN rooftop solar from import_rooftop_solar script
p_max_pu_path = "data/generators-p_max_pu.csv"
p_max_pu_df = pd.read_csv(p_max_pu_path, index_col=0, parse_dates=True)

# Join addditional rooftop solar cap factor data
renewable_cf_df = renewable_cf_df.merge(
    p_max_pu_df,
    how='left',
    left_on='datetime',   # column in left DF
    right_index=True      # use index of right DF
)

loads_time_series_df.shape, renewable_cf_df.shape

# Drop extra day off loads time series - must be same length as generators time series
loads_time_series_df = loads_time_series_df.drop(loads_time_series_df.index[-1])
loads_time_series_df.shape, renewable_cf_df.shape
# Save the dataframes to CSV files for PyPSA
# Filename equates to how to view in PyPSA
loads_time_series_df.to_csv('data/loads_t.p_set.csv', index_label="")
renewable_cf_df.to_csv('data/generators_t.p_max_pu.csv', index_label="")

# JOIN rooftop solar generators (non-timeseries) to other generators
# Note: generators.csv source: Open Electricity Facilities, status: operating & committed (Snowy 2.0 included).
# Note: 600MW battery added in TAS1 as starting point for scale-up, even though it is not yet built. 
# Hydro in Victoria example: https://explore.openelectricity.org.au/facilities/vic1/?tech=hydro&status=operating,committed
generators = pd.read_csv("data/generators.csv")
rooftop_generators_partial = pd.read_csv("data/rooftop_generators_partial.csv")

# Concatenate and reset index
generators = pd.concat([generators, rooftop_generators_partial],
                       ignore_index=True) \
                        .set_index('name')

# Pushing to csv will concatenate duplicate rooftop solar generators and this type of warning will become apparent:
# WARNING:pypsa.io:The following generators are already defined and will be skipped (use overwrite=True to overwrite): NSW1-ROOFTOP-SOLAR
# generators.to_csv('data/generators.csv') #.set_index('name')

n = pypsa.Network()

# Load remaining static data
# generators already in memory, so no need to load again
# generators = pd.read_csv("data/generators.csv").set_index('name')
buses = pd.read_csv("data/buses.csv", index_col=0)
loads = pd.read_csv("data/loads.csv", index_col=0)
lines = pd.read_csv("data/lines.csv", index_col=0)
links = pd.read_csv("data/links.csv", index_col=0)

# Source: Facilities in the Open Electricity.
# SA example: https://explore.openelectricity.org.au/facilities/sa1/?selected=LBBESS&tech=battery_discharging&status=operating,committed
# max_hours = nominal energy capacity (MWh) / p_nom (MW) * 0.8 (allow for not fully charging or discharging)
# Store and dispatch efficiency via 2024 ISP Inputs and Assumptions workbook,
# Storage properties sheet
storage_units = pd.read_csv("data/storage_units.csv", index_col=0)

# Load time series data
load_ts = pd.read_csv("data/loads_t.p_set.csv", index_col=0)
load_ts.index = pd.to_datetime(load_ts.index, errors="raise")
print(load_ts.index.dtype) 
# Below should be empty if all dates are valid
load_ts[~load_ts.index.notnull()]
generators_ts = pd.read_csv("data/generators_t.p_max_pu.csv", index_col=0)
generators_ts.index = pd.to_datetime(generators_ts.index, errors="raise")
print(generators_ts.index.dtype)
# Below should be empty if all dates are valid
generators_ts[~generators_ts.index.notnull()]
# Check if the time series indices are aligned
if not load_ts.index.equals(generators_ts.index):    
    raise ValueError("Time series indices are not aligned")

# Add components
for name, row in buses.iterrows():
    n.add("Bus", name, **row.to_dict())

for name, row in loads.iterrows():
    n.add("Load", name, **row.to_dict())

for name, row in lines.iterrows():
    n.add("Line", name, **row.to_dict())

for name, row in generators.iterrows():
    n.add("Generator", name, **row.to_dict())

for name, row in links.iterrows():
    n.add("Link", name, **row.to_dict())
    
for name, row in storage_units.iterrows():
    n.add("StorageUnit", name, **row.to_dict())    


# Set time series
n.set_snapshots(load_ts.index)
n.loads_t.p_set = load_ts
n.generators_t.p_max_pu = generators_ts

assert all(n.generators_t.p_max_pu.index == n.snapshots)

# Remove negative values in p_max_pu (Hydro is the culprit)
n.generators_t.p_max_pu = n.generators_t.p_max_pu.clip(lower=0.0, upper=1.0)

# Add one unserved energy generator per load bus
# Acts like a dummy "load shedding" generator
for bus in n.loads.bus.unique():
    gen_name = f"{bus}-UNSERVED"
    n.add("Generator",
          name=gen_name,
          bus=bus,
          carrier="Unserved Energy",
          p_nom_extendable=True,
          p_nom=0,
          marginal_cost=10000,  # Very expensive fallback
          capital_cost=0,       # Optional: make it purely operational cost
    )


# Diagnostic output
print(f"Loaded {len(n.buses)} buses")
print(f"Loaded {len(n.loads)} loads with time series of shape {n.loads_t.p_set.shape}")
print(f"Loaded {len(n.generators)} generators with time series of shape {n.generators_t.p_max_pu.shape}")
print(f"Loaded {len(n.lines)} lines")
print(f"Loaded {len(n.links)} links")
print(f"Loaded {len(n.generators)} generators")
print(f"Loaded {len(n.storage_units)} storage units")


# Review the network components & timeseries
n.buses
n.lines
n.links
n.generators
n.loads
n.storage_units
# Timeseries
# Demand (loads)
n.loads_t.p_set
# Capacity or availability factors for RE generators
n.generators_t.p_max_pu


# Check total demand makes sense
n.loads_t.p_set.sum(axis=1).max()

# Tell PyPSA: “if there's no water/wind/sun, it's okay to produce nothing.”
# Set by default, no need to set again
# n.generators.loc[n.generators.carrier.isin(["Hydro", "Wind", "Solar"]), "p_min_pu"] = 0.0

# Add carrier colours for plotting
# Don't try to use UK spelling for "colour"!!
carrier_list = n.generators.carrier.unique()
carrier_colors = {
    "Biomass": '#127E2A',
    "Hydro": '#1E81D4',
    "Black Coal": "#39322D",
    "Solar": '#FDB324',
    "Rooftop Solar": '#FFE066',
    "Wind": '#3BBFE5',
    "Diesel": "#D486BA",
    "Brown Coal": "#715E50",
    "Gas": '#E6622D',
    "ROR": '#8ab2d4',
    "Battery": '#814ad4',
    "Pump Hydro": '#104775',
    "AC": '#999999',
    "DC": "#3277AF",
    "Unserved Energy": "#F40B16"
}

for carrier, color in carrier_colors.items():
    n.carriers.loc[carrier, 'color'] = color


# Plot high-level NEM network (buses, lines & links)

# Use PlateCarree projection for lat/lon
crs = ccrs.PlateCarree()

# Create figure and map
fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={'projection': crs})

# Add base map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.set_extent([110, 155, -45, -10])  # Australia-wide view

# Plot buses
ax.scatter(
    n.buses["x"], n.buses["y"],
    color='red', s=50, transform=crs, zorder=5,
    label="Buses"
)

# Label each bus
for name, row in n.buses.iterrows():
    ax.text(
        row["x"], row["y"], name,
        transform=crs, fontsize=9, ha='left', va='bottom'
    )

# Plot lines (AC)
first_line = True
for _, line in n.lines.iterrows():
    x0, y0 = n.buses.loc[line.bus0, ["x", "y"]]
    x1, y1 = n.buses.loc[line.bus1, ["x", "y"]]
    color = carrier_colors.get(line.carrier, "gray")  # fallback if undefined
    ax.plot([x0, x1], [y0, y1], color=color, transform=crs, zorder=3,
            label="Lines (AC)" if first_line else None)
    first_line = False

# Plot links (DC)
first_link = True
for _, link in n.links.iterrows():
    x0, y0 = n.buses.loc[link.bus0, ["x", "y"]]
    x1, y1 = n.buses.loc[link.bus1, ["x", "y"]]
    color = carrier_colors.get(link.carrier, "blue")
    ax.plot([x0, x1], [y0, y1], color=color, linestyle="--", transform=crs, zorder=3,
            label="Links (DC)" if first_link else None)
    first_link = False

plt.title("PyPSA High-level NEM Network (Buses, Lines, Links)")
plt.legend()
plt.tight_layout()
plt.show()

# Add contraints for LOR based on generator availability
from functools import partial

def _add_LOR_constraints(n, snapshots, threshold):
    print(f"Adding LOR constraints for {len(snapshots)} snapshots...")
    m = n.model
    
    # Get all active generators
    active_gens = n.generators.index[n.generators.p_nom > 0]
    
    # Create availability matrix - use 1.0 for missing generators
    avail = pd.DataFrame(
        index=snapshots,
        columns=active_gens,
        data=1.0  # Default to always available
    )
    
    # Fill in actual availability data where it exists
    for gen in active_gens:
        if gen in n.generators_t.p_max_pu.columns:
            avail[gen] = n.generators_t.p_max_pu[gen]
    
    # Multiply by p_nom to get actual availability
    avail = avail.multiply(n.generators.loc[active_gens, 'p_nom'], axis=1)
    
    if isinstance(threshold, dict):
        th = threshold
    else:
        th = {bus: threshold for bus in n.buses.index}

    constraint_count = 0
    
    for bus in n.buses.index:
        # Consider all active generators at this bus
        gens = n.generators.index[
            (n.generators.bus == bus) & (n.generators.p_nom > 0)
        ]
        if not len(gens):
            continue
            
        for t in snapshots:
            # Sum available capacity
            available_capacity = avail.loc[t, gens].sum()
            
            # Calculate RHS
            rhs = available_capacity - th.get(bus, 0.0)
            
            # Calculate LHS
            lhs = sum(m["Generator-p"].loc[t, g] for g in gens)
            
            # Add constraint
            m.add_constraints(
                lhs <= rhs, 
                name=f"LOR_{bus}_{t}"
            )
            constraint_count += 1
            
            if constraint_count % 100 == 0:
                print(f"Added {constraint_count} constraints...")
    
    print(f"Total LOR constraints added: {constraint_count}")

# Your hook - only specify threshold
lor_hook = partial(
    _add_LOR_constraints,
    threshold = {
        "NSW1":705,
        "VIC1":550,
        "QLD1":710,
        "SA1":195,
        "TAS1":140
    }
)

# Check which generators are missing from p_max_pu
active_gens = n.generators.index[n.generators.p_nom > 0]
missing_gens = set(active_gens) - set(n.generators_t.p_max_pu.columns)

print(f"Active generators: {len(active_gens)}")
print(f"Generators with p_max_pu data: {len(n.generators_t.p_max_pu.columns)}")
print(f"Missing generators: {len(missing_gens)}")
print(f"Missing generators list: {list(missing_gens)}")

# Check what types of generators are missing
if missing_gens:
    missing_types = n.generators.loc[list(missing_gens), 'carrier'].value_counts()
    print(f"Missing generator types: {missing_types}")

# Test with limited snapshots
n.optimize(
    snapshots=n.snapshots[:24],  # This is what limits the snapshots!
    solver_name="gurobi",
    extra_functionality=lor_hook,
    solver_options={
        "TimeLimit": 600,
        "MIPGap": 0.05,
    }
)

n.optimize(solver_name = "gurobi")

# Export the baseline network to NetCDF format
n.export_to_netcdf("results/high-level_nem.nc")
n.export_to_netcdf("results/scenarios/lor_constraint_test.nc")
n = pypsa.Network("results/high-level_nem.nc")
n = pypsa.Network("results/scenarios/lor_constraint_test.nc")

# After optimization, check if reserves are maintained
def check_reserve_margins(n, threshold):
    """Check if reserve margins are being maintained"""
    
        
    # Get availability data
    active_gens = n.generators.index[n.generators.p_nom > 0]
    available_gens = n.generators_t.p_max_pu.columns.intersection(active_gens)
    
    avail = pd.DataFrame(
        index=n.snapshots,
        columns=active_gens,
        data=1.0
    )
    
    for gen in active_gens:
        if gen in n.generators_t.p_max_pu.columns:
            avail[gen] = n.generators_t.p_max_pu[gen]
    
    avail = avail.multiply(n.generators.loc[active_gens, 'p_nom'], axis=1)
    
    results = []
    
    for bus in threshold.keys():
        bus_gens = n.generators.index[
            (n.generators.bus == bus) & (n.generators.p_nom > 0)
        ]
        if len(bus_gens) == 0:
            continue
            
        # Get dispatch and availability for this bus
        dispatch = n.generators_t.p[bus_gens].sum(axis=1)
        available = avail[bus_gens].sum(axis=1)
        
        # Calculate actual reserves
        actual_reserves = available - dispatch
        required_reserves = threshold[bus]
        
        # Check if reserves are maintained
        violations = (actual_reserves < required_reserves).sum()
        min_reserve = actual_reserves.min()
        avg_reserve = actual_reserves.mean()
        
        results.append({
            'bus': bus,
            'required_reserve': required_reserves,
            'min_actual_reserve': min_reserve,
            'avg_actual_reserve': avg_reserve,
            'violations': violations,
            'violation_rate': violations / len(actual_reserves) * 100
        })
    
    return pd.DataFrame(results)

# Use it after optimization
reserve_analysis = check_reserve_margins(n, {
    "NSW1":705,
    "VIC1":550,
    "QLD1":710,
    "SA1":195,
    "TAS1":140
})

reserve_analysis



stats_df = n.statistics()

# Detect any unserved energy
# Step 1: Get names of generators with carrier == "Unserved Energy"
unserved_gen_names = n.generators[n.generators.carrier == "Unserved Energy"].index

# Step 2: Use those names to extract columns from generators_t.p
if not unserved_gen_names.empty:
    unserved = n.generators_t.p[unserved_gen_names]
    total_unserved_energy = unserved.sum().sum()  # Sum over time and generators
    print(f"Total unserved energy: {total_unserved_energy:.2f} MWh")
else:
    print("No 'Unserved Energy' generators found.")

# Review any unserved energy by bus (region)
n.generators[n.generators.carrier == "Unserved Energy"] \
    .groupby(["bus", "carrier"]).p_nom_opt.sum().reset_index() \
    .pivot(index='bus', columns='carrier', values='p_nom_opt') \
    .fillna(0).astype(int).sort_index()

# Group by bus and carrier to sum the optimised nominal power
n.generators.groupby(["bus", "carrier"]).p_nom_opt.sum().reset_index() \
    .pivot(index='bus', columns='carrier', values='p_nom_opt') \
    .fillna(0).astype(int).sort_index()
    
###--- HELPER FUNCTIONS ---###

# RESET NETWORK function
def initialize_network_with_unserved_energy(network_path):
    """
    Initialize a PyPSA network and add unserved energy generators with carrier colors.
    
    Parameters:
    network_path (str): Path to the network file (.nc)
    
    Returns:
    pypsa.Network: Initialized network with unserved energy generators and carrier colors
    """
        
    # Initialize network
    n = pypsa.Network()
    n = pypsa.Network(network_path)
    
    # Add unserved energy generators
    for bus in n.loads.bus.unique():
        gen_name = f"{bus}-UNSERVED"
        n.add("Generator",
              name=gen_name,
              bus=bus,
              carrier="Unserved Energy",
              p_nom_extendable=True,
              p_nom=0,
              marginal_cost=10000,
              capital_cost=0)
    
    # Define carrier colors
    carrier_colors = {
        "Biomass": '#127E2A',
        "Hydro": '#1E81D4',
        "Black Coal": "#39322D",
        "Solar": '#FDB324',
        "Rooftop Solar": '#FFE066',
        "Wind": '#3BBFE5',
        "Diesel": "#D486BA",
        "Brown Coal": "#715E50",
        "Gas": '#E6622D',
        "ROR": '#8ab2d4',
        "Battery": '#814ad4',
        "Pump Hydro": '#104775',
        "AC": '#999999',
        "DC": "#3277AF",
        "Unserved Energy": "#F40B16"
        }
    
    # Assign colors to carriers
    for carrier, color in carrier_colors.items():
        n.carriers.loc[carrier, 'color'] = color
    
    return n

n = initialize_network_with_unserved_energy("results/high-level_nem.nc")

#--- END UTILITY FUNCTIONS ---###

def plot_network(n, show_buses=True, show_loading=True, show_linecap=True):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from pypsa.plot import add_legend_patches

    # 1) Filter out any generators with zero optimized capacity
    gen_nonzero = n.generators[n.generators.p_nom_opt > 0]

    # 2) Calculate bus generation capacities (only non-zero gens)
    capacities = gen_nonzero.groupby(["bus", "carrier"]).p_nom_opt.sum()

    # 3) Calculate line loading
    n.lines['loading'] = n.lines_t.p0.abs().max() / n.lines.s_nom

    # 4) Set up plot
    fig = plt.figure(figsize=(15, 9))
    ax  = plt.axes(projection=ccrs.Mercator())

    # 5) Draw network, using capacities for bus sizes
    n.plot(
        ax=ax,
        line_colors=(n.lines['loading'] if show_loading else None),
        line_cmap=plt.cm.viridis,
        line_widths=(n.lines.s_nom / n.lines.s_nom.max() * 4 if show_linecap else 1),
        bus_sizes=(capacities / (capacities.max() * 4) if show_buses else 1),
        boundaries=(110, 155, -45, -10)
    )

    # 6) Colorbar for line loading
    if show_loading:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=n.lines['loading'].min(), vmax=n.lines['loading'].max())
        )
        sm._A = []
        plt.colorbar(sm, ax=ax, orientation='vertical', label='Line Loading')

    # 7) Legend for generators (only carriers with non-zero gens)
    if show_buses:
        unique_carriers = gen_nonzero.carrier.unique()
        carrier_colors = [n.carriers.loc[carrier, 'color']
                          for carrier in unique_carriers if carrier in n.carriers.index]

        add_legend_patches(
            ax,
            carrier_colors,
            unique_carriers,
            legend_kw=dict(frameon=False, bbox_to_anchor=(0, 1))
        )

    plt.title('Network Line Loading and Generator Capacities (filtered zero-output)')
    plt.show()

plot_network(n,show_buses=True,show_loading=True,show_linecap=True)


def increase_demand_by_percentage(network, percentage):
    network.loads_t.p_set *= (1 + percentage / 100)


# Plotting function for generation dispatch
# Interactive version (optional)


def plot_dispatch(n, time="2024", days=None, regions=None,
                   show_imports=True, show_curtailment=True,
                   scenario_name=None, scenario_objective=None, interactive=False):
     """
     Plot a generation dispatch stack by carrier for a PyPSA network, with optional
     net imports/exports and a region‑filtered curtailment overlay.

     Parameters
     ----------
     n : pypsa.Network
         The PyPSA network to plot.
     time : str, default "2024"
         Start of the time window (e.g. "2024", "2024-07", or "2024-07-15").
     days : int, optional
         Number of days from `time` to include in the plot.
     regions : list of str, optional
         Region bus names to filter by. If None, the entire network is included.
     show_imports : bool, default True
         Whether to include net imports/exports in the dispatch stack.
     show_curtailment : bool, default True
         Whether to calculate and plot VRE curtailment (Solar, Wind, Rooftop Solar).
     scenario_name : str, optional
         Scenario label to display below the title.
     scenario_objective : str, optional
         Objective description to display next to the legend.
     interactive : bool, default False
         Whether to create an interactive plot using Plotly instead of matplotlib.

     Notes
     -----
     - All power values are converted to GW.
     - Curtailment is plotted as a dashed black line if enabled.
     - Demand (load) is plotted as a solid green line.
     - Storage charging and net exports (negative values) are shown below zero.
     """
     
     
     if interactive:
         import plotly.graph_objects as go
         from plotly.subplots import make_subplots
         import plotly.express as px
     else:
         import matplotlib.pyplot as plt
     
     # 1) REGION MASKS
     if regions is not None:
         gen_mask   = n.generators.bus.isin(regions)
         sto_mask   = n.storage_units.bus.isin(regions) if not n.storage_units.empty else []
         store_mask = n.stores.bus.isin(regions) if not n.stores.empty else []
         region_buses = set(regions)
         
     else:
        gen_mask = pd.Series(True, index=n.generators.index)
        sto_mask = pd.Series(True, index=n.storage_units.index) if not n.storage_units.empty else pd.Series(dtype=bool)
        store_mask = pd.Series(True, index=n.stores.index) if not n.stores.empty else pd.Series(dtype=bool)
        region_buses = set(n.buses.index)

     # 2) AGGREGATE BY CARRIER (GW)
     def _agg(df_t, df_stat, mask):
         return (
             df_t.loc[:, mask]
                 .T
                 .groupby(df_stat.loc[mask, 'carrier'])
                 .sum()
                 .T
                 .div(1e3)
         )

     p_by_carrier = _agg(n.generators_t.p, n.generators, gen_mask)
     if not n.storage_units.empty:
         p_by_carrier = pd.concat([p_by_carrier,
                                   _agg(n.storage_units_t.p, n.storage_units, sto_mask)],
                                  axis=1)
     if not n.stores.empty:
         p_by_carrier = pd.concat([p_by_carrier,
                                   _agg(n.stores_t.p, n.stores, store_mask)],
                                  axis=1)

     # 3) TIME WINDOW
     parts = time.split("-")
     if len(parts) == 1:
         start = pd.to_datetime(f"{parts[0]}-01-01")
     elif len(parts) == 2:
         start = pd.to_datetime(f"{parts[0]}-{parts[1]}-01")
     else:
         start = pd.to_datetime(time)

     if days is not None:
         end = start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
     elif len(parts) == 1:
         end = pd.to_datetime(f"{parts[0]}-12-31 23:00")
     elif len(parts) == 2:
         end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
     else:
         end = start + pd.Timedelta(hours=23)

     p_slice = p_by_carrier.loc[start:end].copy()
     # drop carriers with zero activity
     zero = p_slice.columns[p_slice.abs().sum() == 0]
     p_slice.drop(columns=zero, inplace=True)

     # 4) IMPORTS/EXPORTS
     if show_imports:
         ac = ( n.lines_t.p0.loc[start:end, n.lines.bus1.isin(region_buses) & ~n.lines.bus0.isin(region_buses)].sum(axis=1)
              + n.lines_t.p1.loc[start:end, n.lines.bus0.isin(region_buses) & ~n.lines.bus1.isin(region_buses)].sum(axis=1) )
         dc = ( n.links_t.p0.loc[start:end, n.links.bus1.isin(region_buses) & ~n.links.bus0.isin(region_buses)].sum(axis=1)
              + n.links_t.p1.loc[start:end, n.links.bus0.isin(region_buses) & ~n.links.bus1.isin(region_buses)].sum(axis=1) )
         p_slice['Imports/Exports'] = (ac + dc).div(1e3)
         if 'Imports/Exports' not in n.carriers.index:
             n.carriers.loc['Imports/Exports','color']='#7f7f7f'

     # 5) LOAD SERIES
     if regions:
         load_cols = [c for c in n.loads[n.loads.bus.isin(regions)].index if c in n.loads_t.p_set]
         load_series = n.loads_t.p_set[load_cols].sum(axis=1)
     else:
         load_series = n.loads_t.p_set.sum(axis=1)
     load_series = load_series.loc[start:end].div(1e3)

     # 6) VRE CURTAILMENT (GW) if requested
     if show_curtailment:
         vre = ['Solar','Wind', 'Rooftop Solar']
         mask_vre = gen_mask & n.generators.carrier.isin(vre)
         avail = (n.generators_t.p_max_pu.loc[start:end, mask_vre]
                  .multiply(n.generators.loc[mask_vre,'p_nom'], axis=1))
         disp  = n.generators_t.p.loc[start:end, mask_vre]
         curtail = (avail.sub(disp, fill_value=0)
                        .clip(lower=0)
                        .sum(axis=1)
                        .div(1e3))
     else:
         curtail = None

     # 7) PLOT
     title_tail = f" for {', '.join(regions)}" if regions else ''
     plot_title = f"Dispatch by Carrier: {start.date()} to {end.date()}{title_tail}"
     
     if interactive:
         # PLOTLY INTERACTIVE PLOT
         fig = go.Figure()
         
         fig.update_layout(
            plot_bgcolor='#F0FFFF',
            xaxis=dict(gridcolor='#DDDDDD'),
            yaxis=dict(gridcolor='#DDDDDD')        # Plot area background
         ) 

         # Prepare data for stacked area plot
         positive_data = p_slice.where(p_slice > 0).fillna(0)
         negative_data = p_slice.where(p_slice < 0).fillna(0)
         
         # Add positive generation as stacked area
         for i, col in enumerate(positive_data.columns):
             if positive_data[col].sum() > 0:
                 color = n.carriers.loc[col, 'color']
                 # Only add points where value > 0.001
                 mask = positive_data[col].abs() > 0.001
                 if mask.any():
                     fig.add_trace(go.Scatter(
                         x=positive_data.index[mask],
                         y=positive_data[col][mask],
                         mode='lines',
                         fill='tonexty' if i > 0 else 'tozeroy',
                         line=dict(width=0, color=color),
                         fillcolor=color,
                         name=col,
                         stackgroup='positive',
                         hovertemplate='<b>%{fullData.name}</b><br>Power: %{y:.3f} GW<extra></extra>',
                         showlegend=True
                     ))
         
         # Add negative generation (storage charging, exports)
         for col in negative_data.columns:
             if negative_data[col].sum() < 0:
                 color = n.carriers.loc[col, 'color']
                 # Only add points where value < -0.001
                 mask = negative_data[col].abs() > 0.001
                 if mask.any():
                     fig.add_trace(go.Scatter(
                         x=negative_data.index[mask],
                         y=negative_data[col][mask],
                         mode='lines',
                         fill='tonexty',
                         line=dict(width=0, color=color),
                         fillcolor=color,
                         name=col,
                         stackgroup='negative',
                         hovertemplate='<b>%{fullData.name}</b><br>Power: %{y:.2f} GW<extra></extra>',
                         showlegend=True
                     ))
         
         # Add demand line (always show)
         fig.add_trace(go.Scatter(
             x=load_series.index,
             y=load_series,
             mode='lines',
             line=dict(color='green', width=2),
             name='Demand',
             hovertemplate='<b>Demand</b><br>Power: %{y:.2f} GW<extra></extra>',
             showlegend=True
         ))
         
         # Add curtailment line if requested
         if show_curtailment and curtail is not None:
             fig.add_trace(go.Scatter(
                 x=curtail.index,
                 y=curtail,
                 mode='lines',
                 line=dict(color='black', width=2, dash='dash'),
                 name='Curtailment',
                 hovertemplate='<b>Curtailment</b><br>Power: %{y:.2f} GW<extra></extra>',
                 showlegend=True
                 ))
         
         # Update layout
         fig.update_layout(
             title=plot_title,
             xaxis_title='Time',
             yaxis_title='Power (GW)',
             hovermode='x unified',
             hoverlabel=dict(
                 bgcolor="white",
                 bordercolor="black",
                 font_size=12,
             ),
             legend=dict(
                 x=1.02,
                 y=1,
                 bgcolor='rgba(255,255,255,0.8)',
                 bordercolor='rgba(0,0,0,0.2)',
                 borderwidth=1
             ),
             width=800,
             height=500
         )
         
         # Add scenario annotations
         annotations = []
         if scenario_name:
             annotations.append(
                 dict(
                     x=1.02, y=-0.05,
                     xref='paper', yref='paper',
                     text=f"Scenario: {scenario_name}",
                     showarrow=False,
                     font=dict(size=10, color='gray'),
                     xanchor='center',
                     yanchor='top'
                 )
             )
         
         if annotations:
             fig.update_layout(annotations=annotations)
         
         fig.show()
         
     else:
         # MATPLOTLIB STATIC PLOT 
         fig, ax = plt.subplots(figsize=(8.4, 6.5)) #12,6.5
         cols = p_slice.columns.map(lambda c: n.carriers.loc[c,'color'])
         p_slice.where(p_slice>0).plot.area(ax=ax,linewidth=0,color=cols)
         neg = p_slice.where(p_slice<0).dropna(how='all',axis=1)
         if not neg.empty:
             neg_cols=[n.carriers.loc[c,'color'] for c in neg.columns]
             neg.plot.area(ax=ax,linewidth=0,color=neg_cols)
         load_series.plot(ax=ax,color='g',linewidth=1.5,label='Demand')
         if show_curtailment and curtail is not None:
             curtail.plot(ax=ax,color='k',linestyle='--',linewidth=1.2,label='Curtailment')

         # limits & legend
         up = max(p_slice.where(p_slice>0).sum(axis=1).max(),
                  load_series.max(),
                  curtail.max() if curtail is not None else 0)
         dn = min(p_slice.where(p_slice<0).sum(axis=1).min(), load_series.min())
         ax.set_ylim(dn if not np.isclose(up,dn) else dn-0.1, up)
        #  fig.patch.set_facecolor('#F0FFFF') 
         ax.set_facecolor('#F0FFFF')
         h,l = ax.get_legend_handles_labels()
         seen={} ; fh,fl=[],[]
         for hh,ll in zip(h,l):
             if ll not in seen: fh.append(hh);fl.append(ll);seen[ll]=True
         ax.legend(fh,fl,loc=(1.02,0.67), fontsize=9)

         # scenario text
         if scenario_objective:
             ax.text(1.02,0.01,f"Objective:\n{scenario_objective}",transform=ax.transAxes,
                     fontsize=8,va='bottom',ha='left',bbox=dict(facecolor='white',alpha=0.7,edgecolor='none'))
         if scenario_name:
             ax.text(1.02,-0.05,f"Scenario: {scenario_name}",transform=ax.transAxes,
                     fontsize=9,color='gray',ha='center',va='top')

         ax.set_ylabel('GW')
         ax.set_title(plot_title)
         plt.tight_layout()
         plt.show()

    
    
# plot_dispatch function examples
plot_dispatch(n, time="2024-07-01", days=3, regions=["NSW1"], show_imports=True)
plot_dispatch(n, time="2024-07-01", days=3, show_imports=False)
plot_dispatch(n, time="2024-05-24", days=6, regions=["SA1"], show_imports=False)
plot_dispatch(n, time="2024-05-24", days=6, regions=["SA1"], show_imports=True)



# TODO: confirm if there is time series limits available for lines (they change over time)


# Inconnector Utilisation over time
# Define your interconnectors (can include both links and lines)
interconnectors = [
    'QNI_NSW_to_QLD',
    'QNI_QLD_to_NSW',
    'VIC_to_NSW',
    'NSW_to_VIC',
    'V_SA_VIC_to_SA',
    'V_SA_SA_to_VIC',
    'BASSLINK_TAS_TO_VIC',
    'BASSLINK_VIC_TO_TAS',
    'MURRAYLINK_VIC_TO_SA',
    'MURRAYLINK_SA_TO_VIC',
    'PEC_SA1_to_NSW',
    'MARINUS_S1_TAS_TO_VIC',
    'MARINUS_S1_VIC_TO_TAS',
    'VNI_WEST_VIC_to_NSW',
    'VNI_WEST_NSW_to_VIC'
    ]            

# Calculate utilisation
line_util = n.lines_t.p0.abs().div(n.lines.s_nom, axis=1) * 100  # %
link_util = n.links_t.p0.abs().div(n.links.p_nom, axis=1) * 100  # %

# Set up plot
num = len(interconnectors)
fig, axs = plt.subplots(num, 1, figsize=(12, 2.5 * num), sharex=True)

for i, name in enumerate(interconnectors):
    ax = axs[i] if num > 1 else axs  # handle 1-subplot edge case
    if name in line_util.columns:
        line_util[name].plot(ax=ax, label=name, color='tab:blue')
    elif name in link_util.columns:
        link_util[name].plot(ax=ax, label=name, color='tab:orange', linestyle='--')
    else:
        ax.set_visible(False)
        continue

    ax.set_ylabel("% Utilisation")
    ax.set_title(f"Utilisation of {name}")
    ax.axhline(100, linestyle="--", color="red", alpha=0.5)
    ax.grid(True)
    # ax.legend()

plt.xlabel("Time")
plt.tight_layout()
plt.show()


# Mean utilisation of interconnectors
# Mean utilisation (mean over time)
mean_line_util = line_util.mean()
mean_link_util = link_util.mean()

# Combine
util_df = pd.concat([mean_line_util, mean_link_util]).sort_values(ascending=False)

# Plot bar chart
fig, ax = plt.subplots(figsize=(10, 5))
util_df.plot(kind="bar", ax=ax, color="steelblue")
ax.axhline(100, linestyle="--", color="red", label="Nominal Limit")
ax.set_ylabel("Mean Utilisation (%)")
ax.set_title("Mean Utilisation of Interconnectors")
ax.grid(True, axis='y')
ax.legend()
plt.tight_layout()
plt.show()



def plot_generator_capacity_by_carrier(network):
    """
    Plot total generator capacity by carrier for a given PyPSA network,
    excluding carriers with zero total capacity, and colour bars by carrier colour.
    """
    # 1) sum and filter
    capacity_by_carrier = (
        network.generators
               .groupby("carrier")
               .p_nom_opt
               .sum()
               .div(1e3)
    )
    capacity_by_carrier = capacity_by_carrier[capacity_by_carrier > 0]

    # 2) get colors
    colors = [
        network.carriers.loc[c, 'color']
            if c in network.carriers.index and 'color' in network.carriers.columns
            else 'gray'
        for c in capacity_by_carrier.index
    ]

    # 3) create fig & ax, set backgrounds
    fig, ax = plt.subplots(figsize=(8, 5))
    # fig.patch.set_facecolor('#F0FFFF')    # full-figure bg
               # axes bg

    # 4) plot onto that ax
    capacity_by_carrier.plot.barh(ax=ax, color=colors)

    # 5) labels & layout
    ax.set_facecolor('#F0FFFF')
    ax.set_xlabel("GW")
    ax.set_title("Total Generator Capacity by Carrier")
    fig.tight_layout()
    plt.show()

plot_generator_capacity_by_carrier(n)

def plot_storage_dispatch(network, include_stores=True, carrier=None):
    """
    Plot storage dispatch and state of charge for a selected storage carrier.
    
    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    include_stores : bool, default True
        Whether to include Store components (e.g. hydrogen tanks).
    carrier : str or None, default None
        Specific carrier to filter by (e.g., "Battery"). If None, all carriers are included.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---------- StorageUnits ----------
    if carrier is not None:
        mask_su = network.storage_units.carrier == carrier
        storage_names = network.storage_units.index[mask_su]
        label_prefix = f"{carrier} "
    else:
        storage_names = network.storage_units.index
        label_prefix = ""

    p_storage = network.storage_units_t.p[storage_names].sum(axis=1).div(1e3)
    state_of_charge = (
        network.storage_units_t.state_of_charge[storage_names].sum(axis=1).div(1e3)
        if hasattr(network.storage_units_t, "state_of_charge") else None
    )

    # ---------- Stores (optional) ----------
    if include_stores and not network.stores.empty:
        if carrier is not None:
            mask_store = network.stores.carrier == carrier
            store_names = network.stores.index[mask_store]
        else:
            store_names = network.stores.index

        p_store = network.stores_t.p[store_names].sum(axis=1).div(1e3)
        soc_store = (
            network.stores_t.e[store_names].sum(axis=1).div(1e3)
            if hasattr(network.stores_t, "e") else None
        )

        p_storage = p_storage.add(p_store, fill_value=0)
        if state_of_charge is not None and soc_store is not None:
            state_of_charge = state_of_charge.add(soc_store, fill_value=0)
        elif soc_store is not None:
            state_of_charge = soc_store

    # ---------- Plotting ----------
    if p_storage.empty and (state_of_charge is None or state_of_charge.empty):
        print("No storage dispatch or state of charge data to plot.")
        return

    p_storage.plot(label=f"{label_prefix}Dispatch [GW]", ax=ax)
    
    if state_of_charge is not None and not state_of_charge.empty:
        state_of_charge.plot(label=f"{label_prefix}State of Charge [GWh]", ax=ax, linestyle='--')

    ax.set_ylabel("Power [GW] / Energy [GWh]")
    ax.set_title(f"Storage Dispatch and State of Charge{f' for {carrier}' if carrier else ''}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# All storage (batteries, PHES, etc.)
plot_storage_dispatch(n)




def plot_peak_demand_vs_capacity(network, stack_carrier=False, relative=False, sort_by="Max Demand"):
    """
    Plot peak demand vs installed generation capacity per bus, filtering out zero-capacity carriers.

    Parameters:
    -----------
    network : pypsa.Network
        The PyPSA network object.
    stack_carrier : bool, default=False
        Whether to stack capacity by carrier or plot total capacity.
    relative : bool, default=False
        If True, plot capacity as a percentage of peak demand.
    sort_by : str, optional
        Column to sort buses by on the x-axis. Options: "Max Demand", "Total Capacity", or None.
    """
    # --- 1) Max demand per bus ---
    max_demand = network.loads_t.p_set.max()

    # --- 2) Capacity per bus and carrier ---
    capacity_by_bus_carrier = (
        network.generators.groupby(['bus', 'carrier'])
        .p_nom_opt.sum()
        .unstack(fill_value=0)
    )
    # Filter out carriers with zero total capacity
    nonzero_carriers = capacity_by_bus_carrier.columns[capacity_by_bus_carrier.sum(axis=0) > 0]
    capacity_by_bus_carrier = capacity_by_bus_carrier[nonzero_carriers]
    total_capacity = capacity_by_bus_carrier.sum(axis=1)

    # --- 3) Combine DataFrame and filter out zero-demand & zero-capacity buses ---
    df = pd.concat([max_demand.rename("Max Demand"),
                    total_capacity.rename("Total Capacity")], axis=1).fillna(0)
    df = df[(df["Max Demand"] > 0) | (df["Total Capacity"] > 0)]

    # --- 4) Relative scaling if requested ---
    if relative:
        # avoid div-by-zero
        df["Max Demand"] = df["Max Demand"].replace(0, np.nan)
        relative_capacity = capacity_by_bus_carrier.div(df["Max Demand"], axis=0) * 100
        df["Total Capacity"] = df["Total Capacity"] / df["Max Demand"] * 100
        df["Max Demand"] = 100
        ylabel = "Capacity as % of Peak Demand"
    else:
        # convert to GW
        df[["Max Demand", "Total Capacity"]] = df[["Max Demand", "Total Capacity"]] / 1e3
        relative_capacity = capacity_by_bus_carrier / 1e3
        ylabel = "GW"

    # --- 5) Sort if needed ---
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)

    # --- 6) Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    bar_pos = np.arange(len(df))

    if stack_carrier:
        # stack each non-zero carrier
        bottom = np.zeros(len(df))
        for carrier in capacity_by_bus_carrier.columns:
            vals = relative_capacity[carrier].reindex(df.index).fillna(0).values
            color = (network.carriers.loc[carrier, 'color']
                     if 'color' in network.carriers.columns and carrier in network.carriers.index
                     else None)
            ax.bar(bar_pos + bar_width/2, vals, bar_width,
                   bottom=bottom, label=carrier, color=color)
            bottom += vals
        # plot peak demand on left
        ax.bar(bar_pos - bar_width/2, df["Max Demand"], bar_width,
               label='Peak Demand', color='gray', alpha=0.7)
    else:
        ax.bar(bar_pos - bar_width/2, df["Max Demand"], bar_width,
               label='Peak Demand', color='gray', alpha=0.7)
        ax.bar(bar_pos + bar_width/2, df["Total Capacity"], bar_width,
               label='Total Capacity', color='tab:blue')

    # --- 7) Labels and legend ---
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Bus")
    ax.set_title("Peak Demand vs Generation Capacity per Bus" + (" (Relative)" if relative else ""))
    ax.grid(True)
    # place legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.show()


plot_peak_demand_vs_capacity(n, stack_carrier=True)
plot_peak_demand_vs_capacity(n, stack_carrier=True, relative=True)

# Generators that the optimisation is allowed to resize
extendable_gens = n.generators[n.generators.p_nom_extendable]

if not extendable_gens.empty:
    print("Extendable generators:\n")
    print(extendable_gens[["bus", "carrier", "p_nom", "p_nom_opt"]])
else:
    print("No generators have p_nom_extendable = True.")


def plot_total_demand_vs_generation(network, stack_carrier=False, relative=False):
    """
    Plot total electricity demand vs generation per bus.

    Parameters:
    - network: PyPSA Network object
    - stack_carrier: If True, stack generation by carrier (color-coded)
    - relative: If True, show generation as % of total demand (demand = 100%)
    """
    # Total demand per bus in GWh
    total_demand_per_bus = network.loads_t.p_set.sum().div(1e3)
    total_demand_per_bus.name = "Total Demand"

    # Total generation per generator in GWh
    gen_energy = network.generators_t.p.sum().div(1e3)
    gen_info = network.generators[["bus", "carrier"]]
    gen_energy_by_carrier = (
        gen_info.assign(energy=gen_energy)
        .groupby(["bus", "carrier"])["energy"]
        .sum()
        .unstack(fill_value=0)
    )
    total_generation_per_bus = gen_energy_by_carrier.sum(axis=1)
    total_generation_per_bus.name = "Total Generation"

    # Join and filter
    generation_vs_demand = pd.concat([total_demand_per_bus, total_generation_per_bus], axis=1).fillna(0)
    generation_vs_demand = generation_vs_demand.loc[
        (generation_vs_demand["Total Demand"] > 0) | (generation_vs_demand["Total Generation"] > 0)
    ]

    if relative:
        generation_vs_demand["Total Demand"].replace(0, np.nan, inplace=True)  # avoid div by 0
        relative_generation = gen_energy_by_carrier.div(generation_vs_demand["Total Demand"], axis=0) * 100
        generation_vs_demand["Total Generation"] = (
            generation_vs_demand["Total Generation"] / generation_vs_demand["Total Demand"] * 100
        )
        generation_vs_demand["Total Demand"] = 100
        ylabel = "Generation vs Demand (%)"
    else:
        relative_generation = gen_energy_by_carrier
        ylabel = "GWh"

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    bar_positions = np.arange(len(generation_vs_demand))

    if stack_carrier:
        bottom = np.zeros(len(generation_vs_demand))
        carriers = [
        c for c in relative_generation.columns
        if (c in network.carriers.index) and (relative_generation.loc[generation_vs_demand.index, c].sum() > 0)
        ]
        for carrier in carriers:
            values = relative_generation.get(carrier, pd.Series(0, index=generation_vs_demand.index))
            color = network.carriers.loc[carrier, 'color'] if 'color' in network.carriers.columns else None
            ax.bar(
                bar_positions + bar_width / 2, values, bar_width,
                label=f'Generation ({carrier})', bottom=bottom, color=color
            )
            bottom += values.values

        ax.bar(
            bar_positions - bar_width / 2,
            generation_vs_demand["Total Demand"],
            bar_width,
            label="Total Demand",
            color="gray", alpha=0.7
        )
    else:
        ax.bar(
            bar_positions - bar_width / 2,
            generation_vs_demand["Total Demand"],
            bar_width,
            label="Total Demand",
            color="gray", alpha=0.7
        )
        ax.bar(
            bar_positions + bar_width / 2,
            generation_vs_demand["Total Generation"],
            bar_width,
            label="Total Generation",
            color="tab:blue"
        )

    ax.set_xlabel("Bus")
    ax.set_ylabel(ylabel)
    ax.set_title("Total Demand vs Total Generation per Bus" + (" (Relative)" if relative else ""))
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(generation_vs_demand.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()



plot_total_demand_vs_generation(n, stack_carrier=True, relative=False)


n.statistics.curtailment().reset_index()
n.storage_units
n.generators
n.carriers

# Solving Troubleshooting and diagnostics
gen = "NSW1-HYDRO"
df = pd.DataFrame({
    "min": n.generators.loc[gen, "p_min_pu"] * n.generators.loc[gen, "p_nom"],
    "max": n.generators_t.p_max_pu[gen] * n.generators.loc[gen, "p_nom"]
}, index=n.snapshots)

problem_times = df[df["max"] < df["min"]]
print(problem_times)

n.generators_t.p.sum().sort_values(ascending=False)
(n.generators_t.p.sum(axis=1) - n.loads_t.p_set.sum(axis=1)).plot()
n.links_t.p0.plot(title="HVDC Interconnector Flows")
n.buses_t.marginal_price.plot(title="Regional Prices (LMPs)")

# Export dispatch results to CSV
n.generators_t.p.to_csv("results/dispatch.csv")
n.buses_t.marginal_price.to_csv("results/prices.csv")

n.generators_t.p.head(25)
n.loads_t.p_set.sum(axis=1).plot(title="Total system load over time")

n.links_t.p0[["BASSLINK_TAS_TO_VIC", "MURRAYLINK_VIC_TO_SA"]].plot()
n.buses_t.marginal_price.head()

n.lines_t.p0
n.lines_t.p1
n.links_t.p0
n.lines_t.p1

total_cost = (n.generators_t.p * n.generators["marginal_cost"]).sum().sum()
print(f"Total system generation cost: ${total_cost:,.0f} AUD")



# Script to generate a simplified sensitivity analysis template
# that can be plugged into a PyPSA workflow.

#--- GENERATE SCENARIOS ---####

# Intergrate csv input assumptions for tech - this better documents scenarios
def dict_to_multiline_string(d):
    return "\n".join(f"{k}: {v}" for k, v in d.items()) if d else ""

def generate_scenarios(
        network,
        tech_assumptions_path,
        export_dir="results/scenarios"
    ):
    """
    Apply scaling from a technology-assumptions CSV, solve each scenario, export
    the solved network, and return a summary DataFrame.

    Extra metrics collected:
    • **Unserved Energy (GWh)** - total unserved energy across the system.
    • **Generator Capacity (GW)** - total generator capacity by carrier.
      (built-in `n.generators.p_nom`)

    • **Gas Generation (GWh)** - total dispatched energy from carrier 'Gas'.
    • **Wind and Solar Curtailment (GWh)** - renewable curtailment across the whole
      system (built-in `n.statistics.curtailment`).
    • **Battery Capacity (GW)** - total battery storage capacity in the system.

    Parameters
    ----------
    network : pypsa.Network
        The base PyPSA network object to scale up/down.
    tech_assumptions_path : str
        Path to CSV file with scaling assumptions for generators/storage.
    export_dir : str, default "results/scenarios"
        Directory path to export .nc and CSV results.
    

    Returns
    -------
    pandas.DataFrame
        Scenario metrics including battery capacity (GW), generator capacity (GW), curtailment (GWh), unserved energy (GWh) and gas generation (GWh).
    """

    
    os.makedirs(export_dir, exist_ok=True)
    results = []

    tech_df = pd.read_csv(tech_assumptions_path)
    scenario_names = tech_df["scenario"].unique()
    timer_all = perf_counter()

    for scenario in scenario_names:
        print(f"\u2192 Scenario: {scenario}")

        # ── copy network ───────────────────────────────────────────────
        n = network.copy(snapshots=network.snapshots)

        # ── apply scaling from CSV ────────────────────────────────────
        df_s = tech_df[tech_df["scenario"] == scenario]
        for _, row in df_s.iterrows():
            component, bus, carrier, scale = (
                row["component"], row["bus"], row["carrier"], row["scale_factor"]
            )
            if component == "generator":
                mask = (n.generators.bus == bus) & (n.generators.carrier == carrier)
                n.generators.loc[mask, "p_nom"] *= scale
                if "p_nom_max" in n.generators.columns:
                    n.generators.loc[mask, "p_nom_max"] *= scale
                n.generators.loc[mask, "p_nom_extendable"] = False
            elif component == "storage_unit":
                mask = (n.storage_units.bus == bus) & (n.storage_units.carrier == carrier)
                n.storage_units.loc[mask, "p_nom"] *= scale
                if "p_nom_max" in n.storage_units.columns:
                    n.storage_units.loc[mask, "p_nom_max"] *= scale
                n.storage_units.loc[mask, "p_nom_extendable"] = False

        # ── solve ──────────────────────────────────────────────────────
        # Change to open source solver if required, default is HiGHS
        # n.optimize()
        n.optimize(solver_name="gurobi")

        # ── unserved energy ───────────────────────────────────────────
        if "Unserved Energy" in n.generators.carrier.values:
            ue_cols = n.generators[n.generators.carrier == "Unserved Energy"].index
            ue_df = n.generators_t.p[ue_cols]
            ue_GWh = ue_df.sum().sum() / 1e3
            ue_by_bus = (
                ue_df.sum(axis=0)
                .groupby(n.generators.loc[ue_cols, "bus"]).sum() / 1e3
            ).round(3).to_dict()
        else:
            ue_GWh, ue_by_bus = 0.0, {}

        # ── gas energy ────────────────────────────────────────────────
        gas_idx = n.generators.index[n.generators.carrier == "Gas"]
        gas_GWh = 0.0
        if len(gas_idx):
            gas_GWh = n.generators_t.p[gas_idx].sum().sum() / 1e3  # to GWh
        gas_GWh = round(gas_GWh, 3)
        
        # ── curtailment (built‑in helper) ─────────────────────────
        if hasattr(n, "statistics") and hasattr(n.statistics, "curtailment"):
            curtailment_df = n.statistics.curtailment()
            
            existing_carriers = curtailment_df.index.get_level_values(1).unique()
            wanted_carriers = ['Wind', 'Solar', 'Rooftop Solar']
            valid_carriers = [c for c in wanted_carriers if c in existing_carriers]
            if valid_carriers:
                vre_curt_GWh = curtailment_df.loc[(slice(None), valid_carriers)].sum() / 1e3
            else:
                vre_curt_GWh = 0.0
        else:
            vre_curt_GWh = 0.0
        
        # ── Battery Capacity ─────────────────────────
        battery_carrier = "Battery"  # adjust as needed
        battery_mask = n.storage_units.carrier == battery_carrier

        battery_capacity_GW = 0.0
        if battery_mask.any():
            battery_capacity_GW = n.storage_units.loc[battery_mask, "p_nom"].sum() / 1e3  # MW to GW
            battery_capacity_GW = round(battery_capacity_GW, 3)
        
        # ── Generator capacity by carrier ─────────────────────────
        capacity_by_carrier = (
            n.generators[n.generators.carrier != 'Unserved Energy']
            .groupby("carrier")["p_nom"]
            .sum()
            .round(3)
            .to_dict()
        )
        
        # ── collect metrics ───────────────────────────────────────────
        results.append({
            "Scenario": scenario,
            "Objective": "\n".join(
                f"{r['bus']}-{r['carrier']}: x{r['scale_factor']}" for _, r in df_s.iterrows()
            ),
            # "Total System Cost (B$)": round(n.objective / 1e9, 3),
            # "Total New Capacity (GW)": round(n.generators.p_nom_opt.sum() / 1e3, 3),
            "Gas Generation (GWh)": gas_GWh,
            # "Total Curtailment (GWh)": round(curt_GWh, 3),
            "Battery Capacity (GW)": battery_capacity_GW,
            "Generator Capacity (GW)": dict_to_multiline_string(capacity_by_carrier),
            "Wind & Solar Curtailment (GWh)": round(vre_curt_GWh, 3),
            "Unserved Energy (GWh)": round(ue_GWh, 3),
            "Unserved by Region (GWh)": dict_to_multiline_string(ue_by_bus)
            
        })

        # ── export solved network ─────────────────────────────────────
        n.export_to_netcdf(os.path.join(export_dir, f"scenario_{scenario}.nc"))

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    df_results.to_csv(os.path.join(export_dir, f"scenarios_summary_{timestamp}.csv"), index=False)
    
    total_time = round(perf_counter() - timer_all, 1)
    print(f"All scenarios completed in {total_time} seconds")
    
    return df_results

# OPTIONAL - add (Project EnergyConnect) interconnector prior to generating scenarios
# Added to csv file, but can also be added here.
# add_pec_interconnector(n, s_nom=800, extendable=False)

# GENERATE SCENARIOS
# RESET network to baseline (n) before running scenarios,
# otherwise scaling factors will be applied to previous scenario.
n = initialize_network_with_unserved_energy("results/high-level_nem.nc")

# Optionally, increase demand given percentage
# 80% gets pretty close to ISP 2039-40 operational demand profile
increase_demand_by_percentage(n, 80)
n.optimize(solver_name="gurobi")

n.generators
n.storage_units
n.lines
n.buses
# Generate scenarios with different technology assumptions
# Note: Ensure the tech_assumptions_path points to a valid CSV file with the correct

df_results = generate_scenarios(n, tech_assumptions_path="data/inputs/tech_assumptions.csv")

df_results



# Optionally; load from previously saved network file:
df_results = pd.read_csv("results/scenarios/scenarios_summary_20250711_1007.csv")
df_results["Objective"] = df_results["Objective"].str.replace("\\n", "\n")
# Choose a scenario to view then assign objective_text (not relevant to baseline)
scenario = "0_2024_baseline"
scenario = "1_BalancedTransition"
scenario = "2_BalancedAggressiveTransition"
scenario = "3.0_VreStorageRampTransition"
scenario = "3.1_VreStorageRampGasReduction"
scenario = "3.1.2_VreStorageRampGasReduction"
scenario = "3.2_VreStorageRampGasReduction"
scenario = "3.3_VreStorageRampGasReduction"
scenario = "4_VreStorageRampTransitionZeroCoal"
scenario = "4_4xVreTransitionZeroCoal"
scenario = "5_5xVreTransitionZeroCoal"
scenario = "6.0_6xVreTransitionZeroCoal"
scenario = "6.1_6xVreTransitionZeroCoal"
scenario = "6.2_6xVreTransitionZeroCoal"
scenario = "7.0_6xVre4xRoofZeroCoal"
scenario = "8.0_6xVre&BatteryZeroCoal"
scenario = "8.1_6xVre&BatteryZeroCoal"
scenario = "8.2_6xVre&BatteryZeroCoal"
scenario = "1.4_10xVre&BatteryZeroFF"
scenario = "1.5_11xVre&BatteryZeroFF"
scenario = "8.2.1_6xVreCurtailReview"
objective_text = df_results.loc[df_results["Scenario"] == scenario, "Objective"].values[0]


# Load scenarios for analysis
# Baseline (2024)
n = pypsa.Network("results/high-level_nem.nc")



# Scenarios
n = pypsa.Network("results/scenarios/scenario_1_BalancedTransition.nc")
n = pypsa.Network("results/scenarios/scenario_2_BalancedAggressiveTransition.nc")
n = pypsa.Network("results/scenarios/scenario_3.0_VreStorageRampTransition.nc")
n = pypsa.Network("results/scenarios/scenario_3.1_VreStorageRampGasReduction.nc")
n = pypsa.Network("results/scenarios/scenario_3.1.2_VreStorageRampGasReduction.nc")
n = pypsa.Network("results/scenarios/scenario_3.2_VreStorageRampGasReduction.nc")
n = pypsa.Network("results/scenarios/scenario_3.3_VreStorageRampGasReduction.nc")
n = pypsa.Network("results/scenarios/scenario_4_VreStorageRampTransitionZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_4_4xVreTransitionZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_5_5xVreTransitionZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_6.0_6xVreTransitionZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_6.1_6xVreTransitionZeroCoal.nc")

n = pypsa.Network("results/scenarios/scenario_6.2_6xVreTransitionZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_7.0_6xVre4xRoofZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_8.0_6xVre&BatteryZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_8.1_6xVre&BatteryZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_8.2_6xVre&BatteryZeroCoal.nc")
n = pypsa.Network("results/scenarios/scenario_1.4_10xVre&BatteryZeroFF.nc")
n = pypsa.Network("results/scenarios/scenario_1.5_11xVre&BatteryZeroFF.nc")
n = pypsa.Network("results/scenarios/scenario_8.2.1_6xVreCurtailReview.nc")



# Plot dispatch at various intervals
plot_dispatch(n, time="2024-04", days=90, regions=["NSW1"], scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024", regions=["TAS1"], scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024", regions=None, scenario_name=scenario, scenario_objective=objective_text)
# Curtailment can show here
plot_dispatch(n, time="2024-06-01", days=60, regions=None, scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024-06-12", days=5, regions=["NSW1"], scenario_name=scenario, scenario_objective=objective_text)

# Curtailment can show here
plot_dispatch(n, time="2024-06-10", days=6, regions=["VIC1"], show_imports=True, interactive=False, scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024-01-10", days=10, regions=["VIC1"], show_imports=True, interactive=True, scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024-08", days=6, regions=["VIC1"], show_imports=True, interactive=True, scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024", regions=["VIC1"], show_imports=True)
plot_dispatch(n, time="2024", regions=["SA1"], show_imports=True)
plot_dispatch(n, time="2024-05-15", days=10, regions=["NSW1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, show_curtailment=True)
# Whole day of curtailed solar - 19.05
plot_dispatch(n, time="2024-05-18", days=10, regions=["VIC1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=False)
plot_dispatch(n, time="2024-04-01", days=10, regions=["QLD1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=True)
plot_dispatch(n, time="2024-05", regions=["VIC1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024-05-21", days=8, regions=["VIC1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=True)
plot_dispatch(n, time="2024-06-17", days=6, regions=["NSW1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=False)
plot_dispatch(n, time="2024-06-17", days=2, regions=["NSW1"], show_imports=True, scenario_name=None, scenario_objective=None)
plot_dispatch(n, time="2024-07-24", days=7, regions=["VIC1"], show_imports=False)
# Low wind period
plot_dispatch(n, time="2024-05-20", days=10, regions=["SA1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=False)
plot_dispatch(n, time="2024-05-20", days=10, regions=["NSW1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=False)
plot_dispatch(n, time="2024-06-21", days=7, regions=["SA1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=True)
plot_dispatch(n, time="2024-04-07", days=90, regions=["TAS1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, interactive=True)
plot_dispatch(n, time="2024", regions=["QLD1"], show_imports=True)
plot_dispatch(n, time="2024-06-12", days=8, regions=["TAS1"], show_imports=True, interactive=True, scenario_name=scenario, scenario_objective=objective_text)
plot_dispatch(n, time="2024-07", regions=["QLD1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text, show_curtailment=True)
plot_dispatch(n, time="2024-06-05",days=8, regions=["VIC1"], show_imports=True, scenario_name=scenario, scenario_objective=objective_text)


# Note: bus and region are synonymous in this context.
def curtailment_region_carrier(n, carriers=['Rooftop Solar', 'Solar', 'Wind']):
    """
    Calculate curtailment % by bus and carrier (excluding Hydro) and pivot carriers wide.
    Adds one monthly curtailed TWh string column per carrier named
    'monthly_curtailment_<carrier>_twh_str'.

    Fixed so that 'month' in the intermediate DataFrame is a Timestamp,
    which lets us do .to_period('M') later without error.
    """
    
    records = []
    monthly_data = {carrier: [] for carrier in carriers}

    # create a PeriodIndex of each snapshot’s month
    snapshot_month = n.snapshots.to_series().dt.to_period('M')

    for carrier in carriers:
        mask = n.generators.carrier == carrier
        gens = n.generators[mask]

        for bus, gens_in_bus in gens.groupby('bus'):
            idx = gens_in_bus.index.intersection(n.generators_t.p_max_pu.columns)

            # overall curtailment % per bus/carrier
            if len(idx) == 0:
                pct = 0.0
            else:
                pot = (n.generators_t.p_max_pu[idx]
                       .multiply(n.generators.loc[idx, 'p_nom'], axis=1)
                      ).sum().sum()
                act = n.generators_t.p[idx].sum().sum()
                curtailed = pot - act
                pct = 100 * curtailed / pot if pot > 0 else 0.0

            records.append({'bus': bus, 'carrier': carrier, 'curtailment_pct': round(pct, 3)})

            # monthly curtailed MWh per bus/carrier
            if len(idx) > 0:
                pot_m = (n.generators_t.p_max_pu[idx]
                         .multiply(n.generators.loc[idx, 'p_nom'], axis=1))
                act_m = n.generators_t.p[idx]

                pot_mon = pot_m.groupby(snapshot_month).sum().sum(axis=1)
                act_mon = act_m.groupby(snapshot_month).sum().sum(axis=1)
                curtailed_mon = pot_mon - act_mon
            else:
                # build a zero‐series indexed by each unique period
                curtailed_mon = pd.Series(0.0, index=snapshot_month.unique())

            # *** store month as a Timestamp, not Period ***
            for period, val in curtailed_mon.items():
                monthly_data[carrier].append({
                    'month': period.to_timestamp(),  # <- convert here
                    'bus': bus,
                    'curtailed_mwh': val
                })

    # build the bus×carrier % pivot
    df = pd.DataFrame(records)
    pivot_df = df.pivot(index='bus', columns='carrier', values='curtailment_pct').fillna(0.0)

    # for each carrier, build its monthly TWh‐string column
    for carrier in carriers:
        mon_df = pd.DataFrame(monthly_data[carrier])
        summed = (mon_df.groupby(['bus', 'month'])['curtailed_mwh']
                     .sum()
                     .reset_index())

        # now month is Timestamp, so this works:
        start = summed['month'].min().to_period('M')
        end   = summed['month'].max().to_period('M')
        months_sorted = pd.period_range(start, end, freq='M').to_timestamp()

        ser = {}
        for bus in pivot_df.index:
            subset = summed[summed['bus'] == bus].set_index('month')['curtailed_mwh'].to_dict()
            arr = np.array([round(subset.get(m, 0.0), 3) for m in months_sorted])
            twh = arr / 1e6 # convert MWh to TWh
            ser[bus] = ' '.join(f'{x:.3f}' for x in twh)

        col = f'monthly_curtailment_{carrier.lower().replace(" ", "_")}_twh_str'
        pivot_df[col] = pivot_df.index.map(ser)

    return pivot_df


pivot_df = curtailment_region_carrier(n)

from great_tables import GT, md, system_fonts, nanoplot_options

curtailment_tbl = GT(data=pivot_df \
    .reset_index() \
    .round(2) \
    .rename(columns={'bus': 'Region',
                     'monthly_curtailment_rooftop_solar_twh_str': 'Rooftop Solar Curtailment',
                     'monthly_curtailment_solar_twh_str': 'Solar Curtailment',
                     'monthly_curtailment_wind_twh_str': 'Wind Curtailment'
                     })
    )

# Generate great table for curtailment by region and carrier
# Note: Scale-up objective is hardcoded in the source note.
# objective_text.replace('\n', ', ') 
curtailment_gt = curtailment_tbl.tab_header(
        title="NEM Variable Renewable Energy Curtailment by Region"
        ) \
        .tab_spanner(
        label="Curtailment (%)",
        columns=['Rooftop Solar', 'Solar', 'Wind']
        ) \
        .tab_spanner(
            label="Monthly Curtailment Profiles (TWh)",
            columns=['Rooftop Solar Curtailment',
                     'Solar Curtailment',
                     'Wind Curtailment']
        ) \
        .data_color(
        columns=['Rooftop Solar', 'Solar', 'Wind'],
        palette=[
            "#31a354", "#78c679", "#ffffcc",
            "#fafa8c", "#f4cd1e", "#f8910b"],
        domain=[0, 100]
        ) \
        .cols_width(
            {'Wind': '120px', 'Solar': '120px', 'Rooftop Solar': '120px'}
        ) \
        .cols_align(    
             align='center'
        ) \
        .tab_source_note(
            md(
                '<br><div style="text-align: left;">'
                "**Calculation**: Curtailment = (Potential energy - Actual energy) / Potential energy * 100."
                "</div>"
                           
            )
        ) \
        .tab_source_note(
            source_note=md("**Scenario**: 8.2.1_6xVreCurtailReview (Zero coal & 83% reduction in GPG). **Scale-up objective from 2024 baseline**: *NSW1-Black Coal: x0.0, NSW1-Gas: x0.5, NSW1-Rooftop Solar: x3.0, NSW1-Solar: x3.0, NSW1-Wind: x6.0, NSW1-Battery: x8.0, QLD1-Black Coal: x0.0, QLD1-Gas: x0.5, QLD1-Rooftop Solar: x3.0, QLD1-Solar: x3.0, QLD1-Wind: x6.0, QLD1-Battery: x6.0, SA1-Gas: x0.5, SA1-Rooftop Solar: x2.0, SA1-Solar: x3.0, SA1-Wind: x2.0, SA1-Battery: x6.0, TAS1-Gas: x0.5, TAS1-Rooftop Solar: x3.0, TAS1-Wind: x5.0, TAS1-Battery: x8.0, VIC1-Gas: x0.5, VIC1-Brown Coal: x0.0, VIC1-Rooftop Solar: x4.0, VIC1-Solar: x4.0, VIC1-Wind: x4.0, VIC1-Battery: x8.0*")
        ) \
        .fmt_nanoplot("Rooftop Solar Curtailment",
                      plot_type="bar",
                      options=nanoplot_options(
                          data_bar_fill_color="#FFE066",
                          y_ref_line_fmt_fn="GWh",
                          )
        ) \
        .fmt_nanoplot("Solar Curtailment",
                      plot_type="bar",
                      options=nanoplot_options(
                          data_bar_fill_color="#FDB324",
                          y_ref_line_fmt_fn="GWh",
                          )
        ) \
        .fmt_nanoplot("Wind Curtailment",
                      plot_type="bar",
                      options=nanoplot_options(
                          data_bar_fill_color="#3BBFE5",
                          y_ref_line_fmt_fn="GWh",
                          )
        ) \
        .tab_options(
            source_notes_font_size='x-small',
            source_notes_padding=3,
            heading_subtitle_font_size='small',
            table_font_names=system_fonts("humanist"),
            data_row_padding='1px',
            heading_background_color='#F0FFFF',
            source_notes_background_color='#F0FFF0',
            column_labels_background_color='#F0FFF0',
            table_background_color='snow',
            data_row_padding_horizontal=3,
            column_labels_padding_horizontal=58
        ) \
            .opt_table_outline()

curtailment_gt

curtailment_gt.save("curtailment_by_region_and_carrier.png")

# Review single day curtailment
# 1) pick your target date
target_day = pd.to_datetime("2024-05-19")

# 2) grab all snapshots on that day
day_snaps = [ts for ts in n.snapshots if ts.date() == target_day.date()]


# availability (MW) and dispatch (MW) for VRE
gen_mask = n.generators.bus.isin(['VIC1'])
mask_vre = ["Wind","Solar"]
mask_vre = gen_mask & n.generators.carrier.isin(mask_vre)

avail = (
    n.generators_t.p_max_pu.loc[day_snaps, mask_vre]
      .multiply(n.generators.loc[mask_vre, "p_nom"], axis=1)
)
disp  = n.generators_t.p.loc[day_snaps, mask_vre]

# per‐snapshot curtailment (MW), clipped at zero → then convert to GW
curt_ts = avail.sub(disp, fill_value=0).clip(lower=0).sum(axis=1).div(1e3)

curt_ts.plot(title=f"VRE Curtailment on {target_day.strftime('%Y-%m-%d')}", ylabel="GW")

# Count approximate LORs

def count_reserve_breaches_df(n, threshold_mw=850):
    """
    Count how many hours in 2024 the reserve margin
    (available capacity minus load) at each bus dips below threshold_mw.
    Returns a DataFrame with breach counts and breach ratios.

    Parameters
    ----------
    n : pypsa.Network
        A solved PyPSA network.
    threshold_mw : float or dict
        If float, same threshold (in MW) for every bus.
        If dict, keys are bus names and values are thresholds for those buses;
        any bus not in the dict uses the global default (if provided) or 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          'breach_hours': number of hours below threshold
          'breach_ratio': breach_hours / total_snapshots
        Index is bus name.
    """
    avail = n.generators_t.p_max_pu.multiply(n.generators.p_nom, axis=1)
    avail_by_bus = avail.T.groupby(n.generators.bus).sum().T

    loads = n.loads_t.p_set
    load_by_bus = loads.T.groupby(n.loads.bus).sum().T

    all_buses = sorted(set(avail_by_bus.columns) | set(load_by_bus.columns))
    avail_by_bus = avail_by_bus.reindex(columns=all_buses, fill_value=0)
    load_by_bus  = load_by_bus.reindex(columns=all_buses, fill_value=0)
    avail_by_bus = avail_by_bus.reindex(index=load_by_bus.index, fill_value=0)

    reserve = avail_by_bus.subtract(load_by_bus, fill_value=0)

    if isinstance(threshold_mw, dict):
        thresh = pd.Series({bus: threshold_mw.get(bus, 0.0) for bus in all_buses})
    else:
        thresh = pd.Series(threshold_mw, index=all_buses)

    breaches = (reserve.lt(thresh, axis=1)).sum()

    total_snapshots = reserve.shape[0]

    df = pd.DataFrame({
        'breach_hours': breaches,
        'breach_ratio': breaches / total_snapshots
    })

    return df

# For per-region thresholds
# Source: 2024 ISP Inputs and Assumptions workbook.xlsx (sheet: Reserves)
per_region = {"NSW1":705, "VIC1":550, "QLD1":710, "SA1":195, "TAS1":140}
print(count_reserve_breaches_df(n, threshold_mw=per_region))

# Note: a MSL constraint could be added to the network to prevent this. This however is likely to increase solving time and complexity.
# Review gas generators
gas = (
    n.generators
     .query("carrier == 'Gas'")
     .filter(['bus','p_nom'], axis=1)
     .reset_index()
     .rename(columns={'index':'name'})
     )

# add per-region lookup
gas['reserve'] = gas['bus'].map(per_region)

# Reorder columns
gas = gas[['bus', 'p_nom', 'reserve']]

print(gas)

# Reviewing the p_nom of gas generators in this scenario, there is enough coverage for MSL directed events. TAS1 has the lowest reserve margin, but still has 140 MW of gas generation available.

# ------------------------------------------------------------------
# Add Cellars Hill Battery 600 MW battery at bus TAS1
# ------------------------------------------------------------------
# Streamlined major project status 
# If this is not added, scenarios multiply by zero, giving no scale-up
# Note: this battery seems the most advanced, but is not committed yet in 
# Open Electricity.
# Now added to storage_units.csv, so this is not needed.
name = "TAS1-BATTERY"

if name in n.storage_units.index:
    # If it already exists, just update p_nom
    n.storage_units.loc[name, "p_nom"] = 600
else:
    # Otherwise create it with some reasonable defaults
    n.add(
        "StorageUnit",
        names       =[name],
        bus         =["TAS1"],
        carrier     =["Battery"],
        p_nom       =[600],          # MW
        max_hours   =[2],            # energy = 2 × p_nom  (adjust as needed)
        efficiency_store=[0.95],      # charge efficiency
        efficiency_dispatch=[0.95],   # discharge efficiency
        capital_cost=[922000],            # leave 0 if you’re not using investment optimisation
        marginal_cost=[0],           # storage usually treated as zero‐marginal
        p_nom_extendable=[False],    # fixed-size asset
    )


# Total supply by generator
n.statistics.supply(comps=["Generator"]).droplevel(0) / 1e3  # Convert to GWh

def calculate_total_curtailment(network, carriers=None):
    """
    Calculate total curtailment (in GWh) for specified carrier(s) across the network.

    Parameters
    ----------
    network : pypsa.Network
    carriers : list of str, optional
        List of generator carriers to include. If None, use all generators with p_max_pu.

    Returns
    -------
    pandas.DataFrame
        Total curtailment (GWh) by carrier and bus.
    """
    gen = network.generators
    gen_t = network.generators_t

    # Only use generator names present in both p and p_max_pu
    valid_gens = gen.index.intersection(gen_t.p.columns).intersection(gen_t.p_max_pu.columns)

    if carriers is not None:
        carrier_mask = gen.loc[valid_gens, "carrier"].isin(carriers)
        relevant_gens = valid_gens[carrier_mask]
    else:
        relevant_gens = valid_gens

    # Calculate curtailment
    dispatch = gen_t.p[relevant_gens]
    availability = gen_t.p_max_pu[relevant_gens].multiply(gen.loc[relevant_gens, "p_nom"], axis=1)

    curtailment = (availability - dispatch).clip(lower=0)
    curtailment_GWh = curtailment.sum() / 1e3  # Convert to GWh

    # Add metadata and group
    meta = gen.loc[relevant_gens, ["bus", "carrier"]]
    result = curtailment_GWh.to_frame("curtailment_GWh").join(meta)
    summary = result.groupby(["carrier", "bus"]).sum().reset_index()

    return summary



n = pypsa.Network("results/scenarios/scenario_3.3_VreStorageRampGasReduction.nc")
# Doesn't look correct, higher using baseline than 3.3. Could be soaked up with exports and storage.
curt_summary = calculate_total_curtailment(n, carriers=["Wind", "Solar", "Rooftop Solar"])
print(curt_summary)

calculate_total_curtailment(n).groupby("carrier").curtailment_GWh.sum()

# Total supply by generator
n.statistics.supply(comps=["Generator"]).droplevel(0) / 1e3  # Convert to GWh

# Wind and Solar curtailment as a percentage of total supply
n.statistics.curtailment().loc[(slice(None), ['Wind','Solar'])].sum() / 1e3 / (n.statistics.supply(comps=["Generator"]).sum() / 1e3)


# Calculate total supply excluding certain carriers (default: "Unserved Energy")
# This is useful to get the total generation excluding unserved energy or other non-generating carriers
def total_supply_excluding(network, exclude_carriers=None):
    """Total generator supply (GWh) excluding certain carriers."""
    if exclude_carriers is None:
        exclude_carriers = ["Unserved Energy"]

    s = network.statistics.supply(comps=["Generator"])  # MWh, Multi-Index
    keep = ~s.index.get_level_values("carrier").isin(exclude_carriers)
    return s.loc[keep].sum() / 1e3                      # → GWh

total_gen = total_supply_excluding(n)
vre_curt  = n.statistics.curtailment() \
    .loc[(slice(None), ["Wind", "Solar"])] \
        .sum() / 1e3                             

pct = round(vre_curt / total_gen * 100, 2)
print(f"Wind+Solar curtailment: {pct}% of real generation")



# Residual load plots show net demand after subtracting variable renewable generation (typically wind + solar), revealing when the system relies on dispatchable generation, storage, or imports.
# Positive residual = system must dispatch other resources (e.g. coal, gas, imports, storage discharge).
# Negative residual = renewable surplus (could go to storage, exports, or be curtailed).

def plot_residual_load(n, time="2024", days=None, regions=None, carriers=("Solar", "Wind", "Rooftop Solar")):
    """
    Plot residual load (demand minus VRE generation).

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object.
    time : str
        Start date ("YYYY", "YYYY-MM", or "YYYY-MM-DD").
    days : int, optional
        Number of days to plot.
    regions : list of str, optional
        List of buses to include. If None, includes all.
    carriers : tuple of str
        VRE carriers to subtract from load.
    """
    

    # Define time window
    parts = time.split("-")
    if len(parts) == 1:
        start = pd.to_datetime(f"{parts[0]}-01-01")
        end = pd.to_datetime(f"{parts[0]}-12-31 23:00")
    elif len(parts) == 2:
        start = pd.to_datetime(f"{parts[0]}-{parts[1]}-01")
        end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
    elif len(parts) == 3:
        start = pd.to_datetime(time)
        end = start + pd.Timedelta(hours=23)
    if days is not None:
        end = start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)

    # Filter buses
    if regions:
        buses = set(regions)
        load_series = n.loads_t.p_set[n.loads[n.loads.bus.isin(buses)].index].sum(axis=1)
        gen_mask = n.generators.bus.isin(buses)
    else:
        load_series = n.loads_t.p_set.sum(axis=1)
        gen_mask = pd.Series(True, index=n.generators.index)

    # VRE generation
    vre_mask = gen_mask & n.generators.carrier.isin(carriers)
    vre_gen = n.generators_t.p.loc[:, vre_mask].sum(axis=1)

    # Residual load
    residual = load_series - vre_gen

    # Convert to GW
    residual /= 1e3
    load_series /= 1e3
    vre_gen /= 1e3

    # Time slice
    residual = residual.loc[start:end]
    load_series = load_series.loc[start:end]
    vre_gen = vre_gen.loc[start:end]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    residual.plot(ax=ax, label="Residual Load", color="black")
    load_series.plot(ax=ax, label="Total Demand", color="green", alpha=0.6)
    vre_gen.plot(ax=ax, label="VRE Generation", color="orange", alpha=0.6)

    ax.set_ylabel("GW")
    ax.set_title(f"Residual Load: {start.date()} to {end.date()}" + (f" | {', '.join(regions)}" if regions else ""))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


plot_residual_load(n, time="2024-09", days=30, regions=["SA1"])
plot_residual_load(n, time="2024-06", days=30, regions=["NSW1"])
plot_residual_load(n, time="2024-06", days=30, regions=["VIC1"])
plot_residual_load(n, time="2024-09", days=30, regions=["QLD1"])
plot_residual_load(n, time="2024-09", days=30, regions=None)




def plot_gas_energy_across_scenarios(nc_dir, gas_carrier="Gas"):
    """
    Builds a bar chart of total Gas generation (GWh) for every .nc file in nc_dir.
    Assumes filenames look like 'scenario_<Scenario>.nc'.
    """
    data = []
    for path in glob.glob(os.path.join(nc_dir, "scenario_*.nc")):
        scenario = os.path.basename(path).replace("scenario_", "").replace(".nc", "")
        n = pypsa.Network(path)

        idx = n.generators.index[n.generators.carrier == gas_carrier]
        gas_GWh = n.generators_t.p[idx].sum().sum() / 1e3
        data.append({"Scenario": scenario, "Gas_GWh": gas_GWh})

    df = pd.DataFrame(data).set_index("Scenario").sort_values(by='Gas_GWh')

    # ── bar chart ───────────────────────────────────────────────────────
    ax = df["Gas_GWh"].plot.bar(color="#E6622D", figsize=(8, 6))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.margins(x=0.01)  # optional: tiny side padding
    ax.set_ylabel("Gas Generation (GWh)")
    ax.set_title(f"Total Gas Dispatch Across Scenarios ({gas_carrier})")
    plt.tight_layout()
    plt.show()

    return df  # handy if you want to inspect the numbers too

gas_scenarios = plot_gas_energy_across_scenarios(
    nc_dir="results/scenarios"
)

gas_scenarios = gas_scenarios.reset_index()

baseline_val = gas_scenarios.loc[gas_scenarios["Scenario"] == "0_2024_baseline", "Gas_GWh"].iloc[0]

gas_scenarios["Gas_%_vs_baseline"] = ((gas_scenarios["Gas_GWh"] / baseline_val) - 1.0) * 100
gas_scenarios = gas_scenarios.round({"Gas_GWh": 3, "Gas_%_vs_baseline": 1})  # keep numbers tidy
gas_scenarios

# Optionally remove transmission lines from the network
n.lines["s_nom"] = 0
n.lines["s_nom_extendable"] = False
n.links["p_nom"] = 0
n.links["p_nom_extendable"] = False

# Review Hydro generation
# Get indices of Hydro generators
hydro_gens = n.generators.index[n.generators.carrier == 'Hydro']

# Find intersection with columns in p_max_pu (in case some Hydro gens don't have p_max_pu)
hydro_cols = hydro_gens.intersection(n.generators_t.p_max_pu.columns)

# Select only Hydro columns from p_max_pu
hydro_p_max_pu = n.generators_t.p_max_pu[hydro_cols]
hydro_p = n.generators_t.p[hydro_cols]

# VIC1 capacity factor is very low, particularly the median @ 7%
print(n.generators.loc[hydro_cols, ['p_nom', 'carrier']])
print(n.generators_t.p_max_pu[hydro_cols].describe())

# Plot hydro dispatch vs potential
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

def plot_hydro_dispatch_vs_potential(n):
    hydro_gens  = n.generators.index[n.generators.carrier == 'Hydro']
    hydro_cols  = hydro_gens.intersection(n.generators_t.p_max_pu.columns)
    n_hydro     = len(hydro_cols)
    ncols       = 2
    nrows       = (n_hydro + ncols - 1) // ncols

    # set up the date locator/formatter once
    locator   = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(8, 4*nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    legend_handles, legend_labels = [], []

    for i, gen in enumerate(hydro_cols):
        ax = axes[i]
        potential = n.generators_t.p_max_pu[gen] * n.generators.loc[gen, 'p_nom']
        dispatch  = n.generators_t.p[gen]

        ln1 = ax.plot(
            n.snapshots, potential,
            label='Potential', linestyle=':', color='orange', alpha=0.6
        )
        ln2 = ax.plot(
            n.snapshots, dispatch,
            label='Dispatch', linestyle='-', color='#1E81D4'
        )

        # apply concise date formatting
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelrotation=0, labelsize=8)

        ax.set_title(f'Hydro Gen: {gen}')
        ax.set_facecolor('#F0FFFF')
        ax.grid(True)

        if i == 0:
            legend_handles = ln1 + ln2
            legend_labels  = ['Potential', 'Dispatch']

    # drop any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # one shared legend & axis labels
    fig.legend(legend_handles, legend_labels,
               loc='upper center', ncol=2, frameon=False)
    fig.supxlabel('Time')
    fig.supylabel('Power (MW)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



plot_hydro_dispatch_vs_potential(n)

# Ilustrate the sensitivity of Hydro generation to marginal cost
# Set marginal_cost to zero
# Get indices of Hydro generators
hydro_gens = n.generators.index[n.generators.carrier == 'Hydro']

n.generators.loc[hydro_gens, 'marginal_cost'] = 0.0
n.optimize()
n.optimize(solver="gurobi")
plot_hydro_dispatch_vs_potential(n)


# EDA on solar vs rooftop solar CF in 4 regions
# Solar vs rooftop solar CF in 4 regions
# Solar vs rooftop solar CF in 4 regions
solar_cf = n.generators_t.p_max_pu[['SA1-SOLAR', 'SA1-ROOFTOP-SOLAR',
                         'VIC1-SOLAR', 'VIC1-ROOFTOP-SOLAR',
                         'QLD1-SOLAR', 'QLD1-ROOFTOP-SOLAR',
                         'NSW1-SOLAR', 'NSW1-ROOFTOP-SOLAR']].reset_index()

solar_cf['hour'] = solar_cf['snapshot'].dt.hour
hours = sorted(solar_cf['hour'].unique())
even_hours = [h for h in hours if h % 2 == 0]
even_positions = [hours.index(h) for h in even_hours]

combos = [
    ['SA1-SOLAR', 'SA1-ROOFTOP-SOLAR'],
    ['VIC1-SOLAR', 'VIC1-ROOFTOP-SOLAR'],
    ['QLD1-SOLAR', 'QLD1-ROOFTOP-SOLAR'],
    ['NSW1-SOLAR', 'NSW1-ROOFTOP-SOLAR']
]

fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(8, 8),
    sharex=True, sharey=True   # share y for common scale
)
axes = axes.flatten()

width = 0.35
x = np.arange(len(hours))

for ax, (sol_col, rt_col) in zip(axes, combos):
    sol_data = [solar_cf[solar_cf['hour']==h][sol_col].dropna().values for h in hours]
    rt_data  = [solar_cf[solar_cf['hour']==h][rt_col].dropna().values for h in hours]

    bp1 = ax.boxplot(
        sol_data,
        positions=x - width/2, widths=width, patch_artist=True,
        boxprops=dict(facecolor='#FDB324', edgecolor='black'),
        medianprops=dict(color='blue'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(markeredgecolor='black')
    )
    bp2 = ax.boxplot(
        rt_data,
        positions=x + width/2, widths=width, patch_artist=True,
        boxprops=dict(facecolor='#FFE066', edgecolor='black'),
        medianprops=dict(color='green'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(markeredgecolor='black')
    )

    ax.set_title(f"{sol_col} vs {rt_col}")
    ax.set_xticks(even_positions)
    ax.set_xticklabels(even_hours)
    ax.set_facecolor('#F0FFFF')
    # no individual ylabel here

# shared legend
handles = [bp1["boxes"][0], bp2["boxes"][0]]
labels  = ['Utility-scale Solar', 'Rooftop Solar']
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))

# one common y-axis label
fig.supxlabel('Hour of Day')
fig.supylabel('Capacity Factor')

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# What got dispatched.

solar_dispatch = n.generators_t.p[['SA1-SOLAR', 'SA1-ROOFTOP-SOLAR',
                         'VIC1-SOLAR', 'VIC1-ROOFTOP-SOLAR',
                         'QLD1-SOLAR', 'QLD1-ROOFTOP-SOLAR',
                         'NSW1-SOLAR', 'NSW1-ROOFTOP-SOLAR']].reset_index()

solar_dispatch['hour'] = solar_dispatch['snapshot'].dt.hour
hours = sorted(solar_dispatch['hour'].unique())
combos = [
    ['SA1-SOLAR', 'SA1-ROOFTOP-SOLAR'],
    ['VIC1-SOLAR', 'VIC1-ROOFTOP-SOLAR'],
    ['QLD1-SOLAR', 'QLD1-ROOFTOP-SOLAR'],
    ['NSW1-SOLAR', 'NSW1-ROOFTOP-SOLAR']
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()   # so we can iterate easily

width = 0.35
x = np.arange(len(hours))

for ax, (sol_col, rt_col) in zip(axes, combos):
    # collect per-hour data
    sol_data = [solar_dispatch[solar_dispatch['hour']==h][sol_col].dropna().values for h in hours]
    rt_data  = [solar_dispatch[solar_dispatch['hour']==h][rt_col].dropna().values for h in hours]

    # light blue boxes for utility-scale
    bp1 = ax.boxplot(
        sol_data,
        positions=x - width/2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor='#FDB324', edgecolor='black'),
        medianprops=dict(color='blue'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(markeredgecolor='black')
    )
    # light green boxes for rooftop
    bp2 = ax.boxplot(
        rt_data,
        positions=x + width/2,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor='#FFE066', edgecolor='black'),
        medianprops=dict(color='green'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(markeredgecolor='black')
    )

    ax.set_title(f"{sol_col} vs {rt_col}")
    ax.set_xticks(x)
    ax.set_xticklabels(hours)
    ax.set_ylabel('Dispatch (MW)')

# shared legend
handles = [bp1["boxes"][0], bp2["boxes"][0]]
labels  = ['Utility-scale Solar', 'Rooftop Solar']
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.97))

fig.tight_layout(rect=[0,0,1,0.95])  # leave room for legend
plt.show()


# Wind vs Rooftop Solar by hour
# This is useful to compare the two VRE sources in terms of their capacity factors.
cap_factors = n.generators_t.p_max_pu.reset_index()

cap_factors['hour'] = cap_factors['snapshot'].dt.hour
hours = sorted(cap_factors['hour'].unique())

# Prepare data lists for each hour
wind_data = [cap_factors[cap_factors['hour'] == h]['VIC1-WIND'].dropna().values for h in hours]
rooftop_data = [cap_factors[cap_factors['hour'] == h]['VIC1-ROOFTOP-SOLAR'].dropna().values for h in hours]
util_data = [cap_factors[cap_factors['hour'] == h]['VIC1-SOLAR'].dropna().values for h in hours]

x = np.arange(len(hours))
group_width = 0.6              # total horizontal span of the three boxes
box_width   = group_width / 3  # width of each individual box

# compute offsets so boxes sit at -0.2, 0, +0.2 around each x
offsets = np.array([-1, 0, +1]) * box_width

plt.figure(figsize=(12,6))

# wind
plt.boxplot(
    wind_data,
    positions = x + offsets[0],
    widths    = box_width,
    patch_artist=True,
    boxprops  = dict(facecolor='#3BBFE5', edgecolor='black'),
    medianprops=dict(color='blue'),
    whiskerprops=dict(color='black'),
    capprops  = dict(color='black'),
    flierprops=dict(markeredgecolor='black')
)

# util solar
plt.boxplot(
    util_data,
    positions = x + offsets[1],
    widths    = box_width,
    patch_artist=True,
    boxprops  = dict(facecolor='#FDB324', edgecolor='black'),
    medianprops=dict(color='orange'),
    whiskerprops=dict(color='black'),
    capprops  = dict(color='black'),
    flierprops=dict(markeredgecolor='black')
)

# rooftop
plt.boxplot(
    rooftop_data,
    positions = x + offsets[2],
    widths    = box_width,
    patch_artist=True,
    boxprops  = dict(facecolor='#FFE066', edgecolor='black'),
    medianprops=dict(color='gold'),
    whiskerprops=dict(color='black'),
    capprops  = dict(color='black'),
    flierprops=dict(markeredgecolor='black')
)

plt.xticks(x, hours)
plt.xlabel('Hour of Day')
plt.ylabel('CF')
plt.title('VIC1: Wind vs Utility‐Solar vs Rooftop‐Solar by Hour')
plt.legend(
    [plt.Line2D([0],[0], color='#3BBFE5', lw=6),
     plt.Line2D([0],[0], color='#FDB324', lw=6),
     plt.Line2D([0],[0], color='#FFE066', lw=6)],
    ['VIC1‐WIND', 'VIC1‐UTILITY‐SOLAR', 'VIC1‐ROOFTOP‐SOLAR'],
    loc='upper right'
)

plt.tight_layout()
plt.show()

# Dispatch by hour for VIC1-WIND, VIC1-ROOFTOP-SOLAR, and VIC1-UTILITY-SOLAR
dispatch = n.generators_t.p.reset_index()

dispatch['hour'] = dispatch['snapshot'].dt.hour
hours = sorted(dispatch['hour'].unique())

# Prepare data lists for each hour
wind_data = [dispatch[dispatch['hour'] == h]['VIC1-WIND'].dropna().values for h in hours]
rooftop_data = [dispatch[dispatch['hour'] == h]['VIC1-ROOFTOP-SOLAR'].dropna().values for h in hours]
util_data = [dispatch[dispatch['hour'] == h]['VIC1-SOLAR'].dropna().values for h in hours]

x = np.arange(len(hours))
group_width = 0.6              # total horizontal span of the three boxes
box_width   = group_width / 3  # width of each individual box

# compute offsets so boxes sit at -0.2, 0, +0.2 around each x
offsets = np.array([-1, 0, +1]) * box_width

plt.figure(figsize=(12,6))

# wind
plt.boxplot(
    wind_data,
    positions = x + offsets[0],
    widths    = box_width,
    patch_artist=True,
    boxprops  = dict(facecolor='#3BBFE5', edgecolor='black'),
    medianprops=dict(color='blue'),
    whiskerprops=dict(color='black'),
    capprops  = dict(color='black'),
    flierprops=dict(markeredgecolor='black')
)

# util solar
plt.boxplot(
    util_data,
    positions = x + offsets[1],
    widths    = box_width,
    patch_artist=True,
    boxprops  = dict(facecolor='#FDB324', edgecolor='black'),
    medianprops=dict(color='orange'),
    whiskerprops=dict(color='black'),
    capprops  = dict(color='black'),
    flierprops=dict(markeredgecolor='black')
)

# rooftop
plt.boxplot(
    rooftop_data,
    positions = x + offsets[2],
    widths    = box_width,
    patch_artist=True,
    boxprops  = dict(facecolor='#FFE066', edgecolor='black'),
    medianprops=dict(color='gold'),
    whiskerprops=dict(color='black'),
    capprops  = dict(color='black'),
    flierprops=dict(markeredgecolor='black')
)

plt.xticks(x, hours)
plt.xlabel('Hour of Day')
plt.ylabel('Dispatch (MW)')
plt.title('VIC1: Wind vs Utility‐Solar vs Rooftop‐Solar by Hour')
plt.legend(
    [plt.Line2D([0],[0], color='#3BBFE5', lw=6),
     plt.Line2D([0],[0], color='#FDB324', lw=6),
     plt.Line2D([0],[0], color='#FFE066', lw=6)],
    ['VIC1‐WIND', 'VIC1‐UTILITY‐SOLAR', 'VIC1‐ROOFTOP‐SOLAR'],
    loc='upper right'
)

plt.tight_layout()
plt.show()



# Curtailment analysis (Per-time-step, per-generator mean of curtailment)
p     = n.generators_t.p                  # actual dispatch (MW)
pu    = n.generators_t.p_max_pu           # per‐unit availability
p_nom = n.generators.p_nom                # capacity (MW)

curt = 1 - p.div(pu.mul(p_nom, axis=1))

# === 1) Stack to long form ===
curt_long = curt.stack().reset_index()
curt_long.columns = ['snapshot', 'generator', 'curtailment']

# === 2) Map each generator → its carrier ===
curt_long['tech'] = curt_long['generator'].map(n.generators['carrier'])

# === 3) Group the snapshot by generator, then compute mean curtailment ===
# group by generator
mean_curt_by_generator = (
    curt_long
      .groupby('generator')['curtailment']
      .mean()
)
print(mean_curt_by_generator)



# 1) Compute per-generator potential & actual energy
#    (sum over all snapshots at once)

# potential MWh per generator = sum_over_t (p_max_pu * p_nom)
pot = (n.generators_t.p_max_pu
         .mul(n.generators.p_nom, axis=1)   # shape (T × G)
       ).sum(axis=0)                        # now length-G Series

# actual MWh per generator = sum_over_t p
act = n.generators_t.p.sum(axis=0)        # length-G Series

# 2) Build a DataFrame so we can join bus/carrier metadata
df = pd.DataFrame({
    'pot_mwh': pot,
    'act_mwh': act
})
df['curt'] = df['pot_mwh'] - df['act_mwh']
df['curt_pct'] = 100 * df['curt'] / df['pot_mwh']

df

# bring in the generator attributes
df = df.join(n.generators[['bus','carrier']])

df = df.query("carrier in ['Wind', 'Solar', 'Rooftop Solar']") \
       .sort_values(by='curt_pct', ascending=False)

# 3a) If you want per-generator curtailments:
print(df['curt_pct'].dropna())


