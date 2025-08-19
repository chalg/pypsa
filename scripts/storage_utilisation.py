import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Load the network data
n = pypsa.Network("results/scenarios/8.3_VreCurtailReview.nc")


class StorageUtilisationAnalyser:
    """
    Comprehensive storage utilisation analysis for PyPSA Australian NEM model using storage_units
    """
    
    def __init__(self, network):
        self.n = network
        self.regions = ['QLD1', 'NSW1', 'VIC1', 'SA1', 'TAS1']  # Based on your data
        self.setup_data()
    
    def setup_data(self):
        """Extract and prepare storage unit data"""
        # Get storage unit data - different structure than stores
        self.storage_power = self.n.storage_units_t.p  # Power flows (+ = discharging, - = charging for storage_units!)
        self.storage_soc = self.n.storage_units_t.state_of_charge  # State of charge (should be 0-1)
        self.storage_p_nom = self.n.storage_units.p_nom_opt if 'p_nom_opt' in self.n.storage_units.columns else self.n.storage_units.p_nom
        self.storage_max_hours = self.n.storage_units.max_hours
        
        # Calculate energy capacity (MWh) from power capacity and max_hours
        self.storage_capacity = self.storage_p_nom * self.storage_max_hours
        
        # Convert SOC from absolute energy values (MWh) to percentages
        self.storage_energy = self.storage_soc.copy()  # SOC is actually energy stored
        self.soc_pct = self.storage_energy.divide(self.storage_capacity, axis=1) * 100
        
        print("\nStorage Unit Details:")
        print("-" * 50)
        for unit in self.storage_capacity.index:
            p_cap = self.storage_p_nom[unit]
            e_cap = self.storage_capacity[unit]
            max_hrs = self.storage_max_hours[unit]
            max_energy = self.storage_energy[unit].max()
            soc_range = f"{self.soc_pct[unit].min():.1f}% to {self.soc_pct[unit].max():.1f}%"
            print(f"{unit}: {p_cap:.0f} MW, {e_cap:.1f} MWh ({max_hrs:.2f}h)")
            print(f"  Max energy stored: {max_energy:.1f} MWh - SOC range: {soc_range}")
            
            # Check if SOC exceeds 100% (indicates capacity calculation issue)
            if self.soc_pct[unit].max() > 100:
                actual_capacity = self.storage_energy[unit].max()
                print(f"  WARNING: SOC > 100%! Actual capacity might be {actual_capacity:.1f} MWh")
    
    def analyse_underutilisation(self):
        """Main underutilisation analysis"""
        print("\n" + "="*60)
        print("STORAGE UNDERUTILISATION ANALYSIS")
        print("="*60)
        
        results = {}
        
        for unit in self.storage_soc.columns:
            print(f"\n--- {unit} ---")
            
            # Basic utilisation metrics
            max_soc = self.soc_pct[unit].max()
            min_soc = self.soc_pct[unit].min()
            avg_soc = self.soc_pct[unit].mean()
            capacity_mwh = self.storage_capacity[unit]
            p_capacity_mw = self.storage_p_nom[unit]
            
            # Time spent at different SOC levels
            time_empty = (self.soc_pct[unit] < 5).sum()  # < 5% SOC
            time_very_low = (self.soc_pct[unit] < 20).sum()  # < 20% SOC
            time_full = (self.soc_pct[unit] > 95).sum()  # > 95% SOC
            time_very_high = (self.soc_pct[unit] > 80).sum()  # > 80% SOC
            time_idle = (abs(self.storage_power[unit]) < 0.05 * p_capacity_mw).sum()  # < 5% of capacity
            
            total_periods = len(self.soc_pct[unit])
            
            # Calculate actual energy utilisation
            energy_range_used = self.storage_energy[unit].max() - self.storage_energy[unit].min()
            utilisation_rate = (energy_range_used / capacity_mwh) * 100
            
            # Cycling analysis
            daily_throughput = abs(self.storage_power[unit]).resample('D').sum() * 0.5  # Convert to MWh (30min periods)
            avg_daily_throughput = daily_throughput.mean()
            theoretical_daily_max = capacity_mwh * 2  # Full charge/discharge cycle
            throughput_utilisation = (avg_daily_throughput / theoretical_daily_max) * 100
            
            # Power utilisation
            max_power_used = abs(self.storage_power[unit]).max()
            power_utilisation = (max_power_used / p_capacity_mw) * 100
            
            print(f"Capacity: {p_capacity_mw:.0f} MW / {capacity_mwh:.1f} MWh")
            print(f"SOC Range: {min_soc:.1f}% to {max_soc:.1f}% (avg: {avg_soc:.1f}%)")
            print(f"Energy Range Used: {utilisation_rate:.1f}% of capacity")
            print(f"Max Power Used: {max_power_used:.1f} MW ({power_utilisation:.1f}% of capacity)")
            print(f"Avg Daily Throughput: {throughput_utilisation:.1f}% of theoretical max")
            
            print(f"\nTime Distribution:")
            print(f"  Empty (<5% SOC): {time_empty/total_periods*100:.1f}% ({time_empty} periods)")
            print(f"  Very Low (<20% SOC): {time_very_low/total_periods*100:.1f}% ({time_very_low} periods)")
            print(f"  Very High (>80% SOC): {time_very_high/total_periods*100:.1f}% ({time_very_high} periods)")
            print(f"  Full (>95% SOC): {time_full/total_periods*100:.1f}% ({time_full} periods)")
            print(f"  Idle (<5% power): {time_idle/total_periods*100:.1f}% ({time_idle} periods)")
            
            # Store results
            results[unit] = {
                'max_soc': max_soc,
                'min_soc': min_soc,
                'avg_soc': avg_soc,
                'utilisation_rate': utilisation_rate,
                'power_utilisation': power_utilisation,
                'throughput_utilisation': throughput_utilisation,
                'time_empty_pct': time_empty/total_periods*100,
                'time_full_pct': time_full/total_periods*100,
                'time_idle_pct': time_idle/total_periods*100,
                'capacity_mwh': capacity_mwh,
                'capacity_mw': p_capacity_mw
            }
        
        return results
    
    def find_underutilisation_periods(self, min_duration_hours=24):
        """Identify extended periods of underutilisation"""
        print(f"\n" + "="*60)
        print(f"EXTENDED UNDERUTILISATION PERIODS (>{min_duration_hours}h)")
        print("="*60)
        
        min_periods = min_duration_hours * 2  # 30-minute periods
        
        for unit in self.storage_soc.columns:
            print(f"\n--- {unit} ---")
            
            # Find periods where storage is nearly empty for extended time
            empty_mask = self.soc_pct[unit] < 10
            empty_periods = self.find_consecutive_periods(empty_mask, min_periods)
            
            # Find periods where storage is nearly full for extended time  
            full_mask = self.soc_pct[unit] > 90
            full_periods = self.find_consecutive_periods(full_mask, min_periods)
            
            # Find periods of minimal activity
            power_threshold = 0.05 * self.storage_p_nom[unit]  # 5% of capacity
            idle_mask = abs(self.storage_power[unit]) < power_threshold
            idle_periods = self.find_consecutive_periods(idle_mask, min_periods)
            
            print(f"Extended Empty Periods (<10% SOC for >{min_duration_hours}h):")
            self.print_periods(empty_periods)
            
            print(f"\nExtended Full Periods (>90% SOC for >{min_duration_hours}h):")
            self.print_periods(full_periods)
            
            print(f"\nExtended Idle Periods (<{power_threshold:.1f}MW activity for >{min_duration_hours}h):")
            self.print_periods(idle_periods)
    
    def find_consecutive_periods(self, mask, min_length):
        """Find consecutive True periods in boolean mask"""
        periods = []
        start = None
        
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_length:
                    periods.append((mask.index[start], mask.index[i-1], i-start))
                start = None
        
        # Check if last period extends to end
        if start is not None and len(mask) - start >= min_length:
            periods.append((mask.index[start], mask.index[-1], len(mask)-start))
            
        return periods
    
    def print_periods(self, periods):
        """Print formatted period information"""
        if not periods:
            print("  None found")
        else:
            for start, end, length in periods:
                hours = length / 2  # Convert 30-min periods to hours
                print(f"  {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} ({hours:.1f}h)")
    
    def seasonal_analysis(self):
        """Analyse seasonal utilisation patterns"""
        print(f"\n" + "="*60)
        print("SEASONAL Utilisation PATTERNS")
        print("="*60)
        
        # Add season column (Australian seasons)
        def get_aus_season(month):
            if month in [12, 1, 2]: return 'Summer'
            elif month in [3, 4, 5]: return 'Autumn'  
            elif month in [6, 7, 8]: return 'Winter'
            else: return 'Spring'
        
        seasons = self.soc_pct.index.month.map(get_aus_season)
        
        for unit in self.soc_pct.columns:
            print(f"\n--- {unit} ---")
            seasonal_stats = self.soc_pct[unit].groupby(seasons).agg(['mean', 'max', 'min', 'std'])
            
            # Calculate seasonal energy utilisation
            seasonal_energy = self.storage_energy[unit].groupby(seasons)
            seasonal_range = seasonal_energy.max() - seasonal_energy.min()
            seasonal_utilisation = (seasonal_range / self.storage_capacity[unit]) * 100
            
            # Calculate seasonal activity levels
            seasonal_power = abs(self.storage_power[unit]).groupby(seasons)
            seasonal_activity = seasonal_power.mean() / self.storage_p_nom[unit] * 100
            
            print("Seasonal Analysis:")
            for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
                if season in seasonal_stats.index:
                    avg_soc = seasonal_stats.loc[season, 'mean']
                    max_soc = seasonal_stats.loc[season, 'max'] 
                    min_soc = seasonal_stats.loc[season, 'min']
                    utilisation = seasonal_utilisation.get(season, 0)
                    activity = seasonal_activity.get(season, 0)
                    print(f"  {season:8s}: SOC {min_soc:4.1f}-{max_soc:4.1f}% (avg {avg_soc:4.1f}%), "
                          f"Range {utilisation:4.1f}%, Activity {activity:4.1f}%")
    
    def identify_oversized_storage(self, soc_threshold=80, power_threshold=80):
        """Identify potentially oversized storage units"""
        print(f"\n" + "="*60)
        print(f"POTENTIALLY OVERSIZED STORAGE")
        print(f"(SOC never >={soc_threshold}% OR Power never >={power_threshold}%)")
        print("="*60)
        
        oversized = []
        
        for unit in self.soc_pct.columns:
            max_soc = self.soc_pct[unit].max()
            max_power_pct = (abs(self.storage_power[unit]).max() / self.storage_p_nom[unit]) * 100
            energy_range = self.storage_energy[unit].max() - self.storage_energy[unit].min()
            energy_utilisation = (energy_range / self.storage_capacity[unit]) * 100
            
            # Check if either energy or power capacity is underutilized
            energy_oversized = max_soc < soc_threshold
            power_oversized = max_power_pct < power_threshold
            
            if energy_oversized or power_oversized:
                oversized.append(unit)
                
                print(f"{unit}:")
                print(f"  Current: {self.storage_p_nom[unit]:.0f} MW / {self.storage_capacity[unit]:.1f} MWh")
                print(f"  Peak SOC reached: {max_soc:.1f}%")
                print(f"  Peak Power used: {max_power_pct:.1f}%")
                print(f"  Energy range used: {energy_utilisation:.1f}%")
                
                if energy_oversized:
                    suggested_energy = self.storage_capacity[unit] * (max_soc + 10) / 100  # Add 10% buffer
                    suggested_hours = suggested_energy / self.storage_p_nom[unit]
                    print(f"  Suggested energy: {suggested_energy:.1f} MWh ({suggested_hours:.2f} hours)")
                
                if power_oversized:
                    suggested_power = self.storage_p_nom[unit] * (max_power_pct + 10) / 100  # Add 10% buffer
                    print(f"  Suggested power: {suggested_power:.0f} MW")
                    
                print()
        
        if not oversized:
            print("No obviously oversized storage units identified with current thresholds.")
            
        return oversized
    
    def low_renewable_periods_analysis(self):
        """Analyse storage behavior during low renewable periods"""
        print(f"\n" + "="*60)
        print("LOW RENEWABLE PERIOD ANALYSIS")
        print("="*60)
        
        # This requires renewable generation data - check if available
        renewable_gens = []
        if hasattr(self.n, 'generators_t') and hasattr(self.n.generators_t, 'p'):
            # Look for solar/wind generators (adjust carrier names as needed)
            renewable_carriers = ['solar', 'wind', 'Solar', 'Wind', 'PV', 'onwind', 'offwind']
            for carrier in renewable_carriers:
                gens = self.n.generators[self.n.generators.carrier.str.contains(carrier, case=False, na=False)]
                renewable_gens.extend(gens.index.tolist())
        
        if renewable_gens:
            print(f"Found {len(renewable_gens)} renewable generators")
            
            # Calculate total renewable output by region/time
            renewable_output = self.n.generators_t.p[renewable_gens].sum(axis=1)
            
            # Find low renewable periods (bottom 10%)
            low_threshold = renewable_output.quantile(0.1)
            low_renewable_mask = renewable_output < low_threshold
            
            print(f"Low renewable threshold: {low_threshold:.1f} MW")
            print(f"Low renewable periods: {low_renewable_mask.sum()} out of {len(renewable_output)} ({low_renewable_mask.sum()/len(renewable_output)*100:.1f}%)")
            
            # Analyse storage behavior during these periods
            for unit in self.storage_soc.columns:
                low_period_soc = self.soc_pct[unit][low_renewable_mask]
                normal_period_soc = self.soc_pct[unit][~low_renewable_mask]
                
                print(f"\n{unit} during low renewable periods:")
                print(f"  Average SOC: {low_period_soc.mean():.1f}% (vs {normal_period_soc.mean():.1f}% normally)")
                print(f"  Min SOC: {low_period_soc.min():.1f}% (vs {normal_period_soc.min():.1f}% normally)")
                print(f"  Times empty (<5%): {(low_period_soc < 5).sum()} periods")
        else:
            print("No renewable generators found - cannot analyse low renewable periods")
            print("Check generator carrier names in n.generators.carrier")
    
    def plot_utilisation_summary(self):
        """Create summary plots of storage utilisation"""
        n_units = len(self.soc_pct.columns)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. SOC Time Series
        ax1 = axes[0, 0]
        for unit in self.soc_pct.columns:
            ax1.plot(self.soc_pct.index, self.soc_pct[unit], label=unit.replace('-BATTERY', ''), alpha=0.8)
        ax1.set_ylabel('State of Charge (%)')
        ax1.set_title('Storage SOC Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 2. Utilisation Summary Bar Chart
        ax2 = axes[0, 1]
        max_socs = [self.soc_pct[unit].max() for unit in self.soc_pct.columns]
        avg_socs = [self.soc_pct[unit].mean() for unit in self.soc_pct.columns]
        unit_labels = [unit.replace('-BATTERY', '') for unit in self.soc_pct.columns]
        
        x = range(len(self.soc_pct.columns))
        ax2.bar([i-0.2 for i in x], max_socs, width=0.4, label='Peak SOC', alpha=0.7, color='orange')
        ax2.bar([i+0.2 for i in x], avg_socs, width=0.4, label='Average SOC', alpha=0.7, color='blue')
        ax2.set_ylabel('SOC (%)')
        ax2.set_title('Storage Utilisation Summary')
        ax2.set_xticks(x)
        ax2.set_xticklabels(unit_labels, rotation=0)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 3. Power Utilisation
        ax3 = axes[1, 0]
        max_powers = [abs(self.storage_power[unit]).max() for unit in self.storage_power.columns]
        capacities = [self.storage_p_nom[unit] for unit in self.storage_power.columns]
        power_utils = [(mp/cap)*100 for mp, cap in zip(max_powers, capacities)]
        
        bars = ax3.bar(unit_labels, power_utils, alpha=0.7, color='green')
        ax3.set_ylabel('Power Utilisation (%)')
        ax3.set_title('Maximum Power Utilisation')
        ax3.set_xticklabels(unit_labels, rotation=0)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, util in zip(bars, power_utils):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{util:.1f}%', ha='center', va='bottom')
        
        # 4. Daily SOC Range Distribution
        ax4 = axes[1, 1]
        daily_soc = self.soc_pct.resample('D')
        daily_ranges = daily_soc.max() - daily_soc.min()
        
        for unit in daily_ranges.columns:
            ax4.hist(daily_ranges[unit], bins=20, alpha=0.5, 
                    label=unit.replace('-BATTERY', ''))
        ax4.set_xlabel('Daily SOC Range (%)')
        ax4.set_ylabel('Number of Days')
        ax4.set_title('Daily SOC Range Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage functions
def analyse_storage_utilisation(network):
    """
    Main function to run complete storage utilisation analysis
    
    Args:
        network: PyPSA network object with solved storage_units_t data
    """
    
    analyser = StorageUtilisationAnalyser(network)
    
    # Run all analyses
    utilisation_results = analyser.analyse_underutilisation()
    analyser.find_underutilisation_periods(min_duration_hours=48)  # 2+ day periods
    analyser.seasonal_analysis()
    oversized = analyser.identify_oversized_storage(soc_threshold=75, power_threshold=75)
    analyser.low_renewable_periods_analysis()
    
    # Create plots
    analyser.plot_utilisation_summary()
    
    return utilisation_results, oversized

# Quick summary function
def storage_utilisation_summary(network):
    """Quick summary of storage utilisation"""
    analyser = StorageUtilisationAnalyser(network)
    results = analyser.analyse_underutilisation()
    return results

# Run the analysis:
results, oversized_storage = analyse_storage_utilisation(n)
# or for quick summary:
# summary = storage_utilisation_summary(n)

# OPTIONAL adjustment - toggle to more conservative SOC
# artificially incentivise staying charged
# Update storage units' marginal costs directly in the network object
n.storage_units["marginal_cost"] = -0.01
n.storage_units["marginal_cost_storage"] = -0.01
n.storage_units

n.optimize(solver='gurobi')

# Run the analysis again with updated network:
results, oversized_storage = analyse_storage_utilisation(n)