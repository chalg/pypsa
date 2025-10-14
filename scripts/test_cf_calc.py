import pandas as pd
import numpy as np

# Load the test data
dispatchload_joined = pd.read_csv('data/test_data.csv')

# Method 1: Nested np.where (most efficient)
is_solar = dispatchload_joined['fuel_source_descriptor'] == 'Solar'

denominator = np.where(
    is_solar, 
    dispatchload_joined['max_cap_generation_mw'],
    dispatchload_joined['reg_cap_generation_mw']
)

dispatchload_joined['cf'] = np.where(
    dispatchload_joined['reg_cap_generation_mw'],
    dispatchload_joined['initialmw'] / denominator,
    None
)

dispatchload_joined