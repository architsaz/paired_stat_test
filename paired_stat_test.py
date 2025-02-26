import pandas as pd
import numpy as np
from utils.mystat import perform_stat_tests
from utils.mytable import read_table

# Define the fields to analyze
fields_to_analyze = ["stat_aneu", "stat_dom", "stat_bod", "stat_nek", "stat_part", "stat_press","stat_bleb.0","stat_bleb.1", "stat_red", "stat_yel", "stat_wht", "stat_rupt"]

# Example usage
file_path = "data/combin_von_mises.txt"
data = read_table(file_path)

# Store results
all_results = []

for field in fields_to_analyze:
    msa1_means, msa2_means = [], []
    msa1_maxs, msa2_maxs = [], []

    for case_name, case_data in data.items():
        # Extract mean and max values, convert 'None' to NaN
        msa1_mean = case_data['msa.1'].get("mean", {}).get(field, np.nan)
        msa2_mean = case_data['msa.2'].get("mean", {}).get(field, np.nan)
        msa1_max = case_data['msa.1'].get("max", {}).get(field, np.nan)
        msa2_max = case_data['msa.2'].get("max", {}).get(field, np.nan)

        msa1_mean = float(msa1_mean) if isinstance(msa1_mean, (int, float, str)) else None
        msa2_mean = float(msa2_mean) if isinstance(msa2_mean, (int, float, str)) else None

        msa1_mean = pd.to_numeric(msa1_mean, errors='coerce')
        msa2_mean = pd.to_numeric(msa2_mean, errors='coerce')

        msa1_max = float(msa1_max) if isinstance(msa1_max, (int, float, str)) else None
        msa2_max = float(msa2_max) if isinstance(msa2_max, (int, float, str)) else None

        msa1_max = pd.to_numeric(msa1_max, errors='coerce')
        msa2_max = pd.to_numeric(msa2_max, errors='coerce')
       
        # Collect non-NaN values
        if not np.isnan(msa1_mean) and not np.isnan(msa2_mean):
            msa1_means.append(float(msa1_mean))
            msa2_means.append(float(msa2_mean))

        if not np.isnan(msa1_max) and not np.isnan(msa2_max):
            msa1_maxs.append(float(msa1_max))
            msa2_maxs.append(float(msa2_max))

    # Convert to numpy arrays
    msa1_means = np.array(msa1_means, dtype=np.float64)
    msa2_means = np.array(msa2_means, dtype=np.float64)
    msa1_maxs = np.array(msa1_maxs, dtype=np.float64)
    msa2_maxs = np.array(msa2_maxs, dtype=np.float64)

    # Perform statistical tests
    mean_results = perform_stat_tests(msa1_means, msa2_means, f'{field} mean')
    max_results = perform_stat_tests(msa1_maxs, msa2_maxs, f'{field} max')

    all_results.append(mean_results)
    all_results.append(max_results)

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv('statistical_comparison_results.csv', index=False)

# Print results
print(results_df)
