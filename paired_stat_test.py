import pandas as pd
import numpy as np
from scipy import stats

def read_table(file_path, expected_cols=15):
    """Reads the data file and structures it into a dictionary."""
    with open(file_path, 'r') as f:
        header = f.readline().strip().split()
        data = []

        for line in f:
            values = line.strip().split()
            values = values[:expected_cols]  # Trim extra columns if more than expected
            values += [np.nan] * (expected_cols - len(values))  # Fill missing columns with NaN
            data.append(values)

    if len(header) < expected_cols:
        header += [f"extra_col_{i}" for i in range(len(header), expected_cols)]

    df = pd.DataFrame(data, columns=header)

    required_cols = {'Casename', 'Study', 'stat_para'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Replace '0' with NaN in specific fields
    fields_to_set_nan = {"stat_yel", "stat_wht", "stat_red", "stat_rupt", "stat_bleb.0", "stat_bleb.1"}
    for field in fields_to_set_nan:
        if field in df.columns:
            df[field] = df[field].replace("0.00", np.nan)

    # Pivot table
    structured_data = {}

    for case in df['Casename'].unique():
        case_data = df[df['Casename'] == case]
        msa1 = case_data[case_data['Study'] == 'msa.1'].set_index('stat_para')
        msa2 = case_data[case_data['Study'] == 'msa.2'].set_index('stat_para')

        structured_data[case] = {
            'msa.1': msa1.drop(columns=['Casename', 'Study'], errors='ignore').to_dict(orient='index'),
            'msa.2': msa2.drop(columns=['Casename', 'Study'], errors='ignore').to_dict(orient='index')
        }

    return structured_data


def perform_stat_tests(msa1, msa2, label):
    """Performs normality check and paired test, returns results."""
    if len(msa1) == 0 or len(msa2) == 0:
        return {
            'Comparison': label,
            'Num Cases': len(msa1),
            'Shapiro-Wilk test p-value': None,
            'Test used': None,
            'Comparison test p-value': None
        }

    # Compute differences
    data_diff = msa2 - msa1

    # Interpret significance
    alpha = 0.05  # Common significance level

    # Check normality (only if at least 3 samples)
    shapiro_p = stats.shapiro(data_diff).pvalue if len(data_diff) >= 3 else np.nan

    # Choose appropriate test
    if shapiro_p > 0.05:
        test_stat, p_value = stats.ttest_rel(msa2, msa1)  # Paired t-test
        test_name = "Paired t-test"
    else:
        test_stat, p_value = stats.wilcoxon(msa2, msa1) if len(data_diff) >= 2 else (None, np.nan)
        test_name = "Wilcoxon signed-rank test"

    if p_value < alpha:
        test_result = "Significant difference"    
    else:
        test_result = "No significant difference"

    return {
        'Comparison': label,
        'Num Cases': len(msa1),
        'Shapiro-Wilk test p-value': shapiro_p,
        'Test used': test_name,
        'Comparison test p-value': p_value,
        'Result of comparison': test_result
    }


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
