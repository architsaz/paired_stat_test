import pandas as pd
import numpy as np
from utils.mystat import perform_stat_tests
from utils.mytable import read_table

# Example usage
#data_name = "eval_ratio"
#data_name = "von_mises"
data_name = "eigen_class"
path = "data/"
file_name = "combin_"+data_name+".txt"
stat_output = "statistical_comparison_"+data_name+".csv"
file_path = path+file_name
data = read_table(file_path,data_name)

# Define the fields to analyze
regions_to_analyze = ["aneu_"+data_name, "dom_"+data_name, "bod_"+data_name, "nek_"+data_name, "part_"+data_name, "press_"+data_name,"all_bleb_"+data_name, "red_"+data_name, "yel_"+data_name, "wht_"+data_name, "rupt_"+data_name]

# Store results
all_results = []

# Define additional indices to analyze
#indices_to_analyze = ['mean', 'max', 'min']  # Add your desired indices
indices_to_analyze = ['1', '2', '3' , '4', '5', '6']  # Add your desired indices

for region in regions_to_analyze:
    stats_data = {index: {'msa1': [], 'msa2': []} for index in indices_to_analyze}

    for case_name, case_data in data.items():
        for index in indices_to_analyze:
            msa1_value = case_data['msa.1'].get(index, {}).get(region, np.nan)
            msa2_value = case_data['msa.2'].get(index, {}).get(region, np.nan)

            msa1_value = float(msa1_value) if isinstance(msa1_value, (int, float, str)) else None
            msa2_value = float(msa2_value) if isinstance(msa2_value, (int, float, str)) else None

            msa1_value = pd.to_numeric(msa1_value, errors='coerce')
            msa2_value = pd.to_numeric(msa2_value, errors='coerce')

            if not np.isnan(msa1_value) and not np.isnan(msa2_value):
                stats_data[index]['msa1'].append(float(msa1_value))
                stats_data[index]['msa2'].append(float(msa2_value))

    # Convert lists to numpy arrays and perform statistical tests
    for index in indices_to_analyze:
        msa1_array = np.array(stats_data[index]['msa1'], dtype=np.float64)
        msa2_array = np.array(stats_data[index]['msa2'], dtype=np.float64)
        result = perform_stat_tests(msa1_array, msa2_array, f'{region} {index}')
        all_results.append(result)

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv(stat_output, index=False)

# Print results
print(results_df)
