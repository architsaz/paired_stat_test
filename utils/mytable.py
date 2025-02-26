import pandas as pd
import numpy as np

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