import pandas as pd
import numpy as np

def read_table(file_path,data_name, expected_cols=16):
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
    fields_to_set_nan = {"yel_"+data_name, "wht_"+data_name, "red_"+data_name, "rupt_"+data_name, "bleb.0_"+data_name, "bleb.1_"+data_name, "all_bleb_"+data_name}
    # Group by 'Casename' and 'Study' to check field values across groups
    for field in fields_to_set_nan:
        if field in df.columns:
            # Ensure the field column is numeric
            df[field] = pd.to_numeric(df[field], errors='coerce')
            # Find rows where the values are close to 0.00
            mask = df.groupby(["Casename", "Study"])[field].transform(lambda x: np.isclose(x, 0.00).all())
            df.loc[mask, field] = np.nan


    print(df.head())
    df.to_csv("modified_data.csv", index=False)        

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