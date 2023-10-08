import pandas as pd

# Input and output file paths
dscovr_file = 'dscovr-new.csv'
kp_file = 'kp-ap.csv'
output_kp_file = 'output_kp.csv'
output_dscovr_file = 'output_dscovr.csv'

# Read dscovr and kp CSV files into DataFrames
dscovr_df = pd.read_csv(dscovr_file)
kp_df = pd.read_csv(kp_file)

# Initialize a list to store unique [year, day] combinations
dates = []

# Process dscovr DataFrame
for index, row in dscovr_df.iterrows():
    cell_value = row[7]
    day = row[1]
    year = row[0]

    if cell_value == 99999.9 or row[4] == 9999.99:
        if [year, day] not in dates:
            dates.append([year, day])

# Filter dscovr DataFrame based on dates
filtered_dscovr_df = dscovr_df[~dscovr_df.apply(lambda row: [row[0], row[1]] in dates, axis=1)]

# Filter kp DataFrame based on dates
filtered_kp_df = kp_df[~kp_df.apply(lambda row: [row[0], row[3]] in dates, axis=1)]

# Save the filtered DataFrames to output CSV files
filtered_dscovr_df.to_csv(output_dscovr_file, index=False)
filtered_kp_df.to_csv(output_kp_file, index=False)

print("Rows with values less than 999.999 have been written to output files.")
