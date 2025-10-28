import pandas as pd
import glob
import os

# Folder where your CSV files are located
folder_path = r'E:\saas-idea-validator\data\raw\raw_batch'
store_path = r'E:\saas-idea-validator\data\raw\raw_marged'  #  Change this to your folder path

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# List to hold data from each CSV
dataframes = []

# Loop through each file and read it
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Merge all CSVs into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Save merged file
output_file = os.path.join(store_path, "merged_output.csv")
merged_df.to_csv(output_file, index=False)

print(f"Merge Successfully")
