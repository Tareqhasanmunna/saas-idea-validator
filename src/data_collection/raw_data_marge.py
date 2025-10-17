import pandas as pd
import glob
import os

# Folder where your CSV files are located
folder_path = 'path/to/your/folder'  # 🔹 Change this to your folder path

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
output_file = os.path.join(folder_path, "merged_output.csv")
merged_df.to_csv(output_file, index=False)

print(f"✅ All CSV files merged successfully!\nSaved as: {output_file}")
# This script merges all CSV files in a specified folder into a single CSV file.

'''import pandas as pd
import glob

# Path where all CSV files are stored
path = "path/to/your/folder"   # 🔹 change this to your folder path

# Get all CSV files in the folder
all_files = glob.glob(path + "/*.csv")

# Read and merge all CSV files
merged_csv = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True)

# Save the merged CSV file
merged_csv.to_csv("merged_file.csv", index=False)

print("✅ All CSV files have been merged into 'merged_file.csv'")
'''