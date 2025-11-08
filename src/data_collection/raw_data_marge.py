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

# Loop through each file and read itimport os
import pandas as pd
import logging

def merge_raw_batches(batch_dir="data/raw/raw_batch", output_dir="data/raw/raw_merged", 
                      remove_duplicates=True, delete_source_batches=False):
    os.makedirs(output_dir, exist_ok=True)
    
    all_dfs = []
    batch_count = 0
    
    for file in os.listdir(batch_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(batch_dir, file))
            all_dfs.append(df)
            batch_count += 1
    
    if not all_dfs:
        return {'success': False, 'merged_file': None, 'row_count': 0, 'batch_count': 0, 'errors': ['No batch files found']}
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    if remove_duplicates and 'post_id' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['post_id'], keep='first')
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"merged_{timestamp}.csv")
    merged_df.to_csv(output_file, index=False)
    
    return {
        'success': True,
        'merged_file': output_file,
        'row_count': len(merged_df),
        'batch_count': batch_count,
        'errors': []
    }

def get_latest_merged_file(output_dir="data/raw/raw_merged"):
    if not os.path.exists(output_dir):
        return None
    
    files = [f for f in os.listdir(output_dir) if f.startswith('merged_') and f.endswith('.csv')]
    if not files:
        return None
    
    return os.path.join(output_dir, sorted(files)[-1])

def get_merge_statistics(merged_file):
    if not os.path.exists(merged_file):
        return {}
    
    df = pd.read_csv(merged_file)
    return {
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
        'missing_values': df.isnull().sum().to_dict()
    }
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Merge all CSVs into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Remove duplicates by post_id, keep first occurrence
merged_df = merged_df.drop_duplicates(subset=['post_id'], keep='first')
# Save merged file
output_file = os.path.join(store_path, "merged_output.csv")
merged_df.to_csv(output_file, index=False)

print(f"Merge Successfully with {len(merged_df)} unique posts")
