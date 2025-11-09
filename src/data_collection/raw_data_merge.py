import pandas as pd
import glob
import os
import sys

def merge_raw_batches():
    """Merge all CSV files from raw_batch folder"""
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    folder_path = os.path.join(project_root, 'data/raw/raw_batch')
    store_path = os.path.join(project_root, 'data/raw/raw_marged')
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        return {
            'success': False,
            'message': f"No CSV files found in {folder_path}",
            'output_file': None,
            'row_count': 0
        }
    
    print(f"[MERGE] Found {len(csv_files)} batch files")
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"[MERGE] Loaded {os.path.basename(file)}: {len(df)} rows")
        except Exception as e:
            print(f"[MERGE] Error reading {file}: {e}")
            continue
    
    if not dataframes:
        return {
            'success': False,
            'message': "No valid CSV files loaded",
            'output_file': None,
            'row_count': 0
        }
    
    # Merge
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"[MERGE] Concatenated: {len(merged_df)} rows")
    
    # Remove duplicates
    if 'post_id' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['post_id'], keep='first')
        print(f"[MERGE] After dedup: {len(merged_df)} unique posts")
    
    # Save
    os.makedirs(store_path, exist_ok=True)
    output_file = os.path.join(store_path, "merged_output.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"[MERGE] Saved to {output_file}")
    
    return {
        'success': True,
        'message': f"Merged {len(merged_df)} unique posts",
        'output_file': output_file,
        'row_count': len(merged_df)
    }
from src.utils.auto_cleaner import delete_old_files

# After saving merged file:
print(f"[MERGE] Saved to {output_file}")

# ✅ Clean raw_batch folder after merge
import shutil, glob
for f in glob.glob(os.path.join(folder_path, "*")):
    try:
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)
    except Exception as e:
        print(f"[CLEANUP] Couldn't remove {f}: {e}")
print("[CLEANUP] raw_batch folder cleaned after merge.")


if __name__ == "__main__":
    result = merge_raw_batches()
    print(result)