import os
import pandas as pd
import shutil

# Configurations
report_csv = r'E:\saas-idea-validator\reports\model_comparison.csv'  # Update to your path
models_folder = r'E:\saas-idea-validator\models'                    # Folder containing your .joblib model files
best_model_dest = r'E:\saas-idea-validator\best_sl_model'              # Folder to copy the best model to

# Read the model performance report
df = pd.read_csv(report_csv)

# Select the best model by accuracy
best_model_row = df.loc[df['accuracy'].idxmax()]
best_model_name = best_model_row['model_name']

# Append .joblib extension, if missing
if not best_model_name.endswith('.joblib'):
    best_model_name += '.joblib'

# Full source and destination paths
source_path = os.path.join(models_folder, best_model_name)
destination_path = os.path.join(best_model_dest, best_model_name)

# Create destination folder if it doesn't exist
os.makedirs(best_model_dest, exist_ok=True)

# Copy the best model file
try:
    shutil.copy(source_path, destination_path)
    print(f"Best model '{best_model_name}' copied to '{best_model_dest}'")
except FileNotFoundError:
    print(f"Model file not found: {source_path}. Please check the model file and path.")
