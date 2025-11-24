"""
Data Balancing Pipeline - Remove imbalanced class and rebalance remaining classes

Steps:
1. Load dataset
2. Drop Class 0 (lowest data)
3. Balance Classes 1 & 2 using pandas sampling (NO imblearn required)
4. Save balanced dataset
5. Show statistics
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
DATASET_PATH = r"E:\saas-idea-validator\data\processed\vectorized_features.csv"
OUTPUT_DIR = Path(r"E:\saas-idea-validator\data\processed\balanced")
OUTPUT_FILE = OUTPUT_DIR / "vectorized_features_balanced.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD & ANALYZE ORIGINAL DATA ==========
def load_and_analyze_original():
    """Load dataset and show class distribution"""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading Original Dataset")
    logger.info("="*80)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"✓ Loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Analyze class distribution
    logger.info(f"\n📊 Original Class Distribution:")
    class_counts = df['label_numeric'].value_counts().sort_index()
    
    for class_id in class_counts.index:
        count = class_counts[class_id]
        pct = count / len(df) * 100
        class_name = {0: 'Bad', 1: 'Neutral', 2: 'Good'}.get(class_id, 'Unknown')
        logger.info(f"   Class {class_id} ({class_name}): {count:5d} samples ({pct:5.1f}%)")
    
    logger.info(f"   Total: {len(df)} samples")
    
    return df, class_counts


# ========== DROP CLASS 0 ==========
def drop_class_0(df):
    """Remove Class 0 (lowest data)"""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Dropping Class 0 (Bad - Lowest Data)")
    logger.info("="*80)
    
    df_filtered = df[df['label_numeric'] != 0].copy()
    
    logger.info(f"✓ Dropped Class 0")
    logger.info(f"   Samples removed: {len(df) - len(df_filtered)}")
    logger.info(f"   Samples remaining: {df_filtered.shape[0]} ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Relabel classes: 1→0, 2→1
    df_filtered['label_numeric'] = df_filtered['label_numeric'] - 1
    
    logger.info(f"\n📊 After dropping Class 0 (Relabeled):")
    class_counts = df_filtered['label_numeric'].value_counts().sort_index()
    
    for class_id in class_counts.index:
        count = class_counts[class_id]
        pct = count / len(df_filtered) * 100
        class_name = {0: 'Neutral', 1: 'Good'}.get(class_id, 'Unknown')
        logger.info(f"   Class {class_id} ({class_name}): {count:5d} samples ({pct:5.1f}%)")
    
    return df_filtered, class_counts


# ========== BALANCE REMAINING CLASSES ==========
def balance_classes(df):
    """Balance Classes 0 & 1 using pandas sampling (NO imblearn needed)"""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Balancing Classes 0 & 1 (Neutral & Good)")
    logger.info("="*80)
    
    # Get class counts before balancing
    original_counts = df['label_numeric'].value_counts().sort_index()
    min_count = original_counts.min()
    max_count = original_counts.max()
    
    logger.info(f"   Min class count: {min_count}")
    logger.info(f"   Max class count: {max_count}")
    logger.info(f"   Target balance: {min_count} samples per class")
    
    # Separate by class
    class_0 = df[df['label_numeric'] == 0]
    class_1 = df[df['label_numeric'] == 1]
    
    logger.info(f"\n   Class 0 (Neutral): {len(class_0)} samples")
    logger.info(f"   Class 1 (Good):    {len(class_1)} samples")
    
    # Undersample majority to match minority
    if len(class_0) > len(class_1):
        logger.info(f"   → Undersampling Class 0 to {len(class_1)} samples")
        class_0_balanced = class_0.sample(n=len(class_1), random_state=42)
        class_1_balanced = class_1
    else:
        logger.info(f"   → Undersampling Class 1 to {len(class_0)} samples")
        class_0_balanced = class_0
        class_1_balanced = class_1.sample(n=len(class_0), random_state=42)
    
    # Combine balanced classes
    df_balanced = pd.concat([class_0_balanced, class_1_balanced], ignore_index=True)
    
    # Show results
    logger.info(f"\n✓ Balancing complete!")
    logger.info(f"   Original total: {len(df)} samples")
    logger.info(f"   Balanced total: {len(df_balanced)} samples")
    logger.info(f"   Removed: {len(df) - len(df_balanced)} samples")
    
    logger.info(f"\n📊 After Balancing:")
    balanced_counts = df_balanced['label_numeric'].value_counts().sort_index()
    
    for class_id in balanced_counts.index:
        count = balanced_counts[class_id]
        pct = count / len(df_balanced) * 100
        class_name = {0: 'Neutral', 1: 'Good'}.get(class_id, 'Unknown')
        logger.info(f"   Class {class_id} ({class_name}): {count:5d} samples ({pct:5.1f}%)")
    
    return df_balanced


# ========== SHUFFLE AND SAVE ==========
def shuffle_and_save(df_balanced):
    """Shuffle dataset and save to file"""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Shuffling and Saving")
    logger.info("="*80)
    
    # Shuffle
    df_shuffled = shuffle(df_balanced, random_state=42)
    
    # Reset index
    df_shuffled = df_shuffled.reset_index(drop=True)
    
    # Save
    df_shuffled.to_csv(OUTPUT_FILE, index=False)
    
    logger.info(f"✓ Saved balanced dataset")
    logger.info(f"   Path: {OUTPUT_FILE}")
    logger.info(f"   Shape: {df_shuffled.shape[0]} rows × {df_shuffled.shape[1]} columns")
    
    return df_shuffled


# ========== GENERATE COMPARISON REPORT ==========
def generate_report(df_original, df_balanced):
    """Generate detailed comparison report"""
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Summary Report")
    logger.info("="*80)
    
    print(f"\n{'DATASET BALANCING SUMMARY':<50}")
    print("="*80)
    
    print(f"\n📊 ORIGINAL DATASET:")
    print(f"   Total samples: {len(df_original)}")
    original_dist = df_original['label_numeric'].value_counts().sort_index()
    for class_id in original_dist.index:
        pct = original_dist[class_id] / len(df_original) * 100
        class_map = {0: 'Bad', 1: 'Neutral', 2: 'Good'}
        print(f"   Class {class_id} ({class_map.get(class_id)}): {original_dist[class_id]:5d} ({pct:5.1f}%)")
    
    print(f"\n⚠️  PROBLEMS IDENTIFIED:")
    print(f"   ✗ Class 0 (Bad): Severely underrepresented (~5%)")
    print(f"   ✗ Class imbalance causes model bias")
    print(f"   ✗ Overfitting to majority classes")
    print(f"   ✗ Poor performance on minority class")
    
    print(f"\n✅ ACTIONS TAKEN:")
    print(f"   1. Dropped Class 0 (Bad) - insufficient data")
    print(f"   2. Kept Classes 1 & 2 (Neutral & Good)")
    print(f"   3. Undersampled majority to match minority")
    print(f"   4. Randomly shuffled dataset")
    
    print(f"\n📊 BALANCED DATASET:")
    print(f"   Total samples: {len(df_balanced)}")
    balanced_dist = df_balanced['label_numeric'].value_counts().sort_index()
    for class_id in balanced_dist.index:
        pct = balanced_dist[class_id] / len(df_balanced) * 100
        class_map = {0: 'Neutral', 1: 'Good'}
        print(f"   Class {class_id} ({class_map.get(class_id)}): {balanced_dist[class_id]:5d} ({pct:5.1f}%)")
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"   ✓ Perfect class balance: 50% / 50%")
    print(f"   ✓ No single class dominance")
    print(f"   ✓ Reduced overfitting risk")
    print(f"   ✓ Better generalization expected")
    print(f"   ✓ More robust training signals")
    
    print(f"\n💾 OUTPUT FILE:")
    print(f"   Location: {OUTPUT_FILE}")
    print(f"   Samples:  {len(df_balanced):,}")
    print(f"   Features: {df_balanced.shape[1]}")
    print(f"   Size:     {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
    
    print("\n" + "="*80)
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Update data_loader.py to use balanced dataset:")
    print(f"      DATASET_PATH = r\"{OUTPUT_FILE}\"")
    print(f"   2. Re-run main.py to train on balanced data:")
    print(f"      python main.py")
    print(f"   3. Compare results with original training")
    print("="*80 + "\n")


# ========== MAIN PIPELINE ==========
def main():
    print("\n" + "="*80)
    print("🎯 DATA BALANCING PIPELINE")
    print("   Drop Class 0 → Balance Classes 1 & 2")
    print("="*80)
    
    try:
        # Step 1: Load and analyze
        df_original, original_counts = load_and_analyze_original()
        
        # Step 2: Drop class 0
        df_dropped, dropped_counts = drop_class_0(df_original)
        
        # Step 3: Balance remaining classes
        df_balanced = balance_classes(df_dropped)
        
        # Step 4: Shuffle and save
        df_final = shuffle_and_save(df_balanced)
        
        # Step 5: Report
        generate_report(df_original, df_final)
        
        logger.info("✅ DATA BALANCING COMPLETE!")
        logger.info(f"✓ Balanced dataset ready at: {OUTPUT_FILE}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())