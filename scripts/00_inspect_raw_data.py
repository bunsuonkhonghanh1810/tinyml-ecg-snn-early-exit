import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.ptbxl_raw_loader import (
    CLASS_NAMES,
    load_ptbxl_metadata,
    read_record_signal,
    get_lead_signal,
)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(
        base_dir,
        "src",
        "data",
        "raw",
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    )

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        sys.exit(1)

    print("=== PTB-XL RAW DATASET INSPECTION ===")
    print(f"Dataset path: {dataset_dir}\n")

    metadata_df = load_ptbxl_metadata(dataset_dir)
    total_records = len(metadata_df)
    print(f"Total records: {total_records}")

    print("\n--- Diagnostic Superclasses Distribution ---")
    class_counts = metadata_df["label_name"].value_counts()
    for name, count in class_counts.items():
        pct = (count / total_records) * 100.0
        print(f"  {name:<8}: {count:>6} records ({pct:>5.1f}%)")

    single_label_count = int(metadata_df["is_single_label"].sum())
    multi_label_count = total_records - single_label_count
    print(f"\nSingle-label records : {single_label_count} ({(single_label_count/total_records)*100.1:.1f}%)")
    print(f"Multi-label records  : {multi_label_count} ({(multi_label_count/total_records)*100.1:.1f}%)")

    print("\n--- Stratified Folds Distribution (strat_fold) ---")
    fold_counts = metadata_df["strat_fold"].value_counts().sort_index()
    for fold, count in fold_counts.items():
        role = "Train" if fold <= 8 else ("Val" if fold == 9 else "Test")
        print(f"  Fold {fold:>2}: {count:>5} records [{role}]")

    print("\n--- Sample Waveform Inspection (Record: 00001_lr) ---")
    sample_relative_path = metadata_df.iloc[0]["filename_lr"]
    sample_dat_path = os.path.join(dataset_dir, sample_relative_path + ".dat")
    
    if os.path.exists(sample_dat_path):
        signal_12lead = read_record_signal(sample_dat_path)
        lead1_signal = get_lead_signal(signal_12lead, lead_idx=0)
        
        print(f"File path       : {sample_dat_path}")
        print(f"12-Lead Shape   : {signal_12lead.shape} (Time points x Leads)")
        print(f"Lead I Shape    : {lead1_signal.shape} (1D array)")
        print(f"Lead I Min      : {lead1_signal.min():.4f} mV")
        print(f"Lead I Max      : {lead1_signal.max():.4f} mV")
        print(f"Lead I Mean     : {lead1_signal.mean():.4f} mV")
        print(f"Lead I Std      : {lead1_signal.std():.4f} mV")
        print(f"First 10 values : {np.round(lead1_signal[:10], 4).tolist()}")
        print("\nInspection completed successfully!")
    else:
        print(f"Warning: Sample file {sample_dat_path} not found.")

if __name__ == "__main__":
    main()
