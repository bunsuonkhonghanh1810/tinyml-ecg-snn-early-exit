import os
import sys
import argparse
import time
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.ptbxl_raw_loader import (
    CLASS_NAMES,
    load_ptbxl_metadata,
    read_record_signal,
    get_lead_signal,
)
from src.dataset.preprocessing import preprocess_ecg_lead

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(
            "src", "data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("src", "data", "processed"),
    )
    parser.add_argument("--lead-idx", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()

def process_split(
    df_split,
    dataset_dir: str,
    lead_idx: int = 0,
    window_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    
    total = len(df_split)
    start_time = time.time()
    
    for idx, (_, row) in enumerate(df_split.iterrows()):
        relative_path = row["filename_lr"]
        dat_path = os.path.join(dataset_dir, relative_path + ".dat")
        label = int(row["label_idx"])
        
        if not os.path.exists(dat_path):
            continue
            
        signal_12lead = read_record_signal(dat_path)
        lead_signal = get_lead_signal(signal_12lead, lead_idx=lead_idx)
        
        beats, _ = preprocess_ecg_lead(lead_signal, fs=100.0, window_size=window_size)
        
        for beat in beats:
            all_beats.append(beat)
            all_labels.append(label)
            
        if (idx + 1) % 2500 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            print(f"    Processed {idx + 1}/{total} records ({elapsed:.1f}s) - Accumulated beats: {len(all_beats)}")
            
    if len(all_beats) == 0:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
    x_array = np.array(all_beats, dtype=np.float32)
    y_array = np.array(all_labels, dtype=np.int64)
    return x_array, y_array

def print_class_distribution(y_array: np.ndarray, split_name: str):
    print(f"\n--- {split_name} Class Distribution (Total beats: {len(y_array)}) ---")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_array == cls_idx))
        pct = (count / len(y_array) * 100.0) if len(y_array) > 0 else 0.0
        print(f"  [{cls_idx}] {cls_name:<6}: {count:>7} beats ({pct:>5.1f}%)")

def main():
    args = parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.abspath(os.path.join(base_dir, args.dataset_dir))
    output_dir = os.path.abspath(os.path.join(base_dir, args.output_dir))
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== STARTING ECG DATASET PREPROCESSING ===")
    print(f"Source raw dataset  : {dataset_dir}")
    print(f"Processed output dir: {output_dir}")
    print(f"Target Lead         : Lead I (Index {args.lead_idx})")
    print(f"Heartbeat window    : {args.window_size} samples (1.0s at 100Hz)\n")
    
    metadata_df = load_ptbxl_metadata(dataset_dir)
    valid_df = metadata_df[metadata_df["label_idx"] >= 0].copy()
    
    if args.max_records is not None:
        valid_df = valid_df.iloc[:args.max_records]
        print(f"Subsetting to first {args.max_records} records.")
        
    print(f"Total valid diagnostic records to process: {len(valid_df)}")
    
    train_df = valid_df[valid_df["strat_fold"] <= 8]
    val_df = valid_df[valid_df["strat_fold"] == 9]
    test_df = valid_df[valid_df["strat_fold"] == 10]
    
    print(f"Records split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}\n")
    
    print("[1/3] Processing TRAIN set (Folds 1-8)...")
    x_train, y_train = process_split(train_df, dataset_dir, args.lead_idx, args.window_size)
    print_class_distribution(y_train, "TRAIN")
    train_out_path = os.path.join(output_dir, "train.npz")
    np.savez_compressed(train_out_path, x=x_train, y=y_train)
    print(f"Saved: {train_out_path} ({os.path.getsize(train_out_path) / (1024*1024):.2f} MB)")
    
    print("\n[2/3] Processing VALIDATION set (Fold 9)...")
    x_val, y_val = process_split(val_df, dataset_dir, args.lead_idx, args.window_size)
    print_class_distribution(y_val, "VALIDATION")
    val_out_path = os.path.join(output_dir, "val.npz")
    np.savez_compressed(val_out_path, x=x_val, y=y_val)
    print(f"Saved: {val_out_path} ({os.path.getsize(val_out_path) / (1024*1024):.2f} MB)")
    
    print("\n[3/3] Processing TEST set (Fold 10)...")
    x_test, y_test = process_split(test_df, dataset_dir, args.lead_idx, args.window_size)
    print_class_distribution(y_test, "TEST")
    test_out_path = os.path.join(output_dir, "test.npz")
    np.savez_compressed(test_out_path, x=x_test, y=y_test)
    print(f"Saved: {test_out_path} ({os.path.getsize(test_out_path) / (1024*1024):.2f} MB)")
    
    total_beats = len(y_train) + len(y_val) + len(y_test)
    print(f"\n=== PREPROCESSING COMPLETED ===")
    print(f"Total extracted heartbeats: {total_beats}")
    print(f"Output files stored in    : {output_dir}")

if __name__ == "__main__":
    main()
