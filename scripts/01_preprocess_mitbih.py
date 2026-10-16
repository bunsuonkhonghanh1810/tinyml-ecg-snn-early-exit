import os
import sys
import time
import argparse
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.mitbih_loader import (
    AAMI_CLASSES,
    CLASS_TO_IDX,
    DS1_TRAIN_RECORDS,
    DS2_TEST_RECORDS,
    load_mitbih_record,
    load_mitbih_annotations,
    extract_record_heartbeats,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH Arrhythmia Dataset")
    parser.add_argument("--raw-dir", type=str, default=os.path.join("src", "data", "raw", "mitbih"))
    parser.add_argument("--output-dir", type=str, default=os.path.join("src", "data", "processed"))
    parser.add_argument("--pre-r", type=int, default=70)
    parser.add_argument("--post-r", type=int, default=110)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()

def process_record_list(
    record_ids: List[str],
    raw_dir: str,
    pre_r: int = 70,
    post_r: int = 110,
) -> Tuple[np.ndarray, np.ndarray]:
    all_beats = []
    all_labels = []

    for idx, rec_id in enumerate(record_ids):
        t0 = time.time()
        try:
            signal, lead_name = load_mitbih_record(raw_dir, rec_id)
            annotations = load_mitbih_annotations(raw_dir, rec_id)
            beats, labels, _ = extract_record_heartbeats(
                signal=signal,
                annotations=annotations,
                pre_r=pre_r,
                post_r=post_r,
                filter_signal=True,
                fs=360.0,
            )
            all_beats.append(beats)
            all_labels.append(labels)
            elapsed = time.time() - t0
            print(f"  [{idx+1:02d}/{len(record_ids):02d}] Record {rec_id} ({lead_name:<4}) -> {len(labels):>5} beats ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  Error processing record {rec_id}: {e}")

    if len(all_beats) == 0:
        window_size = pre_r + post_r
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return x, y

def print_distribution(y: np.ndarray, title: str):
    print(f"\n--- {title} Distribution (Total: {len(y):,} beats) ---")
    for idx, name in enumerate(AAMI_CLASSES):
        count = int(np.sum(y == idx))
        pct = (count / len(y) * 100.0) if len(y) > 0 else 0.0
        print(f"  [{idx}] {name:<2}: {count:>7} beats ({pct:>5.1f}%)")

def main():
    args = parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.abspath(os.path.join(base_dir, args.raw_dir))
    output_dir = os.path.abspath(os.path.join(base_dir, args.output_dir))

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"MIT-BIH raw directory not found: {raw_dir}")

    os.makedirs(output_dir, exist_ok=True)
    window_size = args.pre_r + args.post_r

    print("=== MIT-BIH ARRHYTHMIA PREPROCESSING ===")
    print(f"Raw source dir   : {raw_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Heartbeat window : {window_size} samples (pre-R: {args.pre_r}, post-R: {args.post_r}) at 360Hz")
    print(f"Protocol         : de Chazal Inter-Patient Split (DS1: 22 train, DS2: 22 test)\n")

    # 1. Process DS1 (Train/Val pool)
    print("Processing DS1 (Training Records - 22 patients)...")
    x_ds1, y_ds1 = process_record_list(DS1_TRAIN_RECORDS, raw_dir, args.pre_r, args.post_r)
    print_distribution(y_ds1, "DS1 (Train + Val Pool)")

    # 2. Stratified split DS1 into Train and Val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.random_seed)
    train_idx, val_idx = next(sss.split(x_ds1, y_ds1))

    x_train, y_train = x_ds1[train_idx], y_ds1[train_idx]
    x_val, y_val = x_ds1[val_idx], y_ds1[val_idx]

    print_distribution(y_train, "Final TRAIN Set")
    print_distribution(y_val, "Final VALIDATION Set")

    # 3. Process DS2 (Independent Test Set)
    print("\nProcessing DS2 (Test Records - 22 independent patients)...")
    x_test, y_test = process_record_list(DS2_TEST_RECORDS, raw_dir, args.pre_r, args.post_r)
    print_distribution(y_test, "Final TEST Set (DS2)")

    # 4. Save npz files
    train_file = os.path.join(output_dir, "train.npz")
    val_file = os.path.join(output_dir, "val.npz")
    test_file = os.path.join(output_dir, "test.npz")

    np.savez_compressed(train_file, x=x_train, y=y_train)
    np.savez_compressed(val_file, x=x_val, y=y_val)
    np.savez_compressed(test_file, x=x_test, y=y_test)

    print("\n=== PREPROCESSING COMPLETED ===")
    print(f"Train : {train_file} ({len(y_train):,} beats, {os.path.getsize(train_file)/(1024*1024):.2f} MB)")
    print(f"Val   : {val_file} ({len(y_val):,} beats, {os.path.getsize(val_file)/(1024*1024):.2f} MB)")
    print(f"Test  : {test_file} ({len(y_test):,} beats, {os.path.getsize(test_file)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
