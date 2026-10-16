import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(AAMI_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(AAMI_CLASSES)}

AAMI_MAP = {
    # N: Normal and Bundle Branch Block beats
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    # S: Supraventricular ectopic beats
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    # V: Ventricular ectopic beats (PVC, etc.)
    "V": 2,
    "E": 2,
    # F: Fusion beats
    "F": 3,
    # Q: Unknown / Paced beats
    "/": 4,
    "f": 4,
    "Q": 4,
}

# Standard de Chazal et al. (IEEE TBME 2004) Inter-Patient Benchmark Split
DS1_TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
]

DS2_TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
]

def butter_bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 360.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=-1)

def normalize_zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    return (signal - mean) / (std + eps)

def load_mitbih_record(raw_dir: str, record_id: str) -> Tuple[np.ndarray, str]:
    csv_path = os.path.join(raw_dir, f"{record_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Record CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    clean_cols = [c.replace("'", "").strip() for c in df.columns]
    df.columns = clean_cols

    if "MLII" in df.columns:
        lead_name = "MLII"
    else:
        signal_cols = [c for c in df.columns if c != "sample #"]
        lead_name = signal_cols[0]

    signal = df[lead_name].values.astype(np.float32)
    return signal, lead_name

def load_mitbih_annotations(raw_dir: str, record_id: str) -> List[Tuple[int, int, str]]:
    ann_path = os.path.join(raw_dir, f"{record_id}annotations.txt")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    annotations = []
    with open(ann_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                sample_idx = int(parts[1])
                symbol = parts[2]
                if symbol in AAMI_MAP:
                    class_idx = AAMI_MAP[symbol]
                    annotations.append((sample_idx, class_idx, symbol))
            except ValueError:
                continue

    return annotations

def extract_record_heartbeats(
    signal: np.ndarray,
    annotations: List[Tuple[int, int, str]],
    pre_r: int = 70,
    post_r: int = 110,
    filter_signal: bool = True,
    fs: float = 360.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if filter_signal:
        signal = butter_bandpass_filter(signal, fs=fs)

    window_size = pre_r + post_r
    signal_len = len(signal)
    beats = []
    labels = []
    symbols = []

    for sample_idx, class_idx, symbol in annotations:
        start_idx = sample_idx - pre_r
        end_idx = sample_idx + post_r

        if start_idx >= 0 and end_idx <= signal_len:
            segment = signal[start_idx:end_idx]
            if len(segment) == window_size:
                beats.append(segment)
                labels.append(class_idx)
                symbols.append(symbol)

    if len(beats) == 0:
        return (
            np.empty((0, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    beats_array = np.array(beats, dtype=np.float32)
    beats_normalized = normalize_zscore(beats_array)
    labels_array = np.array(labels, dtype=np.int64)

    return beats_normalized, labels_array, symbols
