from typing import List, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def butter_bandpass(lowcut: float = 0.5, highcut: float = 45.0, fs: float = 100.0, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyquist_freq = 0.5 * fs
    low_normalized = max(lowcut / nyquist_freq, 0.001)
    high_normalized = min(highcut / nyquist_freq, 0.999)
    b_coeff, a_coeff = butter(order, [low_normalized, high_normalized], btype="band")
    return b_coeff, a_coeff

def butter_bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 100.0, order: int = 4) -> np.ndarray:
    b_coeff, a_coeff = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = filtfilt(b_coeff, a_coeff, signal, axis=-1)
    return filtered_signal

def normalize_zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean_val = np.mean(signal, axis=-1, keepdims=True)
    std_val = np.std(signal, axis=-1, keepdims=True)
    normalized_signal = (signal - mean_val) / (std_val + eps)
    return normalized_signal

def detect_r_peaks(signal: np.ndarray, fs: float = 100.0, min_distance_ms: float = 400.0) -> np.ndarray:
    min_distance_samples = int(fs * (min_distance_ms / 1000.0))
    adaptive_threshold = np.mean(signal) + 0.5 * np.std(signal)
    peaks, _ = find_peaks(signal, distance=min_distance_samples, height=adaptive_threshold)
    
    if len(peaks) == 0:
        fallback_threshold = np.mean(signal) + 0.2 * np.std(signal)
        peaks, _ = find_peaks(signal, distance=min_distance_samples, height=fallback_threshold)
        
    if len(peaks) == 0:
        peaks, _ = find_peaks(signal, distance=min_distance_samples)
        
    return peaks

def extract_heartbeats(signal: np.ndarray, r_peaks: np.ndarray, window_size: int = 100, pre_r_ratio: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    pre_samples = int(window_size * pre_r_ratio)
    post_samples = window_size - pre_samples
    signal_length = len(signal)
    
    beat_segments = []
    valid_peak_indices = []
    
    for peak_idx in r_peaks:
        start_idx = peak_idx - pre_samples
        end_idx = peak_idx + post_samples
        
        if start_idx >= 0 and end_idx <= signal_length:
            segment = signal[start_idx:end_idx]
            if len(segment) == window_size:
                beat_segments.append(segment)
                valid_peak_indices.append(peak_idx)
                
    if len(beat_segments) == 0:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
    return np.array(beat_segments, dtype=np.float32), np.array(valid_peak_indices, dtype=np.int64)

def preprocess_ecg_lead(signal: np.ndarray, fs: float = 100.0, lowcut: float = 0.5, highcut: float = 45.0, window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    filtered_signal = butter_bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    r_peaks = detect_r_peaks(filtered_signal, fs=fs)
    heartbeats, valid_peaks = extract_heartbeats(filtered_signal, r_peaks, window_size=window_size)
    
    if len(heartbeats) > 0:
        heartbeats = normalize_zscore(heartbeats)
        
    return heartbeats, valid_peaks
