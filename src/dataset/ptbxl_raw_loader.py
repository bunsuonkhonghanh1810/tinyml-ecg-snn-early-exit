import os
import ast
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

def load_ptbxl_metadata(dataset_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(dataset_dir, "ptbxl_database.csv")
    scp_path = os.path.join(dataset_dir, "scp_statements.csv")
    
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes_dict"] = df["scp_codes"].apply(ast.literal_eval)
    
    agg_df = pd.read_csv(scp_path, index_col=0)
    diag_agg = agg_df[agg_df["diagnostic"] == 1]
    
    def map_to_superclasses(scp_dict: Dict[str, float]) -> List[str]:
        superclasses = set()
        for key in scp_dict.keys():
            if key in diag_agg.index:
                cls_name = diag_agg.loc[key, "diagnostic_class"]
                if pd.notna(cls_name) and cls_name in CLASS_NAMES:
                    superclasses.add(cls_name)
        return sorted(list(superclasses))
    
    df["superclasses"] = df["scp_codes_dict"].apply(map_to_superclasses)
    
    def assign_primary_label(superclasses: List[str]) -> int:
        if len(superclasses) == 1:
            return CLASS_TO_IDX[superclasses[0]]
        elif len(superclasses) > 1:
            if "NORM" in superclasses:
                filtered = [c for c in superclasses if c != "NORM"]
                if filtered:
                    return CLASS_TO_IDX[filtered[0]]
            return CLASS_TO_IDX[superclasses[0]]
        return -1
    
    df["label_idx"] = df["superclasses"].apply(assign_primary_label)
    df["label_name"] = df["label_idx"].apply(lambda idx: IDX_TO_CLASS.get(idx, "OTHER"))
    df["is_single_label"] = df["superclasses"].apply(lambda s: len(s) == 1)
    
    return df

def read_record_signal(dat_path: str, num_leads: int = 12, gain: float = 1000.0) -> np.ndarray:
    raw_data = np.fromfile(dat_path, dtype=np.int16)
    signal = raw_data.reshape(-1, num_leads).astype(np.float32) / gain
    return signal

def get_lead_signal(signal: np.ndarray, lead_idx: int = 0) -> np.ndarray:
    return signal[:, lead_idx]
