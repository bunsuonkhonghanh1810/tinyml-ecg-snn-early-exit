# Issue #2: Spiking ResNet-1D Evaluation on PTB-XL & Migration to MIT-BIH Arrhythmia Dataset

- **Status**: Open / Accepted Strategy
- **Module**: `src/models/spiking_resnet1d.py`, `scripts/03_train_spiking_resnet.py`, `src/dataset/`
- **Target Hardware**: Espressif ESP32-S3 (Xtensa Dual-Core 240MHz, 512KB SRAM, 4MB Flash)
- **Author**: Nhom 2 NCKH SV

---

## 1. Context & Objectives

In response to the underfitting and spike quenching observed in the initial 4-block baseline (`001_baseline_snn_v1_evaluation.md`), we implemented and evaluated the **Spiking ResNet-1D (Res-SNN 1D)** architecture:
- Model capacity increased to **357,157 parameters** (~357 KB in INT8, safely within ESP32-S3 4MB Flash and 416KB SRAM1 data pool).
- Multi-scale 1D convolutional kernels ($k = 11, 7, 5$) to span the full cardiac cycle (P-wave, QRS complex, ST-T segment).
- Membrane-level residual connections ($V_{next} = \beta V_{mem} + I_{conv} + I_{shortcut}$) to preserve spike propagation without quenching.
- **Focal Loss** ($\gamma = 2.0$) with dynamic inverse-frequency class weights to combat extreme class imbalance.

Training and evaluation were conducted on a GPU workstation using single-lead (Lead I, 100Hz, 1.0s single-beat window) data from PTB-XL.

---

## 2. Experimental Benchmark Results (Spiking ResNet-1D on PTB-XL)

- **Model Topology**:
  - Stem: Conv1D(1 -> 32, k=11, p=5) + BN + LIF
  - Stage 1: SpikingResBlock1D(32 -> 32, k=7) + MaxPool(2)
  - Stage 2: SpikingResBlock1D(32 -> 64, k=7) + MaxPool(2)
  - Stage 3: SpikingResBlock1D(64 -> 128, k=5) + MaxPool(2)
  - Stage 4: SpikingResBlock1D(128 -> 128, k=5)
  - Head: GlobalAvgPool1D + Dropout(0.3) + Linear(128 -> 5)
- **Evaluation Set**: 24,336 test heartbeats (PTB-XL Fold 10).
- **Test Metrics**:
  - **Macro F1**: **0.3854** (38.54%)
  - **Overall Accuracy**: **44.75%**
  - **Per-class F1 Scores**:
    - `NORM`: **0.6688** (Reliable normal beat identification)
    - `CD`: **0.3955** (Moderate detection of bundle branch blocks)
    - `STTC`: **0.3369** (High confusion with MI)
    - `HYP`: **0.3045** (Poor discrimination)
    - `MI`: **0.2213** (Severe misclassification)
- **Neuromorphic & SynOps Efficiency**:
  - Healthy firing rates across all stages (Stem: 17.13%, Stage 1: 30.86%, Stage 2: 30.60%, Stage 3: 16.31%, Stage 4: 18.69%). Zero spike quenching.
  - SNN SynOps: 2,299,473.8
  - Equivalent ANN MACs: 8,965,120.0
  - **SynOps Energy Reduction**: **74.35%**
- **Test Confusion Matrix**:
  | True \ Pred | NORM | MI | STTC | CD | HYP | Total |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **NORM** | **5,869** | 753 | 2,516 | 129 | 451 | 9,718 |
  | **MI** | 602 | **709** | 1,899 | 148 | 359 | 3,717 |
  | **STTC** | 245 | 276 | **2,033** | 48 | 282 | 2,884 |
  | **CD** | 866 | 668 | 1,920 | **1,540** | 857 | 5,851 |
  | **HYP** | 251 | 286 | 818 | 72 | **739** | 2,166 |

Artifacts recorded at `outputs/spiking_resnet1d_results.json` and `outputs/figures/spiking_resnet1d_cm.png`.

---

## 3. Deep Root Cause Analysis

Despite the architectural improvements and healthy spike activity, Macro F1 remained capped below 40%. The root cause is fundamentally clinical and domain-structural rather than a failure of network capacity:

1. **Lead I Spatial Blindness for Clinical Infarction & Hypertrophy**:
   - PTB-XL labels are whole-record hospital diagnostic statements based on simultaneous 12-lead recordings.
   - **Myocardial Infarction (MI)**: The majority of MI cases are Inferior (manifesting in Leads II, III, aVF) or Anterior (V1–V4). Lead I (horizontal limb lead between Right Arm and Left Arm) exhibits normal or minimally altered morphology during inferior/anterior events.
   - **Hypertrophy (HYP)**: Clinical diagnosis (e.g., Sokolow-Lyon criteria: $S_{V1} + R_{V5} \ge 3.5\text{ mV}$) strictly requires precordial chest leads. It is impossible for any model or cardiologist to reliably diagnose ventricular hypertrophy from a single 1.0s Lead I strip.
2. **Pathological Confusion between MI and STTC**:
   - In the confusion matrix, **51.1% of all MI beats (1,899 / 3,717)** were classified as `STTC`.
   - Both pathologies exhibit ST-segment elevation/depression and T-wave inversion. On a single lead without reciprocal lead views or patient baseline comparison, morphological discrimination is mathematically ambiguous.
3. **Multi-Label Pathology Collapsed into Single-Label Target**:
   - In clinical reality, patients frequently have co-existing conditions (e.g., both chronic MI and acute STTC).
   - Forcing single-label categorization creates artificial label noise: when the model correctly identifies STTC features on an ECG labeled as primary MI, it is penalized twice (false negative for MI, false positive for STTC).
4. **Strong Binary Early-Exit Feasibility**:
   - When evaluated as **Normal vs. Abnormal** (the primary requirement for Exit 1):
     - Total abnormal beats: 14,618.
     - Detected abnormal: 12,654.
     - **Sensitivity on Abnormalities**: **86.56%**.
   - This proves that early-stage filtering works, but attempting multi-class clinical hospital diagnosis on a single wearable lead is ill-posed.

---

## 4. Decision: Migration to MIT-BIH Arrhythmia Database

To align with the project title and wearable hardware constraints on the ESP32-S3, we are migrating the primary evaluation benchmark to the **MIT-BIH Arrhythmia Database** (AAMI EC57 standard):

### 4.1. Alignment with Research Scope
- **Project Title**: *"Nghiên cứu cơ chế thoát sớm thích ứng mức pin trên mạng SNN nhận diện **rối loạn nhịp tim** thời gian thực trên vi điều khiển **ESP32-S3**"* (*Real-Time **Cardiac Arrhythmia** Detection*).
- MIT-BIH focuses strictly on **Cardiac Arrhythmias** (rhythm and ectopic beat anomalies) on wearable ambulatory ECG (modified Lead II), exactly matching single-lead smart wearable devices.
- PTB-XL evaluates 12-lead clinical cardiopathies, which exceed single-lead wearable sensor capabilities.

### 4.2. AAMI EC57 5-Class Taxonomy
The MIT-BIH benchmark categorizes beats into 5 standard classes:
1. **N (Normal / Bundle Branch Block beat)**: ~82.8% of beats.
2. **S (Supraventricular ectopic beat)**: Atrial premature beats, nodal beats.
3. **V (Ventricular ectopic beat)**: Premature ventricular contraction (PVC) — critical life-threatening indicator.
4. **F (Fusion beat)**: Fusion of ventricular and normal beat.
5. **Q (Unknown / Paced beat)**: Unclassifiable.

### 4.3. Perfect Synergy with Early-Exit Neuromorphic Architecture
- In MIT-BIH, **~83%–90% of heartbeats are Normal (Class N)**.
- **Exit 1 (Shallow SNN Block 1 & 2)**: Classifies Class N with high confidence and triggers ESP32-S3 **Light-Sleep mode (2–3mA vs 50mA active)**, achieving 70%–80% battery energy savings.
- **Exit 2 (Deep SNN Blocks)**: Awakens selectively upon detecting morphological anomalies to accurately classify dangerous ectopic beats (Class V / S) and trigger telemetry alerts.

### 4.4. Memory & Firmware Suitability (187-Sample Standard Format)
- Utilizing the standard preprocessed benchmark (`shayanfazeli/heartbeat`):
  - Fixed beat length of **187 samples** ($187 \times \text{int8} = 187\text{ bytes}$).
  - Fits directly into the circular buffer of ESP32-S3 SRAM1 without dynamic allocation.
  - Literature SNN baselines on this benchmark achieve **Macro F1 $\ge 90\% - 93\%$** and **Accuracy $\ge 97\% - 98\%$**.

---

## 5. Next Steps

- [x] Document and archive Spiking ResNet-1D PTB-XL results (`outputs/spiking_resnet1d_results.json`).
- [ ] Implement MIT-BIH dataset loader and pipeline (`src/dataset/mitbih_dataset.py`).
- [ ] Train Spiking ResNet-1D on MIT-BIH (AAMI 5-class) to establish high-F1 baseline ($\ge 90\%$).
- [ ] Implement multi-branch Early-Exit network (`src/models/early_exit_snn.py`) with dynamic confidence thresholding.
