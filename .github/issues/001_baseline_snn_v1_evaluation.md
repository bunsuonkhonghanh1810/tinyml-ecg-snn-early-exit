# Issue #1: Baseline 1D-SNN v1 (4-Block Conv1D-LIF) Underperforming (Macro F1 = 0.4132)

- **Status**: Open / In Progress
- **Module**: `src/models/baseline_snn.py`, `scripts/02_train_baseline.py`
- **Target Hardware**: Espressif ESP32-S3 (Xtensa Dual-Core 240MHz, 512KB SRAM, 4MB Flash)
- **Author**: Nhom 2 NCKH SV

---

## 1. Context & Motivation
In accordance with Task 3 of the project specification (`docs/PROJECT_SPECIFICATION_SNN_EARLY_EXIT.md`), we implemented the initial baseline 1D Spiking Neural Network (without early-exit) to establish a performance and computational benchmark on the PTB-XL ECG dataset (Lead I, 100Hz, single-beat segmentation).

Hardware profile from `paper/esp32s3-mm.pdf`:
- Total internal SRAM: 512 KB (SRAM1 data pool: 416 KB).
- External SPI Flash: 4 MB – 16 MB.
- Hardware Vector SIMD: 16 INT8 operations per CPU cycle.

## 2. Experimental Benchmark Results (v1 Architecture)
- **Model Topology**: 4 Conv1D-LIF blocks:
  - Block 1: Conv1D(1 -> 16, k=5) + BN + LIF(beta=0.85) + MaxPool(2)
  - Block 2: Conv1D(16 -> 32, k=5) + BN + LIF(beta=0.85) + MaxPool(2)
  - Block 3: Conv1D(32 -> 64, k=3) + BN + LIF(beta=0.85) + MaxPool(2)
  - Block 4: Conv1D(64 -> 64, k=3) + BN + LIF(beta=0.85) + AdaptiveAvgPool(1)
  - Head: Linear(64 -> 5)
- **Total Parameters**: 21,749 (~21.7 KB in INT8)
- **Training Setup**: 8 epochs, Cosine Annealing LR (1e-3 -> 1e-5), Adam, Cross-Entropy with class weights, timesteps T = 8.
- **Test Metrics (24,336 test heartbeats)**:
  - **Macro F1**: **0.4132** (Target requirement: >= 0.85)
  - **Overall Accuracy**: **51.19%**
  - **Per-class F1 Scores**:
    - `NORM`: **0.7208** (Strong baseline recognition, validates shallow layer early-exit feasibility)
    - `CD`: **0.5406** (Recognizes substantial QRS morphological widening)
    - `HYP`: **0.3369** (Moderate)
    - `STTC`: **0.3238** (Moderate)
    - `MI`: **0.1440** (Severely underperforming; near zero discrimination)
  - **Neuromorphic Efficiency**:
    - SNN SynOps: 38,495.4
    - Equivalent ANN MACs: 158,336.0
    - **SynOps Reduction**: **75.69%** (Significant computational energy savings)
  - **Confusion Matrix**: Recorded at `outputs/figures/baseline_snn_confusion_matrix.png` (archived as `outputs/figures/baseline_snn_cm_v1_4block_toy.png`).

## 3. Root Cause Analysis
1. **Severe Underfitting**:
   - The 21,749-parameter capacity is vastly undersized for 5-class ECG diagnosis on a single lead.
   - Per `esp32s3-mm.pdf`, SRAM1 has 416 KB. The current model uses < 2.5 KB of membrane state buffer (< 0.6% of SRAM1) and 21.7 KB Flash (< 0.55% of 4MB Flash). We are severely underutilizing the microcontroller.
2. **Narrow Receptive Field**:
   - Kernel sizes (k=3, 5) cover only 30ms - 50ms at 100Hz. Full cardiac cycles span 800ms - 1000ms with P-wave (80ms), PR interval (120-200ms), QRS (80-120ms), and ST segment (100ms). The model cannot capture long-range contextual relationships.
3. **Spike Quenching in Layer 1**:
   - Firing rate at Block 1 was only 9.68%, meaning > 90% of the input signal was suppressed before entering deeper layers.
4. **Insufficient Training Regimen**:
   - Surrogate gradient BPTT requires 20-30 epochs on larger sample pools for proper credit assignment.

## 4. Proposed Solution & Next Steps (Baseline SNN v2)
- [ ] Upgrade to a **Spiking ResNet-1D (Res-SNN 1D)** architecture with residual blocks and membrane-level skip connections.
- [ ] Increase kernel sizes to **k = 7, 11, 15** for multi-scale feature extraction of QRS and ST segments.
- [ ] Expand channel dimensions to 32 -> 64 -> 128 -> 128 -> 256 (~150k - 300k parameters, ~150KB - 300KB INT8, requiring < 25KB SRAM buffer).
- [ ] Introduce Focal Loss to strongly penalize misclassifications on minority classes (`MI`, `HYP`).
- [ ] Train over 20-30 epochs on full/extended training folds.
