import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.ptbxl_raw_loader import CLASS_NAMES

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(base_dir, "src", "data", "processed")
    output_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "train.npz")
    val_path = os.path.join(processed_dir, "val.npz")
    test_path = os.path.join(processed_dir, "test.npz")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        sys.exit(1)

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    x_train, y_train = train_data["x"], train_data["y"]
    x_val, y_val = val_data["x"], val_data["y"]
    x_test, y_test = test_data["x"], test_data["y"]

    time_axis = np.linspace(0.0, 1.0, x_train.shape[1])

    class_descriptions = {
        "NORM": "Normal Beat (Binh thuong)",
        "MI": "Myocardial Infarction (Nhoi mau co tim)",
        "STTC": "ST/T Change (Bien doi doan ST/T)",
        "CD": "Conduction Disturbance (Roi loan dan truyen)",
        "HYP": "Hypertrophy (Phi dai co tim)",
    }

    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(5, 3, wspace=0.3, hspace=0.45, figure=fig)

    np.random.seed(42)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax_wave = fig.add_subplot(grid[cls_idx, 0:2])
        
        cls_indices = np.where(y_train == cls_idx)[0]
        cls_samples = x_train[cls_indices]
        
        mean_wave = np.mean(cls_samples, axis=0)
        std_wave = np.std(cls_samples, axis=0)

        random_picks = np.random.choice(cls_indices, size=3, replace=False)
        colors = ["#90caf9", "#a5d6a7", "#ffcc80"]
        for pick_idx, c in zip(random_picks, colors):
            ax_wave.plot(time_axis, x_train[pick_idx], color=c, alpha=0.6, linewidth=1.0)

        ax_wave.plot(time_axis, mean_wave, color="#1565c0", linewidth=2.2, label="Class Mean")
        ax_wave.fill_between(time_axis, mean_wave - std_wave, mean_wave + std_wave, color="#1565c0", alpha=0.15)
        
        ax_wave.axvline(x=0.35, color="#d32f2f", linestyle="--", linewidth=1.2, alpha=0.7, label="R-peak (t=0.35s)")
        
        desc = class_descriptions.get(cls_name, "")
        ax_wave.set_title(f"Class [{cls_idx}] {cls_name}: {desc} (N={len(cls_indices):,})", fontsize=11, fontweight="bold", loc="left")
        ax_wave.set_ylabel("Z-score Amp", fontsize=9)
        ax_wave.set_xlim(0.0, 1.0)
        ax_wave.set_ylim(-3.5, 4.5)
        ax_wave.grid(True, linestyle=":", alpha=0.6)
        
        if cls_idx == 0:
            ax_wave.legend(loc="upper right", fontsize=8)
        if cls_idx == 4:
            ax_wave.set_xlabel("Time (seconds)", fontsize=10)

    ax_bar = fig.add_subplot(grid[:, 2])

    train_counts = [np.sum(y_train == i) for i in range(len(CLASS_NAMES))]
    val_counts = [np.sum(y_val == i) for i in range(len(CLASS_NAMES))]
    test_counts = [np.sum(y_test == i) for i in range(len(CLASS_NAMES))]

    bar_width = 0.25
    x_indices = np.arange(len(CLASS_NAMES))

    b1 = ax_bar.barh(x_indices - bar_width, train_counts, bar_width, label="Train", color="#1976d2")
    b2 = ax_bar.barh(x_indices, val_counts, bar_width, label="Val", color="#388e3c")
    b3 = ax_bar.barh(x_indices + bar_width, test_counts, bar_width, label="Test", color="#f57c00")

    ax_bar.set_yticks(x_indices)
    ax_bar.set_yticklabels(CLASS_NAMES, fontsize=10, fontweight="bold")
    ax_bar.set_xlabel("Heartbeat Count", fontsize=10)
    ax_bar.set_title("Dataset Distribution (Total: 241,134 Beats)", fontsize=12, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_bar.grid(True, linestyle=":", alpha=0.6, axis="x")

    for bar in b1:
        w = bar.get_width()
        ax_bar.text(w + 1000, bar.get_y() + bar.get_height() / 2, f"{w:,}", va="center", fontsize=7.5, color="#1976d2")

    save_path = os.path.join(output_dir, "ecg_samples_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization successfully saved at: {save_path}")

if __name__ == "__main__":
    main()
