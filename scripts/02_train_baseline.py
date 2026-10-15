import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.ecg_dataset import ECGDataset, compute_class_weights
from src.dataset.ptbxl_raw_loader import CLASS_NAMES
from src.models.baseline_snn import BaselineSNN

def calculate_synops(model, dataloader, device, timesteps=8):
    model.eval()
    conv2_fanout = 16 * 32 * 5 * 25
    conv3_fanout = 32 * 64 * 3 * 12
    conv4_fanout = 64 * 64 * 3 * 1
    fc_fanout = 64 * 5

    total_samples = 0
    total_spikes_b1 = 0
    total_spikes_b2 = 0
    total_spikes_b3 = 0
    total_spikes_b4 = 0

    total_neurons_b1 = 16 * 100
    total_neurons_b2 = 32 * 25
    total_neurons_b3 = 64 * 12
    total_neurons_b4 = 64 * 1

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            bs = x.size(0)
            _, spikes = model(x, return_spikes=True)

            total_samples += bs
            total_spikes_b1 += spikes["b1"].sum().item()
            total_spikes_b2 += spikes["b2"].sum().item()
            total_spikes_b3 += spikes["b3"].sum().item()
            total_spikes_b4 += spikes["b4"].sum().item()

    fr1 = total_spikes_b1 / (total_samples * timesteps * total_neurons_b1)
    fr2 = total_spikes_b2 / (total_samples * timesteps * total_neurons_b2)
    fr3 = total_spikes_b3 / (total_samples * timesteps * total_neurons_b3)
    fr4 = total_spikes_b4 / (total_samples * timesteps * total_neurons_b4)

    mac_conv1 = 1 * 16 * 5 * 100
    synops_conv2 = conv2_fanout * fr1
    synops_conv3 = conv3_fanout * fr2
    synops_conv4 = conv4_fanout * fr3
    synops_fc = fc_fanout * fr4
    total_synops = mac_conv1 + synops_conv2 + synops_conv3 + synops_conv4 + synops_fc

    ann_total_macs = mac_conv1 + conv2_fanout + conv3_fanout + conv4_fanout + fc_fanout
    synops_reduction = (1.0 - total_synops / ann_total_macs) * 100.0

    return {
        "firing_rates": {"b1": fr1, "b2": fr2, "b3": fr3, "b4": fr4},
        "synops": {
            "mac_conv1": mac_conv1,
            "synops_conv2": synops_conv2,
            "synops_conv3": synops_conv3,
            "synops_conv4": synops_conv4,
            "synops_fc": synops_fc,
            "total_synops": total_synops,
            "ann_total_macs": ann_total_macs,
            "synops_reduction_pct": synops_reduction,
        },
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mean_loss = total_loss / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(5)))

    return {
        "loss": mean_loss,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "preds": all_preds,
        "targets": all_targets,
    }

def plot_confusion_matrix(cm, class_names, save_path):
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Baseline 1D-SNN Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    fmt_thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > fmt_thresh else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(base_dir, "src", "data", "processed")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    output_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "train.npz")
    val_path = os.path.join(processed_dir, "val.npz")
    test_path = os.path.join(processed_dir, "test.npz")

    train_dataset = ECGDataset(train_path)
    val_dataset = ECGDataset(val_path)
    test_dataset = ECGDataset(test_path)

    if args.subsample is not None and args.subsample < len(train_dataset):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.subsample, random_state=42)
        indices, _ = next(sss.split(np.zeros(len(train_dataset)), train_dataset.y.numpy()))
        train_dataset = Subset(train_dataset, indices)
        y_for_weights = train_dataset.dataset.y[indices]
    else:
        y_for_weights = train_dataset.y

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(y_for_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = BaselineSNN(timesteps=args.timesteps, num_classes=5, beta=args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, "baseline_snn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        processed_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            processed_count += x.size(0)

        scheduler.step()
        train_loss = running_loss / processed_count
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "hyperparameters": vars(args),
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    synops_metrics = calculate_synops(model, test_loader, device, timesteps=args.timesteps)

    cm_array = np.array(test_metrics["confusion_matrix"])
    cm_plot_path = os.path.join(figures_dir, "baseline_snn_confusion_matrix.png")
    plot_confusion_matrix(cm_array, CLASS_NAMES, cm_plot_path)

    results = {
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["best_val_f1"],
        "test_metrics": {
            "loss": test_metrics["loss"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class_f1": test_metrics["per_class_f1"],
            "accuracy": test_metrics["accuracy"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "synops_metrics": synops_metrics,
        "class_names": CLASS_NAMES,
    }

    result_path = os.path.join(output_dir, "baseline_snn_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- Final Test Evaluation ---")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("Per-class F1:")
    for name, f1 in zip(CLASS_NAMES, test_metrics["per_class_f1"]):
        print(f"  {name:5s}: {f1:.4f}")
    print(f"Total SynOps: {synops_metrics['synops']['total_synops']:.1f}")
    print(f"ANN Equivalent MACs: {synops_metrics['synops']['ann_total_macs']:.1f}")
    print(f"SynOps Reduction: {synops_metrics['synops']['synops_reduction_pct']:.2f}%")
    print(f"Results saved to: {result_path}")
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

if __name__ == "__main__":
    main()
