import os
import sys
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def visualize(config_path="config.yaml", metrics_csv="metrics.csv"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(__file__).parent
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(model_dir / metrics_csv)

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{cfg['model']['name']} — Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] loss_curves.png guardado")

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy", marker="o")
    ax.plot(df["epoch"], df["val_accuracy"], label="Val Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{cfg['model']['name']} — Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_curves.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] accuracy_curves.png guardado")

    print(f"Gráficos guardados en: {plots_dir}")


if __name__ == "__main__":
    visualize()
