import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloaders
from models.resnet50.model import ResNet50VHG
from utils.logger import setup_logging
from utils.metrics import compute_metrics
from utils.extract_embeddings import save_embeddings


def evaluate(config_path="config.yaml", checkpoint="best_model.pth"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(__file__).parent
    log_dir = model_dir / cfg["logging"]["log_dir"]
    logger = setup_logging(log_dir)
    logger.info("Evaluando modelo ResNet50 en test set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_dataloaders(
        csv_path=cfg["data"]["csv_path"],
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        use_red_channel=cfg["data"]["use_red_channel"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = ResNet50VHG(num_classes=cfg["model"]["num_classes"], dropout=cfg["model"]["dropout"], pretrained=False)
    checkpoint_path = model_dir / checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Checkpoint cargado: {checkpoint_path}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = compute_metrics(all_labels, all_preds, num_classes=cfg["model"]["num_classes"])

    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info("Confusion Matrix:")
    for row in results["confusion_matrix"]:
        logger.info(f"  {row}")
    logger.info("Per-Class Metrics:")
    for cls_, metrics in results["per_class"].items():
        logger.info(f"  Clase {cls_}: Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

    embeddings_dir = model_dir / "embeddings"
    save_embeddings(model, loaders["test"], embeddings_dir, device, name="test")
    logger.info(f"Embeddings guardados en: {embeddings_dir}")

    return results


if __name__ == "__main__":
    evaluate()
