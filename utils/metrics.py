import numpy as np
import torch


def compute_metrics(y_true, y_pred, num_classes=4):
    """
    Calcula accuracy, precision, recall y F1 por clase.

    Args:
        y_true: array 1D de etiquetas verdaderas.
        y_pred: array 1D de etiquetas predichas.
        num_classes: número de clases (default 4 para VHG 1-4).

    Returns:
        dict con accuracy global y dicts por clase.
    """
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = correct / total

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        tn = confusion.sum() - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c + 1] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return {
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def compute_loss(model, dataloader, criterion, device):
    """
    Calcula la pérdida promedio en un dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches if total_batches > 0 else 0.0
