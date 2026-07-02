import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloaders
from models.resnet50.model import ResNet50VHG
from utils.logger import setup_logging, CSVLogger, TensorBoardLogger


def train(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(__file__).parent
    log_dir = model_dir / cfg["logging"]["log_dir"]

    logger = setup_logging(log_dir)
    logger.info("=" * 50)
    logger.info("Iniciando entrenamiento ResNet50 VHG")
    logger.info(f"Config: {cfg['model']['name']}")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    csv_logger = CSVLogger(model_dir / "metrics.csv")
    tb_logger = TensorBoardLogger(str(log_dir / "tensorboard")) if cfg["logging"]["tensorboard"] else None

    loaders = create_dataloaders(
        csv_path=cfg["data"]["csv_path"],
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        use_red_channel=cfg["data"]["use_red_channel"],
        num_workers=cfg["data"]["num_workers"],
    )
    logger.info(f"Train: {len(loaders['train'].dataset)} muestras")
    logger.info(f"Val:   {len(loaders['val'].dataset)} muestras")
    logger.info(f"Test:  {len(loaders['test'].dataset)} muestras")

    model = ResNet50VHG(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)
    logger.info(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["training"]["lr_scheduler"]["patience"],
        factor=cfg["training"]["lr_scheduler"]["factor"],
    )

    early_stop_patience = cfg["callbacks"]["early_stop_patience"]
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_model_path = model_dir / "best_model.pth"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(loaders["train"])
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(loaders["val"])
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        csv_logger.log(epoch, avg_train_loss, avg_val_loss, train_acc, val_acc, current_lr)
        if tb_logger:
            tb_logger.log_scalars(epoch, avg_train_loss, avg_val_loss, train_acc, val_acc, current_lr)

        logger.info(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  >> Nuevo mejor modelo guardado (epoch {epoch})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"  >> Early stopping en epoch {epoch}")
                break

    logger.info("=" * 50)
    logger.info(f"Entrenamiento completado. Mejor época: {best_epoch} (val_loss: {best_val_loss:.4f})")
    if tb_logger:
        tb_logger.close()

    return str(best_model_path), str(model_dir / "metrics.csv")


if __name__ == "__main__":
    train()
