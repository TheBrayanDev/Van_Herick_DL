import os
import csv
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def setup_logging(log_dir):
    """
    Configura logging: archivo + consola.

    Args:
        log_dir: ruta al directorio donde guardar training.log.

    Returns:
        logger configurado.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("VanHerick")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class CSVLogger:
    """
    Guarda métricas epoch por epoch en un archivo CSV.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "learning_rate",
        ]
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _write_header(self):
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": lr,
        }
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class TensorBoardLogger:
    """
    Wrapper para TensorBoard.
    """

    def __init__(self, log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)
        self.writer.add_scalar("Learning_rate", lr, epoch)

    def log_hyperparams(self, hparams_dict):
        """Registra hiperparámetros como texto."""
        hparams_str = "\n".join(f"{k}: {v}" for k, v in hparams_dict.items())
        self.writer.add_text("Hyperparameters", hparams_str)

    def close(self):
        self.writer.close()
