import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.drive_utils import load_image
from data.preprocessing import IMG_SIZE
from data.transforms import RedChannelTransform, RGBTransform, AugmentationRed, AugmentationRGB


class VanHerickDataset(Dataset):
    """
    Dataset PyTorch para imágenes Van Herick desde disco local.

    CSV debe tener columnas: paciente, image_name, ojo, split, etiqueta.
    Ruta: data_dir / {paciente} / {ojo} / {image_name}.

    Args:
        csv_path: ruta al CSV.
        data_dir: directorio raíz (pics_by_patients/).
        split: 'train', 'val' o 'test'.
        use_red_channel: True para AlexNet (1 canal), False para RGB.
        augment: aplicar data augmentation (solo train).
    """

    COL_PACIENTE = "paciente"
    COL_IMAGE = "image_name"
    COL_OJO = "ojo"
    COL_SPLIT = "split"
    COL_ETIQUETA = "etiqueta"

    def __init__(self, csv_path, data_dir, split="train", use_red_channel=True, augment=False):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir

        if self.COL_SPLIT in self.df.columns:
            self.df = self.df[self.df[self.COL_SPLIT] == split].reset_index(drop=True)

        if self.COL_ETIQUETA in self.df.columns:
            self.df = self.df.dropna(subset=[self.COL_ETIQUETA]).reset_index(drop=True)
            self.df[self.COL_ETIQUETA] = self.df[self.COL_ETIQUETA].astype(int)

        self.use_red_channel = use_red_channel
        self.augment = augment
        self.img_size = (IMG_SIZE, IMG_SIZE)

        if use_red_channel:
            self.transform = RedChannelTransform()
            self.aug_transform = AugmentationRed() if augment else None
        else:
            self.transform = RGBTransform()
            self.aug_transform = AugmentationRGB() if augment else None

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row):
        paciente = str(row[self.COL_PACIENTE]).strip()
        ojo = str(row[self.COL_OJO]).strip()
        fname = str(row[self.COL_IMAGE]).strip()
        return os.path.join(self.data_dir, paciente, ojo, fname)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._get_image_path(row)

        img = load_image(img_path, target_size=self.img_size)
        if img is None:
            dummy = torch.zeros((1, IMG_SIZE, IMG_SIZE)) if self.use_red_channel else torch.zeros((3, IMG_SIZE, IMG_SIZE))
            label = int(row[self.COL_ETIQUETA]) - 1 if self.COL_ETIQUETA in self.df.columns and pd.notna(row[self.COL_ETIQUETA]) else 0
            return dummy, label

        tensor = self.transform(img)

        if self.augment and self.aug_transform is not None:
            tensor = self.aug_transform(tensor)

        label = int(row[self.COL_ETIQUETA]) - 1 if self.COL_ETIQUETA in self.df.columns and pd.notna(row[self.COL_ETIQUETA]) else 0

        return tensor, label


def create_dataloaders(csv_path, data_dir, batch_size=32, use_red_channel=True, num_workers=2):
    """
    Crea DataLoaders para train, val y test desde disco local.

    Returns:
        dict con 'train', 'val', 'test' DataLoaders.
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        augment = split == "train"
        ds = VanHerickDataset(
            csv_path=csv_path,
            data_dir=data_dir,
            split=split,
            use_red_channel=use_red_channel,
            augment=augment,
        )
        datasets[split] = ds

    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders
