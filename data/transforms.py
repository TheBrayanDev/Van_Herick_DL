import torch
import numpy as np


class RedChannelTransform:
    """
    Transformación para AlexNet: extrae canal rojo y normaliza.
    """

    def __call__(self, img):
        red = img[:, :, 0].astype(np.float32) / 255.0
        return torch.from_numpy(red).unsqueeze(0).float()  # (1, H, W)


class RGBTransform:
    """
    Transformación para modelos pre-entrenados: normaliza a [0,1].
    """

    def __call__(self, img):
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)


class AugmentationRed:
    """
    Data augmentation para canal rojo.
    Parámetros idénticos a Fedullo et al. (2021) — Paper 1, Table I.
    """

    def __init__(self):
        self.rotation_range = 7
        self.shift_range = 0.01
        self.zoom_range = (0.5, 1.75)
        self.brightness_range = (0.9, 1.1)

    def __call__(self, img):
        # img: (1, H, W)
        import torchvision.transforms.functional as F
        import random

        # Rotación
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        img = F.rotate(img, angle, fill=0.0)

        # Shift horizontal/vertical
        h, w = img.shape[1:]
        tx = random.uniform(-self.shift_range, self.shift_range) * w
        ty = random.uniform(-self.shift_range, self.shift_range) * h
        img = F.affine(img, angle=0, translate=(tx, ty), scale=1.0, shear=0.0, fill=0.0)

        # Zoom
        zoom = random.uniform(*self.zoom_range)
        new_h, new_w = int(h * zoom), int(w * zoom)
        if zoom > 1.0:
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            img = F.resize(img, (new_h, new_w))
            img = F.crop(img, top, left, h, w)
        else:
            img = F.resize(img, (new_h, new_w))
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = F.pad(img, (pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h), fill=0.0)

        # Brillo
        brightness = random.uniform(*self.brightness_range)
        img = F.adjust_brightness(img, brightness)

        return img


class AugmentationRGB(AugmentationRed):
    """Data augmentation para imágenes RGB."""
    pass
