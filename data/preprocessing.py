import cv2
import numpy as np


IMG_SIZE = 224


def preprocess_red_channel(img):
    """
    Pipeline para AlexNet (canal rojo).
    Basado en Fedullo et al. (2021) — Paper 1:
    "the red channel was extracted to emphasize the light line"

    1. Redimensionar a IMG_SIZExIMG_SIZE
    2. Extraer canal rojo
    3. Normalizar a [0, 1]

    Returns:
        np.ndarray shape (IMG_SIZE, IMG_SIZE, 1)
    """
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    red = img[:, :, 0].astype(np.float32) / 255.0
    return np.expand_dims(red, axis=-1)


def preprocess_rgb(img):
    """
    Pipeline para modelos pre-entrenados (ResNet, EfficientNet).

    1. Redimensionar a IMG_SIZExIMG_SIZE
    2. Normalizar a [0, 1]

    Returns:
        np.ndarray shape (IMG_SIZE, IMG_SIZE, 3)
    """
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0
