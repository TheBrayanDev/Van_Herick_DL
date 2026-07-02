import os
import numpy as np
from PIL import Image


def load_image(path, target_size=None):
    """
    Carga una imagen desde el sistema de archivos local.

    Args:
        path: ruta absoluta o relativa al archivo de imagen.
        target_size: tuple (w, h) opcional para redimensionar.

    Returns:
        np.ndarray en RGB, o None si hay error.
    """
    try:
        img = Image.open(path)
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f"  [WARNING] Error cargando {path}: {e}")
        return None
