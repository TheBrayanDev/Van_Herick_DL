import os
import torch
import numpy as np
from pathlib import Path


@torch.no_grad()
def save_embeddings(model, dataloader, output_dir, device, name="val"):
    """
    Extrae y guarda embeddings (penúltima capa) + labels + preds.

    Args:
        model: modelo con método forward_embedding().
        dataloader: DataLoader del split a extraer.
        output_dir: directorio donde guardar los .npy.
        device: dispositivo (cuda/cpu).
        name: nombre del split (val, test, etc.).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_embeddings = []
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(device)
        logits, embeddings = model.forward_embedding(images)
        preds = torch.argmax(logits, dim=1)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())

    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)

    np.save(output_dir / f"embeddings_{name}.npy", embeddings_arr)
    np.save(output_dir / f"labels_{name}.npy", labels_arr)
    np.save(output_dir / f"preds_{name}.npy", preds_arr)

    print(f"  [OK] Embeddings guardados: {output_dir / f'embeddings_{name}.npy'}")
    print(f"       Shape: {embeddings_arr.shape}, dtype: {embeddings_arr.dtype}")
