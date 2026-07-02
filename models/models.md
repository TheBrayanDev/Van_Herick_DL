# Modelos — Clasificación VHG (Van Herick)

## Resumen

| Modelo | Parámetros | Canal | Pre-entrenado | Baseline Paper |
|---|---|---|---|---|
| AlexNet modificado | ~57M | Rojo (1) | No | Fedullo et al. (2021) |
| ResNet50 | ~23.5M | RGB (3) | Sí (ImageNet) | Cassanelli et al. (2022) |
| EfficientNet-B0 | ~4M | RGB (3) | Sí (ImageNet) | Tan & Le (2019) |
| ViT-S/16 | ~22M | RGB (3) | Sí (ImageNet-21k) | Dosovitskiy et al. (2020) |

---

## 1. AlexNet modificado

**Paper base:** Fedullo et al. (2021) — *Automatic Grading of Peripheral Anterior Chamber Depth by SL Images*

**Arquitectura:** AlexNet adaptado a 1 canal de entrada (rojo).

**Justificación:**
- Primer modelo del proyecto, implementación desde cero.
- Canal rojo según Fedullo: "the red channel was extracted to emphasize the light line from the slit-lamp".
- Sirve como baseline de CNN tradicional sin transfer learning.

**Detalles técnicos:**
- Entrada: (1, 224, 224) — canal rojo extraído + resize
- 5 capas convolucionales + 3 fully connected
- Dropout: 0.5 en las FC
- Inicialización: Kaiming Normal (conv) + Normal (fc)
- Output: 4 clases (VHG 1-4)

**Hiperparámetros:**
- Batch: 64
- Learning rate: 1e-3 (Adam)
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (patience 5, factor 0.5)
- Early stopping: patience 10

---

## 2. ResNet50

**Paper base:** Cassanelli et al. (2022) — *Deep Learning in the Classification of Anterior Chamber Angle of the Eye*

**Arquitectura:** ResNet50 pre-entrenado en ImageNet + classifier head.

**Justificación:**
- Modelo usado directamente en la literatura de clasificación de ángulo de cámara anterior.
- Transfer learning permite mejor rendimiento con datasets pequeños (~1100 imágenes).
- Es el modelo más comparable con resultados publicados.

**Detalles técnicos:**
- Entrada: (3, 224, 224) — RGB completo
- Backbone: ResNet50 con pesos ImageNet
- Classifier head: Dropout(0.5) → Linear(2048, 4)
- Aprendizaje: fine-tuning completo (no solo la cabeza)

**Hiperparámetros:**
- Batch: 64
- Learning rate: 1e-3 (Adam)
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (patience 5, factor 0.5)
- Early stopping: patience 10

---

## 3. EfficientNet-B0

**Paper base:** Tan & Le (2019) — *EfficientNet: Rethinking Model Scaling for CNNs*

**Arquitectura:** EfficientNet-B0 pre-entrenado en ImageNet + classifier head.

**Justificación:**
- Estado del arte en eficiencia: accuracy de ResNet-50 con 1/8 de parámetros.
- Ideal para el dataset actual (~1100 imágenes): menor riesgo de overfitting.
- Opción óptima si se considera deploy clínico a futuro.

**Detalles técnicos:**
- Entrada: (3, 224, 224) — RGB completo
- Backbone: EfficientNet-B0 con pesos ImageNet
- Classifier head: Dropout(0.5) → Linear(1280, 4)
- Aprendizaje: fine-tuning completo

**Hiperparámetros:**
- Batch: 64
- Learning rate: 1e-3 (Adam)
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (patience 5, factor 0.5)
- Early stopping: patience 10

---

## 4. ViT-S/16

**Paper base:** Dosovitskiy et al. (2020) — *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*

**Arquitectura:** Vision Transformer Small (ViT-S/16) pre-entrenado en ImageNet-21k.

**Justificación:**
- Transformer puro con atención global, alternativa a las CNN tradicionales.
- Menor riesgo de overfitting que ViT-B/16 (22M vs 86M parámetros) con datasets pequeños.
- Comparable en parámetros a ResNet50, ideal para comparar CNN vs Transformer.

**Detalles técnicos:**
- Entrada: (3, 224, 224) — RGB completo
- Patch size: 16×16 (196 patches)
- 12 capas Transformer, hidden dim 384, 6 cabezas de atención
- Classifier head: Dropout(0.5) → Linear(384, 4)
- Aprendizaje: fine-tuning completo

**Hiperparámetros:**
- Batch: 64
- Learning rate: 5e-4 (Adam) — LR menor por ser transformer
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (patience 5, factor 0.5)
- Early stopping: patience 10

---

## Data augmentation común

Parámetros idénticos a Fedullo et al. (2021), Table I:

| Transformación | Rango |
|---|---|
| Rotación | ±7° |
| Shift horizontal/vertical | ±1% |
| Zoom | 0.5× – 1.75× |
| Brillo | 0.9 – 1.1 |

Aplicada solo al split de entrenamiento.

---

## Output por modelo

Cada `train.py` genera en su propia carpeta:

```
models/{modelo}/
├── config.yaml          # Hiperparámetros (editable)
├── model.py             # Definición de la arquitectura
├── train.py             # Entry point de entrenamiento
├── evaluate.py          # Evaluación en test set
├── visualize.py         # Gráficas desde CSV
├── logs/
│   ├── training.log     # Log completo consola + archivo
│   └── tensorboard/     # Eventos para TensorBoard
├── metrics.csv          # Métricas epoch por epoch
├── best_model.pth       # Checkpoint con menor val_loss
└── plots/
    ├── loss_curves.png
    └── accuracy_curves.png
```

---

## Hardware objetivo

- GPU: NVIDIA RTX 5080 (16 GB GDDR7, Blackwell)
- CUDA: 13.3 (driver 610.62+)
- PyTorch: 2.x con CUDA 12.4+ (wheels backward compatibles)
