# Diseño Experimental

## Estrategia de validación: Cross-Validation vs Train-Test

### Decisión: Stratified 5-Fold Cross-Validation

| Aspecto | Train-test | Cross-validation |
|---|---|---|
| Confiabilidad | Una sola medición, alta varianza | Promedio de k folds, estimación robusta |
| Dataset | Aceptable para >10k muestras | **Recomendado para ~1100 imágenes** |
| Ensayo de hiperparámetros | Sesgado (ajustás sobre un split) | No sesgado (promedio sobre k folds) |
| Costo computacional | 1 entrenamiento/modelo | k entrenamientos/modelo |
| Tiempo estimado (RTX 5080) | ~2-3 horas total | ~10-15 horas total (5 folds × 4 modelos) |

**Beneficios:**
- Stratified: mismo balance de VHG en cada fold
- Con 100 pacientes: cada fold = ~80 train, ~20 test
- Resultado: promedio + desviación estándar de métricas → más sólido para el paper
- Se deja corriendo y en ~12 horas están los resultados de los 3 modelos

### Alternativa rechazada: Train-test simple

El CSV ya tiene un split predefinido (`train/val/test` por paciente). Si se opta por simplicidad y velocidad, train-test es viable pero menos representativo.

---

## Pipeline de entrenamiento

```
Para cada modelo (AlexNet, ResNet50, ViT-S/16):
  Para cada fold (1 a 5):
    Train fold → guardar best_model_fold{}.pth
    Evaluate fold → guardar métricas + embeddings de test
  Promediar métricas across folds
  Reportar: accuracy ± std, precision, recall, F1
```

---

## Modelos

Ver [`docs/models.md`](models.md) para especificaciones detalladas de cada arquitectura.

| # | Modelo | Canal | Pre-entrenado | Paper |
|---|---|---|---|---|
| 1 | AlexNet modificado | Rojo | No | Fedullo et al. (2021) |
| 2 | ResNet50 | RGB | Sí | Cassanelli et al. (2022) |
| 3 | ViT-S/16 | RGB | Sí | Dosovitskiy et al. (2020) |

---

## Embeddings para visualización

Cada `evaluate.py` guarda automáticamente los embeddings (features de la penúltima capa) + labels + predicciones en `models/{modelo}/embeddings/`:

```
models/{modelo}/embeddings/
├── embeddings_test.npy   # (N, embedding_dim) — features penúltima capa
├── labels_test.npy       # (N,) — etiquetas verdaderas (0-3)
└── preds_test.npy        # (N,) — predicciones del modelo
```

**Uso:** estos archivos se cargan con `np.load()` para generar gráficas de análisis post-entrenamiento:

| Gráfica | Archivos necesarios | Propósito |
|---|---|---|
| t-SNE / UMAP | embeddings + labels | Visualizar separabilidad de clases en 2D |
| Matriz de confusión | labels + preds | Evaluar errores por clase |
| Precision-Recall per class | labels + preds + probs | Análisis granular |

**Dimensiones de embedding por modelo:**

| Modelo | embedding_dim |
|---|---|
| AlexNet | 4096 |
| ResNet50 | 2048 |
| ViT-S/16 | 384 |

El método `model.forward_embedding(x)` retorna `(logits, embedding)` y está disponible en los 3 modelos.
