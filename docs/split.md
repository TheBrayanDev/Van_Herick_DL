# Estrategia de Split Balanceado

## Problema

El split actual del notebook `Van_Herick_Informacion.ipynb` ordena pacientes por cantidad de imágenes y asigna secuencialmente: primeros 70 a train, siguientes 10 a val, últimos 19 a test.

**Esto genera un sesgo:** train tiene pacientes con pocas imágenes (promedio 8.89), test tiene pacientes con muchas imágenes (promedio 17.68). Si la cantidad de imágenes se correlaciona con el grado VHG, el modelo no generalizará bien.

## Solución: Stratified Split por Etiqueta a nivel Paciente

### Regla fundamental
- **Split por paciente, NO por imagen.** Un mismo paciente no puede estar en train y test simultáneamente (data leakage).
- **Stratified por etiqueta VHG (1-4).** Cada split debe tener la misma proporción de cada clase.

### Pasos

```
1. Asignar a cada paciente su etiqueta VHG
   (puede ser la moda de sus imágenes, o la etiqueta clínica)

2. Agrupar pacientes por VHG

3. Dentro de cada grupo VHG, muestrear aleatoriamente:
   - 70% → train
   - 10% → val
   - 20% → test

4. Verificar que la distribución de VHG sea similar en los 3 splits
```

### Balanceo post-split (data augmentation)

Una vez hecho el split, se aplica data augmentation **solo en train** para balancear las clases:

```python
# Por cada clase VHG, generar imágenes aumentadas hasta tener ~N por clase
# Parámetros de aumento (Fedullo et al. 2021):
aug_params = {
    'rotation_range': 7,
    'width_shift_range': 0.01,
    'height_shift_range': 0.01,
    'zoom_range': [0.5, 1.75],
    'brightness_range': [0.9, 1.1]
}
```

Esto es exactamente lo que hizo **Cassanelli et al. (2022) — Paper 3**: después del split, balancearon train con augmentation para que cada VHG tuviera ~4000 imágenes.

### Código de referencia

```python
from sklearn.model_selection import train_test_split

# pacientes_df: columna 'paciente' y 'vg_h' (etiqueta del paciente)
train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    pacientes_df['paciente'],
    pacientes_df['vg_h'],
    test_size=0.3,          # 30% para val+test
    stratify=pacientes_df['vg_h'],
    random_state=42
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.666,         # 20% test, 10% val del total
    stratify=temp_labels,
    random_state=42
)
```

### Por qué NO stratified por cantidad de imágenes

- No garantiza balance de clases VHG
- Si pacientes con muchas imágenes tienden a ser VHG=4 (como en Paper 3), el split quedaría desbalanceado
- La variable relevante es **la clase**, no la cantidad de imágenes

### Referencias

- **Cassanelli et al. (2022)** — Paper 3: Aplicaron split 60-40, luego augmentation en train para balancear clases a ~4000 imágenes por VHG
- **Fedullo et al. (2021)** — Paper 1: Split 70-30, aumentaron clase minoritaria (central) para equiparar
