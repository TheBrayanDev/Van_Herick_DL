# Van Herick DL — Ejecución en Windows (RTX 5080)

## 1. Prerrequisitos

- [ ] Python 3.11+ instalado ([python.org](https://python.org))
- [ ] Git instalado ([git-scm.com](https://git-scm.com))
- [ ] CUDA 13.3 listo (`nvidia-smi` must show CUDA Version)
- [ ] Driver NVIDIA 610.62+ instalado
- [ ] Proyecto clonado en `C:\proyectos\Van_Herick_DL`
- [ ] Dataset descargado en `C:\proyectos\Van_Herick_DL\pics_by_patients\`
- [ ] `dataset_master_split.csv` colocado dentro de `pics_by_patients\`

---

## 2. Entorno Python

```powershell
cd C:\proyectos\Van_Herick_DL

# Crear virtual env
python -m venv venv
.\venv\Scripts\activate

# Actualizar pip
python -m pip install --upgrade pip
```

---

## 3. Instalar PyTorch con CUDA 13.3

> CUDA 13.3 es muy nuevo; los wheels oficiales de PyTorch con CUDA 12.4/12.6 son **backward compatibles** con tu driver 610.62. Si ya existen wheels CUDA 13.x, usa ese index.

```powershell
# Opción A — CUDA 12.4 (funciona con driver 610.62)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Opción B — CUDA 13.x (si ya existe, probar primero)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Verificar CUDA desde Python

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**
```
CUDA available: True
Device: NVIDIA GeForce RTX 5080
```

---

## 4. Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

## 5. Ajustar config.yaml para Windows

En cada modelo (`models/alexnet/config.yaml`, `models/resnet50/config.yaml`, `models/efficientnet_b0/config.yaml`, `models/vit_s16/config.yaml`), **reemplazar los paths**:

```yaml
data:
  csv_path: "C:\\proyectos\\Van_Herick_DL\\pics_by_patients\\dataset_master_split.csv"
  data_dir: "C:\\proyectos\\Van_Herick_DL\\pics_by_patients"
```

> **IMPORTANTE**: Usar doble backslash `\\` en YAML para Windows.

---

## 6. Entrenar modelos

Cada `train.py` es independiente — se ejecuta desde su propia carpeta.

### 6.1 AlexNet

```powershell
cd C:\proyectos\Van_Herick_DL\models\alexnet
python train.py
```

Output generado:
- `logs/training.log`
- `logs/tensorboard/`
- `metrics.csv`
- `best_model.pth`
- `plots/loss_curves.png`
- `plots/accuracy_curves.png`

### 6.2 ResNet50

```powershell
cd C:\proyectos\Van_Herick_DL\models\resnet50
python train.py
```

### 6.3 EfficientNet-B0

```powershell
cd C:\proyectos\Van_Herick_DL\models\efficientnet_b0
python train.py
```

### 6.4 ViT-S/16

```powershell
cd C:\proyectos\Van_Herick_DL\models\vit_s16
python train.py
```

---

## 7. Evaluar (opcional, post-entrenamiento)

```powershell
cd C:\proyectos\Van_Herick_DL\models\alexnet
python evaluate.py

cd C:\proyectos\Van_Herick_DL\models\resnet50
python evaluate.py

cd C:\proyectos\Van_Herick_DL\models\efficientnet_b0
python evaluate.py

cd C:\proyectos\Van_Herick_DL\models\vit_s16
python evaluate.py
```

---

## 8. Visualizar (opcional)

```powershell
cd C:\proyectos\Van_Herick_DL\models\alexnet
python visualize.py

cd C:\proyectos\Van_Herick_DL\models\resnet50
python visualize.py

cd C:\proyectos\Van_Herick_DL\models\efficientnet_b0
python visualize.py

cd C:\proyectos\Van_Herick_DL\models\vit_s16
python visualize.py
```

---

## 9. Monitoreo con TensorBoard

```powershell
tensorboard --logdir C:\proyectos\Van_Herick_DL\models\alexnet\logs\tensorboard
```

---

## Troubleshooting

### `CUDA not available`
- Verificar que `nvidia-smi` muestra CUDA 13.3
- Probar reinstalar PyTorch con `pip uninstall torch torchvision torchaudio` y luego con `--force-reinstall`

### `OutOfMemoryError`
- Reducir `batch_size` en `config.yaml` de 64 a 32

### `FileNotFoundError` con imágenes
- Verificar que `pics_by_patients/` tiene carpetas `patient_X/OD/` y `patient_X/OS/`
- Verificar que `dataset_master_split.csv` tiene columnas: `paciente, image_name, ojo, split, etiqueta`
