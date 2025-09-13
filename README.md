# Academ Pet Places (🇬🇧)

PyTorch project for scene classification from Places365 using **EfficientNet-B2** and **CutMix**.  
Designed to demonstrate a full pipeline: data loading, augmentations, training, validation, metric logging, inference, and early stopping.  
Focused on reproducibility and reliability of experiments.  

## Features
- EfficientNet-B2 pretrained on ImageNet  
- CutMix augmentation for improved generalization  
- Stage-wise training (classifier-only, then full model)  
- Early stopping with automatic best-model saving  
- Logging metrics to JSON for plotting  
- Inference example included  

## Installation
```bash
git clone https://github.com/Testy12789/places365-academ-pet
cd academ-pet-places
pip install -r requirements.txt
```

## Dataset
Download Places365 and set paths in the code:
```python
config = {
     ...
    "PATH_DS": "path/to/places365",
     ...
}
```

## Usage

### Training
```bash
python train.py
```
- Stage 1: only classifier is trained  
- Stage 2: full model fine-tuning  
- Metrics are logged to `logs/metrics_first_stage.json` and `logs/metrics_second_stage.json`

### Inference
```python
from PIL import Image
img = Image.open("path/to/your/image.jpg").convert("RGB")
# run inference using the trained model
```

### Plotting
```python
from plot import plot_train
plot_train("logs/metrics_first_stage.json")
```

## Notes
- Weights are not included in the repository (`models/` is in `.gitignore`)  
- Replace `path/to/...` with your local dataset paths  
- Configurable parameters: batch size, learning rate, augmentation type, etc.  

## License
Apache 2 License

---

# Academ Pet Places (🇷🇺)

Академический проект на PyTorch для классификации сцен из Places365 с использованием EfficientNet-B2 и CutMix.  
Проект демонстрирует полный пайплайн: загрузка данных, аугментации, обучение, валидация, логирование метрик, инференс и ранний стоп.  
Основное внимание уделено воспроизводимости и надежности экспериментов.

## Особенности
- EfficientNet-B2, предобученная на ImageNet  
- Аугментация CutMix для лучшей обобщаемости  
- Обучение по стадиям (сначала только классификатор, потом вся модель)  
- Ранний стоп с автоматическим сохранением лучших весов  
- Логирование метрик в JSON для построения графиков  
- Пример инференса  

## Установка
```bash
git clone https://github.com/Testy12789/places365-academ-pet
cd academ-pet-places
pip install -r requirements.txt
```

## Датасет
Скачайте Places365 и укажите пути в коде:
```python
config = {
     ...
    "PATH_DS": "path/to/places365",
     ...
}
```

## Использование

### Обучение
```bash
python train.py
```
- Stage 1: обучение только классификатора  
- Stage 2: дообучение всей модели  
- Метрики сохраняются в `logs/metrics_first_stage.json` и `logs/metrics_second_stage.json`

### Инференс
```python
from PIL import Image
img = Image.open("path/to/your/image.jpg").convert("RGB")
# инференс с использованием обученной модели
```

### Построение графиков
```python
from plot import plot_train
plot_train("logs/metrics_first_stage.json")
```

## Примечания
- Весы не включены в репозиторий (`models/` добавлен в `.gitignore`)  
- Замените `path/to/...` на свои локальные пути к датасету  
- Настраиваемые параметры: размер батча, learning rate, тип аугментации и др.

## Лицензия
Apache 2 License
