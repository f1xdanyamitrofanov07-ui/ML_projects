import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
from PIL import Image

# --- 1. СИМУЛЯЦИЯ ДАТАФРЕЙМА (Уже готово) ---
df = pd.DataFrame({
    'image_path': ['img_1.jpg', 'img_2.jpg', 'img_3.jpg'],
    'label': [0, 1, 0]
})
# --- 2. АУГМЕНТАЦИИ ALBUMENTATIONS (Уже готово) ---
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- 3. CUSTOM DATASET (TODO 1) ---
class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        # TODO 1.1: Верни общее количество элементов в датасете
        return len(self.df)
        pass

    def __getitem__(self, idx):
        # TODO 1.2:
        # 1. Получи путь к картинке и метку (label) из self.df по индексу idx
        row = self.df.iloc(idx)
        image_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'label']
        
        # 2. Создай «фейковый» numpy-массив картинки размера (224, 224, 3) типом uint8:
        image_np = np.zeros((224, 224, 3), dtype = np.uint8)
        #    image_np = np.zeros((224, 224, 3), dtype=np.uint8)  (чтобы не искать реальные файлы)
        # 3. Если self.transform задан, примени его к image_np (помни, что Albumentations берет kwarg image=...)
        if self.transform is not None:
            augmented = self.transform(image = image_np)
            image_np = augmented['image']
        # 4. Преврати label в torch.tensor(label, dtype=torch.long)
        labels = torch.tensor(label, dtype= torch.long)
        # 5. Верни извлеченное трансформированное изображение и тензор метки
        return image_np, labels
        pass

# --- 4. DATALOADER И ПРОВЕРКА (TODO 2) ---
# 1. Инициализация датасета
dataset = CustomImageDataset(df, transform=train_transform)

# 2. Обертка в DataLoader (указываем batch_size=2)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Извлечение первого батча
images, labels = next(iter(dataloader))

# 4. Вывод размеров
print(images.shape)
print(labels.shape)