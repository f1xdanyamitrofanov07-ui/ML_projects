
df['user_roll_mean_2'] = (
    df.groupby('user_id')['sales']
    .transform(lambda x: x.shift(1).rolling(2).mean())
)

df['user_roll_mean_2'] = (
    df.groupby('user_id')['sales']
    .shift(1)
    .groupby(df['user_id'])
    .rolling(2).mean()
    .reset_index(level=0, drop=True)
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

# TODO 2: Заполни параметры в SelectFromModel:
# 1. Модель-оценщик: Ridge(alpha=1.0, random_state=42)
# 2. Порог отсечения: 'mean' (выше среднего)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

# TODO 2: Заполни параметры в SelectFromModel:
# 1. Модель-оценщик: Ridge(alpha=1.0, random_state=42)
# 2. Порог отсечения: 'mean' (выше среднего)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selector', SelectFromModel(
        estimator=Ridge(alpha=1.0), # TODO 2.1
        threshold= 'mean'  # TODO 2.2
    )),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

x_batch = torch.randn(16, 10)
y_batch = torch.randn(16, 1)

# TODO 3: Напиши правильный порядок 4 шагов после получения предсказаний:
# 1. Расчет loss
# 2. Обнуление градиентов
# 3. Вычисление градиентов (backprop)
# 4. Шаг оптимизатора

outputs = model(x_batch)

# Напиши 4 строки кода здесь:
loss = criterion(outputs, y_batch)
optimizer.zero_grad()
loss.backward()
optimizer.step()
