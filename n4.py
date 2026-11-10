import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

# 1. Данные
num_cols = ['age', 'income']
cat_cols = ['city', 'education']

# --- СБОРКА ПАЙПЛАЙНА (TODO 1) ---

# TODO 1.1: num_transformer (передаем список кортежей)
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# TODO 1.2: cat_transformer (передаем список кортежей)
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# TODO 1.3: preprocessor (передаем transformers=[...])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# TODO 1.4: full_pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])


# --- КРОСС-ВАЛИДАЦИЯ (TODO 2) ---

kf = KFold(n_splits=5, shuffle=True, random_state=42) # shuffle=True без кавычек

# В cross_val_score передаем: estimator, X, y, cv, scoring
scores = cross_val_score(
    estimator=full_pipeline, 
    X=X, 
    y=y, 
    cv=kf, 
    scoring='neg_root_mean_squared_error'
)