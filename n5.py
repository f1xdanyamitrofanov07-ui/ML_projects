import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df = pd.DataFrame({
    'user_id': ['A', 'A', 'B', 'B', 'A', 'C', 'B', 'C', 'A', 'C'],
    'target': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
df['user_id_te'] = np.nan 

# --- OOF TARGET ENCODING ---
for train_idx, val_idx in kf.split(df):
    # TODO 1: Выдели train_fold и val_fold через df.iloc
    train_fold = df.iloc(train_idx)
    val_fold = df.iloc(val_idx)
    # TODO 2: Посчитай средний target для каждого user_id на train_fold
    # te_map = train_fold.groupby('user_id')['target'].mean()
    te_map = train_fold.groupby('user_id')['target'].mean()
    
    # TODO 3: Заполни df.loc[val_idx, 'user_id_te'] через val_fold['user_id'].map(te_map)
    df.loc[val_idx, 'user_id_te'] = val_fold['user_id'].map(te_map)
    pass

df['user_id_te'] = df['user_id_te'].fillna(df['target'].mean())