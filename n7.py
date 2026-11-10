import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.to_datetime(df['timestamp'])
    hour = data.dt.hour
    is_night = (hour >= 0) & (hour <= 5)
    
    temp_df = pd.DataFrame({
        'user_id': df['user_id'],
        'amount': df['amount'],
        'is_night': is_night.astype(int),
        'crypto_amount': np.where(df['category'] == 'crypto', df['amount'], 0.0)
    })
    
    features = temp_df.groupby('user_id').agg(
        total_amount=('amount', 'sum'),
        mean_amount=('amount', 'mean'),
        tx_count=('amount', 'count'),
        night_tx_ratio=('is_night', 'mean'),
        crypto_amount=('crypto_amount', 'sum')
    ).reset_index()
    
    return features