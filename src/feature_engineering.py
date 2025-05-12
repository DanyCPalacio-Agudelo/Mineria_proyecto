# src/feature_engineering.py
import pandas as pd

def create_calendar_features(df):
    """
    Genera características de calendario como día de la semana, mes, etc.
    """
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df