# run_pipeline.py
import pandas as pd
from src.data_preprocessing import load_data, fill_missing_dates
from src.feature_engineering import create_calendar_features
from src.model_training import train_lightgbm_model
from src.model_evaluation import evaluate_model
from src.utils import print_metrics

# Configuración
PARQUET_PATH = 'data/ventas_por_tienda_dia.parquet'
START_DATE = '2022-01-02'
END_DATE = '2025-02-28'

# 1. Cargar y preparar los datos
df = load_data(PARQUET_PATH)
df = fill_missing_dates(df, START_DATE, END_DATE)

# 2. Crear características del calendario
df = create_calendar_features(df)

# 3. Entrenar el modelo LightGBM
y_train = df['PrecioVta']  # Suponiendo que 'PrecioVta' es la variable dependiente
X_train = df.drop(columns=['PrecioVta'])
model = train_lightgbm_model(y_train, X_train)

# 4. Evaluar el modelo
y_pred = model.predict(X_train)  # Predicción sobre los datos de entrenamiento (puede cambiarse para validación)
mae, rmse = evaluate_model(y_train, y_pred)

# 5. Mostrar métricas
print_metrics(mae, rmse)
