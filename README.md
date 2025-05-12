# Pronóstico de Ventas Diarias por Tienda

Este proyecto tiene como objetivo realizar un pronóstico de las ventas diarias por tienda utilizando técnicas de machine learning. En particular, se utilizan los siguientes métodos:

- **LightGBM** multiserie con `skforecast` para series de tiempo.
- **Regresión lineal** como modelo base de comparación.

## Descripción

El proyecto se basa en un dataset de ventas diarias por tienda, donde cada fila representa las ventas agregadas de una tienda en una fecha. El objetivo es predecir las ventas futuras a partir de datos históricos de cada tienda.

### El cuaderno realiza las siguientes tareas:

1. **Carga de datos**: Se lee un archivo `ventas_por_tienda_dia.parquet` que contiene las ventas por tienda en formato Parquet.
2. **Preparación de datos**: Se completan los huecos del calendario y se generan variables de calendario.
3. **Entrenamiento de modelos**: Se entrena un modelo de regresión lineal como baseline y un modelo LightGBM multiserie.
4. **Evaluación**: Se calculan las métricas de desempeño (MAE, RMSE) para ambos modelos.
5. **Manejo de errores**: Se gestionan índices, tipos y columnas duplicadas para evitar posibles errores.

## Requisitos

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias:

- `pyspark>=3.2.0`
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `lightgbm>=3.3.0`
- `scikit-learn>=0.24.0`
- `skforecast==0.15.0`

Puedes instalar todas las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
