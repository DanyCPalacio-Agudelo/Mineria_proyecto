# src/data_preprocessing.py
import pandas as pd
from pyspark.sql import SparkSession

def load_data(parquet_path):
    """
    Carga el archivo Parquet utilizando PySpark.
    """
    spark = SparkSession.builder.appName('ForecastTiendaDiario').getOrCreate()
    df = spark.read.parquet(parquet_path)
    return df

def fill_missing_dates(df, start_date, end_date):
    """
    Completa los huecos en el calendario para asegurar que todos los días están representados.
    """
    df['FechaVenta'] = pd.to_datetime(df['FechaVenta'])
    df.set_index('FechaVenta', inplace=True)
    df = df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'), fill_value=0)
    return df
