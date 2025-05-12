# src/model_training.py
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

def train_lightgbm_model(y_train, X_train):
    """
    Entrena un modelo LightGBM para la predicción de ventas.
    """
    lgb = LGBMRegressor(objective='regression', n_estimators=600, learning_rate=0.05,
                        subsample=0.8, random_state=42, n_jobs=-1)
    forecaster = ForecasterRecursiveMultiSeries(regressor=lgb, lags=[1,7,14])
    forecaster.fit(series=y_train, exog=X_train)
    return forecaster