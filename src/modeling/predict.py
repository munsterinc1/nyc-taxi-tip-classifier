# src/modeling/predict.py
import pandas as pd
import joblib
from sklearn.metrics import f1_score
import numpy as np

# Importar funciones y constantes de nuestros módulos
from src.data.dataset import load_and_prepare_data
from src.features.build_features import create_and_select_features
from src.config import (
    FEB_2020_DATA_URL,
    APR_2020_DATA_URL,
    features,
    target_col,
    MODEL_OUTPUT_PATH
)

if __name__ == "__main__":
    print("Cargando y preprocesando datos de prueba (Febrero 2020)...")
    
    # Cargar y preparar datos
    taxi_feb_initial = load_and_prepare_data(FEB_2020_DATA_URL)
    # Crear y seleccionar características 
    taxi_test = create_and_select_features(taxi_feb_initial)

    # Preparar X e y (extraído de las llamadas a predict_proba en el notebook)
    X_test = taxi_test[features]
    y_test = taxi_test[target_col]

    print(f"Cargando el modelo desde {MODEL_OUTPUT_PATH}...")
    # Cargar el modelo 
    loaded_rfc = joblib.load(MODEL_OUTPUT_PATH)

    print("Realizando predicciones...")
    # Predicción 
    preds_test = loaded_rfc.predict_proba(X_test)
    preds_test_labels = [p[1] for p in preds_test.round()]

    print("Calculando F1-score...")
    # Cálculo del F1-score
    print(f"F1: {f1_score(y_test, preds_test_labels)}")

    print("\n--- Intentemos realizar la predicción de un solo viaje ---")
    print(taxi_test.head(1)[features].iloc[0].values)
    print(taxi_test.head(1)[target_col].iloc[0])
    print(loaded_rfc.predict_proba(taxi_test.head(1)[features].iloc[0].values.reshape(1, -1))[0][1])

    print("\n--- Motivación: calculemos el desempeño para mayo de 2020 ---")
    taxi_may_initial = load_and_prepare_data(APR_2020_DATA_URL)
    taxi_test_may = create_and_select_features(taxi_may_initial)

    preds_test_may = loaded_rfc.predict_proba(taxi_test_may[features])
    preds_test_labels_may = [p[1] for p in preds_test_may.round()]
    print(f"F1: {f1_score(taxi_test_may[target_col], preds_test_labels_may)}")