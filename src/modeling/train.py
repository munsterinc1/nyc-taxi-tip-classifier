# src/modeling/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Importar funciones y constantes de nuestros módulos
from src.data.dataset import load_and_prepare_data
from src.features.build_features import create_and_select_features
from src.config import (
    JAN_2020_DATA_URL,
    features,
    target_col,
    MODEL_OUTPUT_PATH
)

if __name__ == "__main__":
    print("Cargando y preprocesando datos de entrenamiento (Enero 2020)...")
    
    # Cargar y preparar datos
    taxi_initial = load_and_prepare_data(JAN_2020_DATA_URL)

    # Crear y seleccionar características
    taxi_train = create_and_select_features(taxi_initial)

    # Reducir el tamaño para la prueba 
    taxi_train = taxi_train.head(100000)

    # Preparar X e y (exacto como en el notebook)
    X_train = taxi_train[features]
    y_train = taxi_train[target_col]

    print("Entrenando el modelo RandomForest...")
    # rfc initialization and fit
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
    rfc.fit(X_train, y_train)

    # Asegurarse de que el directorio 'models' exista antes de guardar
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    print(f"Guardando el modelo entrenado en {MODEL_OUTPUT_PATH}...")
    # Guardar el modelo
    joblib.dump(rfc, MODEL_OUTPUT_PATH)
    print("Modelo guardado exitosamente.")