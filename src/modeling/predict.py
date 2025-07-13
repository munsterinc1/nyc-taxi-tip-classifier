# src/modeling/predict.py
import pandas as pd
import joblib
from sklearn.metrics import f1_score
import numpy as np

'''
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
'''

# Importar funciones y constantes de nuestros módulos
from src.data.dataset import load_and_prepare_data
from src.features.build_features import create_and_select_features
from src.config import (
    features,
    target_col,
    MODEL_OUTPUT_PATH,
    MONTH_DATA_URLS # Importamos el diccionario de URLs de todos los meses
)
# Se agrega importanción para la visualización
from src.visualization.plots import plot_monthly_f1_score

'''
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

'''

def evaluate_single_month(model, data_url: str, month_name: str) -> dict:
    """
    Carga, preprocesa y evalúa el modelo para un mes específico,
    calculando el F1-score y la cantidad de ejemplos.

    Args:
        model: El modelo de Machine Learning ya cargado.
        data_url (str): La URL del archivo Parquet para el mes a evaluar.
        month_name (str): El nombre del mes (ej. "Enero 2020").

    Returns:
        dict: Un diccionario con 'mes', 'cantidad_ejemplos' y 'f1_score'.
    """
    print(f"Cargando y preprocesando datos para {month_name}...")
    
    # Cargar y preparar datos (Parte 1 del preprocess original)
    taxi_initial = load_and_prepare_data(data_url)
    # Crear y seleccionar características (Parte 2 del preprocess original)
    taxi_data_processed = create_and_select_features(taxi_initial)

    # Preparar X e y
    X = taxi_data_processed[features]
    y = taxi_data_processed[target_col]

    # Realizar predicciones
    preds = model.predict_proba(X)
    preds_labels = [p[1] for p in preds.round()]

    # Calcular F1-score
    f1 = f1_score(y, preds_labels)
    
    print(f"  F1-score para {month_name}: {f1:.4f} (Ejemplos: {len(X)})")

    return {
        "mes": month_name,
        "cantidad_ejemplos": len(X),
        "f1_score": f1
    }


if __name__ == "__main__":
    print(f"Cargando el modelo entrenado desde {MODEL_OUTPUT_PATH}...")
    loaded_rfc = joblib.load(MODEL_OUTPUT_PATH)

    results = []
    print("\n--- Evaluación mensual del modelo ---")
    for month_name, url in MONTH_DATA_URLS.items():
        # La instrucción es evaluar el modelo entrenado en enero sobre cada conjunto mensual.
        # Por lo tanto, se incluyen todos los meses aquí, incluido enero para el F1-score de entrenamiento.
        month_result = evaluate_single_month(loaded_rfc, url, month_name)
        results.append(month_result)

    print("\n--- Resumen de Resultados Mensuales ---")
    results_df = pd.DataFrame(results)
    # Aseguramos el orden de los meses para la tabla y el gráfico
    month_order = list(MONTH_DATA_URLS.keys())
    results_df['mes'] = pd.Categorical(results_df['mes'], categories=month_order, ordered=True)
    results_df = results_df.sort_values('mes')
    print(results_df.to_string(index=False))

    # Generar el gráfico de rendimiento mensual
    print("\nGenerando gráfico de rendimiento mensual...")
    plot_monthly_f1_score(results_df)