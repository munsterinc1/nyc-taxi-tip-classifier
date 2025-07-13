# src/data/dataset.py
import pandas as pd
from src.config import target_col

def load_and_prepare_data(url: str) -> pd.DataFrame:
    """
    Carga los datos de un archivo Parquet, realiza la limpieza básica
    y crea la variable objetivo 'high_tip'.
    Corresponde a las partes de descarga, limpieza básica y creación de la variable objetivo
    de la función 'preprocess' del notebook.
    """
    # Carga de datos
    taxi_df = pd.read_parquet(url)

    # Basic cleaning (extraído directamente de la función preprocess del notebook)
    df = taxi_df[taxi_df['fare_amount'] > 0].reset_index(drop=True)

    # add target (extraído directamente de la función preprocess del notebook)
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > 0.2

    # convert target to int32 for efficiency (it's just 0s and 1s)
    df[target_col] = df[target_col].astype("int32")

    return df