# src/features/build_features.py
import pandas as pd
from src.config import features, target_col, EPS

def create_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las características adicionales a partir del DataFrame y selecciona
    las columnas finales necesarias.
    Corresponde a la parte de 'add features' y selección/conversión final de
    la función 'preprocess' del notebook.
    """
    # Asegúrate de que las columnas de fecha/hora son de tipo datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # add features (extraído directamente de la función preprocess del notebook)
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    # drop unused columns and convert types (extraído de la última línea de preprocess en el notebook)
    df_processed = df[['tpep_dropoff_datetime'] + features + [target_col]]
    df_processed[features + [target_col]] = df_processed[features + [target_col]].astype("float32").fillna(-1.0)

    return df_processed