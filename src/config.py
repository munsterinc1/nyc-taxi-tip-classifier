# src/config.py

# Definición de características (exactamente como en el notebook original)
numeric_feat = [
    "pickup_weekday",
    "pickup_hour",
    'work_hours',
    "pickup_minute",
    "passenger_count",
    'trip_distance',
    'trip_time',
    'trip_speed'
]
categorical_feat = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
features = numeric_feat + categorical_feat
target_col = "high_tip"
EPS = 1e-7 # Del notebook, usado en trip_speed

# URLs de los datos (extraídas de las llamadas a pd.read_parquet en el notebook)
JAN_2020_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet'
FEB_2020_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet'
APR_2020_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet' # Usada para 'motivación'

# Ruta para guardar/cargar el modelo
MODEL_OUTPUT_PATH = "models/random_forest.joblib"