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
EPS = 1e-7

'''
# URLs de los datos (extraídas de las llamadas a pd.read_parquet en el notebook). Se dejó como texto debido a que necesitamos recorrer más meses.
JAN_2020_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet'
FEB_2020_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet'
APR_2020_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet' # Usada para 'motivación
'''
# Creación de un diccionario para las URLs de los datos mensuales

MONTH_DATA_URLS = {
    "Enero 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet',
    "Febrero 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet',
    "Marzo 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-03.parquet',
    "Abril 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet',
    "Mayo 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-05.parquet',
    "Junio 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-06.parquet',
    "Julio 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-07.parquet',
    "Agosto 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-08.parquet',
    "Septiembre 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-09.parquet',
    "Octubre 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-10.parquet',
    "Noviembre 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-11.parquet',
    "Diciembre 2020": 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-12.parquet'
}

# Ruta para guardar/cargar el modelo
MODEL_OUTPUT_PATH = "models/random_forest.joblib"