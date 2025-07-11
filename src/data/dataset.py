# Este módulo se encargará de la lectura del archivo parquet, la limpieza de datos y la creación de la variable objetivo.

# Lectura de datos

taxi = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet')

# Basic cleaning

df = df[df['fare_amount'] > 0].reset_index(drop=True) 

# Creación de la variable objetivo

def preprocess(df, target_col):

    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > 0.2