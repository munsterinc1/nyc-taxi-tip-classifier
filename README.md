# nyc-taxi-tip-classifier
Tarea N° 1 Clase Desarrollo de Proyectos y Producto de Datos
# Clasificador de Propinas para Viajes en Taxi en NYC

## Descripción del Proyecto
Este proyecto implementa un clasificador de Machine Learning para predecir si un viaje de taxi en la ciudad de Nueva York recibirá una propina alta (mayor al 20% del costo del viaje). El modelo se entrena con datos de viajes de taxi de NYC y se evalúa su rendimiento a lo largo del tiempo para identificar posibles cambios en el comportamiento del modelo.

## Estructura del Proyecto
El repositorio está organizado siguiendo las buenas prácticas de ingeniería de software y ciencia de datos:
- `README.md`: Este archivo, que describe el proyecto y las instrucciones de ejecución.
- `requirements.txt`: Lista de todas las dependencias de Python necesarias.
- `data/`: Contiene los datos.
    - `raw/`: Datos originales descargados (ej. archivos Parquet mensuales).
    - `processed/`: Datos preprocesados y transformados listos para el entrenamiento y la evaluación.
- `models/`: Almacena los modelos entrenados y serializados (ej. con `joblib`).
- `notebooks/`: Contiene notebooks Jupyter/Colab para exploración de datos, prototipado y validación. Aquí encontrarás el notebook original.
- `src/`: Contiene el código fuente modularizado del proyecto.
    - `config.py`: Parámetros de configuración y rutas de archivos.
    - `data/dataset.py`: Funciones para la descarga, carga y limpieza básica de los datos.
    - `features/build_features.py`: Funciones para la ingeniería de características (creación de predictores).
    - `modeling/train.py`: Script para el entrenamiento del modelo.
    - `modeling/predict.py`: Script para la predicción y evaluación del modelo.
    - `visualization/plots.py`: (Opcional) Funciones para generar visualizaciones de resultados.

## Cómo Ejecutar el Proyecto
Para replicar este análisis, sigue los siguientes pasos:

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/munsterinc1/nyc-taxi-tip-classifier
    cd nyc-taxi-tip-classifier
    ```

2.  **Crear y Activar un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    ```

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar el Proceso (placeholder, se completará en Partes 2 y 3):**
    * Descargar y preprocesar los datos.
    * Entrenar el modelo.
    * Evaluar el modelo mensualmente.
    * Generar reportes y visualizaciones.

(Se añadirán instrucciones más detalladas aquí a medida que se desarrollen las Partes 2 y 3 del proyecto).