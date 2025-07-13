# src/visualization/plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_monthly_f1_score(results_df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Genera un gráfico de línea mostrando el F1-score del modelo a lo largo de los meses.
    El gráfico se guarda en un directorio especificado.

    Args:
        results_df (pd.DataFrame): DataFrame con las columnas 'mes' y 'f1_score'.
        output_dir (str): Directorio donde se guardará la figura.
    """
    # Asegura que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(14, 7)) # Aumentamos el tamaño para más meses y mejor legibilidad
    sns.lineplot(data=results_df, x="mes", y="f1_score", marker='o', linestyle='-')
    plt.title('Rendimiento del Modelo (F1-score) por Mes (2020)')
    plt.xlabel('Mes')
    plt.ylabel('F1-score')
    plt.grid(True, linestyle='--', alpha=0.7) # Añadimos un estilo de rejilla
    plt.xticks(rotation=45, ha='right') # Rota las etiquetas del eje X para mayor legibilidad
    plt.ylim(0, 1) # Establece límites para el F1-score de 0 a 1 para mejor comparación
    plt.tight_layout() # Ajusta el layout para evitar el solapamiento de etiquetas
    
    plot_path = os.path.join(output_dir, "f1_score_monthly_performance.png")
    plt.savefig(plot_path)
    print(f"Gráfico guardado en: {plot_path}")
    plt.show()