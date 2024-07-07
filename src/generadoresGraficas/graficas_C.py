import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar los datos del archivo CSV
file_path = './src/resultados/c/FINAL_C045.csv'  # Reemplaza con la ruta a tu archivo CSV
data = pd.read_csv(file_path)


# Crear una gráfica para cada columna con submuestreo
columns = data.columns
subsample_rate = 10  # Muestra solo 1 de cada 10 datos

for column in columns:
    plt.figure(figsize=(10, 5))
    plt.plot(data[column][::subsample_rate], marker='o')
    plt.title(f'Gráfica de {column} (Submuestreo cada {subsample_rate} valores)')
    plt.xlabel('Índice')
    plt.ylabel(column)
    plt.grid(True)
    plt.show()