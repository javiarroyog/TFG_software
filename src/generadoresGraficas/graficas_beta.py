import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('./src/resultados/r/comparativa.csv')

# Filtrar los datos según el valor de beta
data_no_beta = data[data['beta'] == '-']
data_with_beta = data[data['beta'] == '0.7']

# Crear la gráfica de líneas para cada métrica
metrics = ['Precision', 'Recall', 'F1']
for metric in metrics:
    plt.figure()
    plt.plot(data_no_beta['k'], data_no_beta[metric], label=f'Sin beta ({metric})', marker='o')
    plt.plot(data_with_beta['k'], data_with_beta[metric], label=f'Con beta ({metric})', marker='o')
    plt.xlabel('K')
    plt.ylabel(metric)
    plt.title(f'{metric} para diferentes valores de K')
    plt.legend()
    plt.grid(True)
    plt.show()