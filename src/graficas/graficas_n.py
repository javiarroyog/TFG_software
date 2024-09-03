import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos de los archivos CSV
data1 = pd.read_csv('./src/resultados/n/FINAL.csv')
data2 = pd.read_csv('./src/resultados/n_autorVecino/FINAL.csv')

# Definir los valores de N y las métricas
N_values = data1['N']
# Crear gráfica para Precision
plt.figure(figsize=(8, 6))
plt.plot(N_values, data1['Precision'], label='Vecindario normal', marker='o', color='g')
plt.plot(N_values, data2['Precision'], label='Vecindario con autor', marker='o', color='r')
plt.xlabel('N')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend()
plt.grid(True)
plt.xticks(N_values)  # Asegurar que los valores de N en el eje X correspondan con los de las tablas
plt.yticks([i * 0.05 for i in range(21)])  # Establecer los valores del eje Y de 0 a 1 con incrementos de 0.05
plt.ylim(0, 0.3)  # Asegurar que el rango del eje Y vaya de 0 a 1
plt.show()

# Crear gráfica para Recall
plt.figure(figsize=(8, 6))
plt.plot(N_values, data1['Recall'], label='Vecindario normal', marker='o', color='g')
plt.plot(N_values, data2['Recall'], label='Vecindario con autor', marker='o',color='r')
plt.xlabel('N')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend()
plt.grid(True)
plt.xticks(N_values)  # Asegurar que los valores de N en el eje X correspondan con los de las tablas
plt.yticks([i * 0.05 for i in range(21)])  # Establecer los valores del eje Y de 0 a 1 con incrementos de 0.05
plt.ylim(0.2, 0.7)  # Asegurar que el rango del eje Y vaya de 0 a 1
plt.show()

# Crear gráfica para F1
plt.figure(figsize=(8, 6))
plt.plot(N_values, data1['F1'], label='Vecindario normal', marker='o',color = 'g')
plt.plot(N_values, data2['F1'], label='Vecindario con autor', marker='o',color = 'r')
plt.xlabel('N')
plt.ylabel('F1')
plt.title('F1')
plt.legend()
plt.grid(True)
plt.xticks(N_values)  # Asegurar que los valores de N en el eje X correspondan con los de las tablas
plt.yticks([i * 0.05 for i in range(21)])  # Establecer los valores del eje Y de 0 a 1 con incrementos de 0.05
plt.ylim(0, 0.3)  # Asegurar que el rango del eje Y vaya de 0 a 1
plt.show()
