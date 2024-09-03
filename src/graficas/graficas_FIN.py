import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('./src/resultados/FINAL_RWR/FIN.csv')
data2 = pd.read_csv('./src/resultados/FINAL_RWR_FC/FIN.csv')
data3 = pd.read_csv('./src/resultados/FINAL_FC/FIN.csv')

# Extraer las columnas
K = data['K']
Precision = data['Precision']
Recall = data['Recall']
F1 = data['F1']

K2 = data2['K']
Precision2 = data2['Precision']
Recall2 = data2['Recall']
F12 = data2['F1']

K3 = data3['K']
Precision3 = data3['Precision']
Recall3 = data3['Recall']
F13 = data3['F1']

# Configuración de los límites y los ticks en el eje Y para la gráfica de Precision
yticks_precision = [i/100 for i in range(1, 31, 1)]
yticks_recall = [i/100 for i in range(0, 71, 5)]
yticks_f1 = [i/100 for i in range(0, 31, 2)]

# Graficar Precision
plt.figure(figsize=(10, 5))
plt.plot(K, Precision, marker='o', linestyle='-', color='b', label='RWR')
plt.plot(K2, Precision2, marker='o', linestyle='-', color='r', label='HIBRIDO')
plt.plot(K3, Precision3, marker='x', linestyle='--', color='g', label='FC')
plt.xlabel('K')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(loc='best')  # Añadir leyenda
plt.xticks(K)
plt.yticks(yticks_precision)
plt.ylim(0, max(Precision)*1.1)
plt.show()

# Graficar Recall
plt.figure(figsize=(10, 5))
plt.plot(K, Recall, marker='o', linestyle='-', color='b', label='RWR')
plt.plot(K2, Recall2, marker='o', linestyle='-', color='r', label='HIBRIDO')
plt.plot(K3, Recall3, marker='x', linestyle='--', color='g', label='FC')
plt.xlabel('K')
plt.ylabel('Recall')
plt.grid(True)
plt.legend(loc='best')  # Añadir leyenda
plt.xticks(K)
plt.yticks(yticks_recall)
plt.ylim(0, max(Recall)*1.1)
plt.show()

# Graficar F1
plt.figure(figsize=(10, 5))
plt.plot(K, F1, marker='o', linestyle='-', color='b', label='RWR')
plt.plot(K2, F12, marker='o', linestyle='-', color='r', label='HIBRIDO')
plt.plot(K3, F13, marker='x', linestyle='--', color='g', label='FC')
plt.xlabel('K')
plt.ylabel('F1')
plt.grid(True)
plt.legend(loc='best')  # Añadir leyenda
plt.xticks(K)
plt.yticks(yticks_f1)
plt.ylim(0, max(F1)*1.1)
plt.show()
