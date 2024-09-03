import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('./src/resultados/FINAL_RWR/FIN.csv')

# Extraer las columnas
K = data['K']
Precision = data['Precision']
Recall = data['Recall']
F1 = data['F1']

# Configuración de los límites y los ticks en el eje Y
yticks_precision = [i/100 for i in range(0, 31, 2)]
yticks_recall = [i/100 for i in range(0, 71, 5)]
yticks_f1 = [i/100 for i in range(0, 31, 2)]

# Graficar Precision
plt.figure(figsize=(10, 5))
plt.plot(K, Precision, marker='o', linestyle='-', color='b')
plt.xlabel('K')
plt.ylabel('Precision')
plt.grid(True)
plt.xticks(K)
plt.yticks(yticks_precision)
plt.ylim(0, max(Precision)*1.1)
plt.show()

# Graficar Recall
plt.figure(figsize=(10, 5))
plt.plot(K, Recall, marker='o', linestyle='-', color='g')
plt.xlabel('K')
plt.ylabel('Recall')
plt.grid(True)
plt.xticks(K)
plt.yticks(yticks_recall)
plt.ylim(0, max(Recall)*1.1)
plt.show()

# Graficar F1
plt.figure(figsize=(10, 5))
plt.plot(K, F1, marker='o', linestyle='-', color='r')
plt.xlabel('K')
plt.ylabel('F1')
plt.grid(True)
plt.xticks(K)
plt.yticks(yticks_f1)
plt.ylim(0, max(F1)*1.1)
plt.show()