import matplotlib.pyplot as plt
import pandas as pd

# Cargar el archivo CSV
file_path = './procesados/training/autor-coautor-normalizado_filtrado.csv'
df = pd.read_csv(file_path)

# Calcular el número de coautorías por autor
num_coautorías = df.groupby('codigo_autor')['num_coautorías'].sum()

# Definir los segmentos para agrupar el número de coautorías
bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
labels = ['0-1', '1-5', '5-10', '10-20', '20-50', '50-100', '100-200', '200-500', '500-1000']

# Agrupar el número de coautorías en segmentos
segmentos = pd.cut(num_coautorías, bins=bins, labels=labels, right=False)

# Contar cuántos autores tienen coautorías en cada segmento
recuento_segmentos = segmentos.value_counts().sort_index()

# Graficar el recuento de coautorías por segmento
plt.figure(figsize=(12, 8))
recuento_segmentos.plot(kind='bar')
plt.xlabel('Número de Coautorías (Segmentos)')
plt.ylabel('Cantidad de Autores')
plt.title('Recuento de Autores por Número de Coautorías en Segmentos')
plt.tight_layout()

# Guardar la gráfica como imagen
plt.savefig('recuento_coautorías_segmentos.png')

# Mostrar la gráfica
plt.show()
