import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
df = pd.read_csv('./procesados/training/autor-numpublic_training_filtrado.csv')


# Crear categorías en bloques de 5, y agrupar los valores mayores de 100
bins = list(range(0, 105, 5)) + [float('inf')]
labels = [f'{i}-{i+5}' for i in range(0, 100, 5)] + ['100+']
df['publicaciones_bloques'] = pd.cut(df['publicaciones_totales'], bins=bins, labels=labels, right=False)

# Contar el número de autores para cada bloque
publicaciones_count = df['publicaciones_bloques'].value_counts().sort_index()

# Crear la gráfica
plt.figure(figsize=(12, 6))
publicaciones_count.plot(kind='bar')
plt.xlabel('Número de publicaciones')
plt.ylabel('Número de autores')
plt.title('Distribución de autores por número de publicaciones')
plt.xticks(rotation=45)
plt.show()