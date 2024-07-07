import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('./src/resultados/results_final_k.csv')

# Convertir la columna 'K' a string para tratarla como categorías
data['K'] = data['K'].astype(int)

# Obtener la lista de métricas que queremos graficar
metricas = ['Precision', 'Recall', 'F1']

# Calcular las medias para cada K y cada métrica
media_metricas = data.groupby('K')[metricas].mean().reset_index()

# Ordenar los valores de K
media_metricas = media_metricas.sort_values(by='K')

# Crear un gráfico de barras para cada métrica
fig, ax = plt.subplots(figsize=(14, 8))

width = 0.25  # Ancho de las barras
x = range(len(media_metricas['K']))  # Posiciones en el eje X para los valores de K

# Colores para cada métrica
colores = ['b', 'g', 'r']

for i, metrica in enumerate(metricas):
    ax.bar([p + width*i for p in x], 
           media_metricas[metrica], 
           width=width, 
           label=f'{metrica}', 
           color=colores[i % len(colores)])
        
# Configurar las etiquetas y la leyenda
ax.set_xlabel('K')
ax.set_ylabel('Valor de la métrica')
ax.set_title('Media de métricas para cada K')
ax.set_xticks([p + 0.5 * width * (len(metricas) - 1) for p in x])
ax.set_xticklabels(media_metricas['K'].astype(str))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

print(media_metricas)
