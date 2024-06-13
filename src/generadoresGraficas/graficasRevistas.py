import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos combinados
combined_df = pd.read_csv('./procesados/training/autor-revista-normalizado.csv')

# 1. Histograma de la relación de coautorías normalizadas
plt.figure(figsize=(10, 6))
plt.hist(combined_df['publicaciones_normalizadas'], bins=20, edgecolor='black')
plt.title('Distribución de la Relación de Publicaciones en Revistas (Normalizadas)')
plt.xlabel('Relación de Publicaciones en Revistas (Normalizadas)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# 2. Diagrama de dispersión de coautorías normalizadas vs. publicaciones totales
plt.figure(figsize=(10, 6))
plt.scatter(combined_df['num_publicaciones_totales'], combined_df['publicaciones_normalizadas'], alpha=0.6)
plt.title('Publicaciones en una revista (Normalizadas) vs. Publicaciones Totales')
plt.xlabel('Número de Publicaciones Totales')
plt.ylabel('Relación de Publicaciones en una revista (Normalizadas)')
plt.grid(True)
plt.show()

# 3. Gráfico de barras de los coautores con mayor número de coautorías normalizadas
# top_coauthors = combined_df.groupby('codigo_coautor')['coautorías_normalizadas'].sum().nlargest(10)
# plt.figure(figsize=(12, 8))
# top_coauthors.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('Top 10 Coautores con Mayor Número de Coautorías Normalizadas')
# plt.xlabel('Coautor')
# plt.ylabel('Suma de Coautorías Normalizadas')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()
