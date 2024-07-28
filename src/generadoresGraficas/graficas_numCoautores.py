import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('./procesados/training/autor-coautor-normalizado_filtrado.csv')

# Agrupar y contar las coautorías por cada autor
coautor_count = data.groupby('codigo_autor')['num_coautorías'].sum().reset_index()

# Crear la gráfica de barras
plt.figure(figsize=(12, 6))
plt.bar(coautor_count['codigo_autor'], coautor_count['num_coautorías'], color='skyblue')
plt.xlabel('Código de Autor')
plt.ylabel('Número de Coautorías')
plt.title('Número de Coautorías por Autor')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()