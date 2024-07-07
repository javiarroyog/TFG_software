import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('./procesados/training/autor-revista-numpublic_training.csv')

# Contar el número de coautorías para cada autor considerando ambas columnas
coautor_count = pd.concat([data['codigo_autor']]).value_counts()

# Crear un DataFrame con el conteo de coautorías
coautor_count_df = coautor_count.reset_index()
coautor_count_df.columns = ['codigo_autor', 'revistas']

# Limitar el conteo a 50 y agrupar los valores mayores a 50
coautor_count_df['revistas'] = coautor_count_df['revistas'].apply(lambda x: 51 if x > 50 else x)
coautor_count_grouped = coautor_count_df['revistas'].value_counts().sort_index()

# Asegurar que la categoría '50+' esté al final
coautor_count_grouped = coautor_count_grouped.reindex(list(range(1, 51)) + ['50+'], fill_value=0)

# Graficar los resultados
plt.figure(figsize=(12, 6))
coautor_count_grouped.plot(kind='bar')
plt.xlabel('Número de Revistas Diferentes en las que el autor ha publicado')
plt.ylabel('Número de Autores')
plt.title('Distribución del Número de Publicaciones en Diferentes Revistas por Autor')
plt.xticks(rotation=45)
plt.show()