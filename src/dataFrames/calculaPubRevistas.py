import pandas as pd

# Cargar los archivos CSV
df = pd.read_csv('./procesados/training/autor-revista-normalizado.csv')

# Contar para cada revista la cantidad de publicaciones por cada autor (cada autor puede haber publicado más de una vez en esa revista)

# Agrupar por 'codigo_revista' y sumar el número de publicaciones en cada revista
publicaciones_por_revista = df.groupby('codigo_revista')['num_publicaciones_en_revista'].sum().reset_index()

# Renombrar las columnas para mayor claridad
publicaciones_por_revista.columns = ['codigo_revista', 'total_publicaciones_revista']

# Unir la información de publicaciones por revista al DataFrame original
df = df.merge(publicaciones_por_revista, on='codigo_revista', how='left')

# Nueva columna con número de publicaciones de cada autor en cada revista dividido por el total de publicaciones en esa revista
df['publicaciones_normalizadas_revista'] = df['num_publicaciones_en_revista'] / df['total_publicaciones_revista']


# Guardar el archivo CSV
df.to_csv('./procesados/training/autor-revista-biNormalizado.csv', index=False)
