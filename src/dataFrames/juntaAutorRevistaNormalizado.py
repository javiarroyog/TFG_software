import pandas as pd

# Cargar los archivos CSV
df_author = pd.read_csv('./procesados/training/autor-numpublic_training.csv', index_col=0)
df_magazine = pd.read_csv('./procesados/training/autor-revista-numpublic_training.csv', index_col=0)

# Renombrar columnas para consistencia
df_author.rename(columns={'publicaciones_totales': 'num_publicaciones_totales'}, inplace=True)
df_magazine.rename(columns={'num_publicaciones': 'num_publicaciones_en_revista'}, inplace=True)

# Verificar las primeras filas de los dataframes
print(df_author.head())
print(df_magazine.head())

# Unir la información de coautorías con el número total de publicaciones
combined_df = pd.merge(df_author, df_magazine, on='codigo_autor')

# Calcular la relación entre el número d  e coautorías y el número total de publicaciones del autor
combined_df['publicaciones_normalizadas'] = combined_df['num_publicaciones_en_revista'] / combined_df['num_publicaciones_totales']

# Mostrar el dataframe combinado
print(combined_df.head())

# Guardar el dataframe combinado en un nuevo archivo CSV
combined_df.to_csv('./procesados/training/autor-revista-normalizado.csv', index=False)
