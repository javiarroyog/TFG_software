import pandas as pd

# Cargar los archivos CSV
df_coauthor = pd.read_csv('./procesados/training/autor-coautor-numpublic_training.csv', index_col=0)
df_author = pd.read_csv('./procesados/training/autor-numpublic_training.csv', index_col=0)

# Renombrar columnas para consistencia
df_coauthor.rename(columns={'num_publicaciones': 'num_coautorías'}, inplace=True)
df_author.rename(columns={'publicaciones_totales': 'num_publicaciones_totales'}, inplace=True)

# Verificar las primeras filas de los dataframes
print(df_coauthor.head())
print(df_author.head())

# Unir la información de coautorías con el número total de publicaciones
combined_df = pd.merge(df_coauthor, df_author, on='codigo_autor')

# Calcular la relación entre el número de coautorías y el número total de publicaciones del autor
combined_df['coautorías_normalizadas'] = combined_df['num_coautorías'] / combined_df['num_publicaciones_totales']

# Mostrar el dataframe combinado
print(combined_df.head())

# Guardar el dataframe combinado en un nuevo archivo CSV
combined_df.to_csv('./procesados/training/autor-coautor-normalizado.csv', index=False)
