import pandas as pd

# Paso 1: Cargar los Datos
#df_test = pd.read_csv('./procesados/test/autor-revista-numpublic_test.csv',index_col=0)
#df_test = pd.read_csv('./procesados/training/autor-coautor-normalizado.csv')
df_test = pd.read_csv('./procesados/training/autor-numpublic_training.csv')
df_autores_comunes = pd.read_csv('./procesados/autores_en_training_y_en_test_y_num_publicaciones.csv',index_col=0)

# Paso 2: Obtener la Lista de Autores Comunes
autores_comunes = df_autores_comunes['codigo_autor_valido'].unique()

# Paso 3: Filtrar el Conjunto de Prueba para Mantener Solo los Autores Comunes
df_test_filtrado = df_test[df_test['codigo_autor'].isin(autores_comunes)]

print(f"Número de autores en el conjunto de prueba original: {df_test.shape[0]}")
print(f"Número de autores en el conjunto de prueba filtrado: {df_test_filtrado.shape[0]}")

# Paso 4: Guardar el DataFrame Filtrado en un Nuevo Archivo CSV
#df_test_filtrado.to_csv('./procesados/test/autor-revista-numpublic_test_filtrado.csv', index=False)
#df_test_filtrado.to_csv('./procesados/training/autor-coautor-normalizado_filtrado.csv', index=False)
df_test_filtrado.to_csv('./procesados/training/autor-numpublic_training_filtrado.csv', index=False)
