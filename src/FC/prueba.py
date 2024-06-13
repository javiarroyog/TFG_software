import numpy as np
import pandas as pd


def RWR(file_path_coauthors, file_path_authors, n=10, c=0.15, epsilon=1e-5, max_iters=1000):
    # Cargar los archivos CSV
    df_coauthors = pd.read_csv(file_path_coauthors)
    df_authors = pd.read_csv(file_path_authors)

    # Seleccionar y renombrar las columnas según el formato requerido
    df_formatted = df_coauthors[['codigo_autor', 'codigo_coautor', 'coautorías_normalizadas']]

    # Crear diccionarios para mapear códigos a índices y viceversa
    codigo_a_indice = {}
    indice_a_codigo = {}

    # Asignar un índice único a cada autor y coautor
    indice = 0
    for codigo in pd.concat([df_formatted['codigo_autor'], df_formatted['codigo_coautor']]).unique():
        codigo_a_indice[codigo] = indice
        indice_a_codigo[indice] = codigo
        indice += 1

    # Reemplazar los códigos de autor y coautor por sus índices
    df_formatted['id_autor'] = df_formatted['codigo_autor'].map(codigo_a_indice)
    df_formatted['id_coautor'] = df_formatted['codigo_coautor'].map(codigo_a_indice)

    # Seleccionar solo las columnas con índices y coautorías normalizadas
    df_indices = df_formatted[['id_autor', 'id_coautor', 'coautorías_normalizadas']]

    # Guardar el DataFrame formateado en un nuevo archivo CSV usando espacios como separadores y sin encabezados de columna
    ruta = './src/RWR/coauthors.csv'
    df_indices.to_csv(ruta, index=False, sep=' ', header=False)

    # Convertir el DataFrame a una lista de tuplas
    coauthor_list = df_indices.to_records(index=False).tolist()

    # Seleccionar un autor aleatorio que esté en el conjunto de coautores
    autores_en_coautores = df_formatted['codigo_autor'].unique()
    autor_aleatorio_codigo = pd.Series(autores_en_coautores).sample(n=1).iloc[0]
    autor_aleatorio = codigo_a_indice[autor_aleatorio_codigo]

    # Crear un objeto RWR
    rwr = RWR()

    graph_type = 'directed'
    # Leer el grafo de coautores
    rwr.read_graph(ruta, graph_type)

    r = rwr.compute(autor_aleatorio, c=c, epsilon=epsilon, max_iters=max_iters)

    # Excluir el propio autor de los resultados
    r[autor_aleatorio] = 0

    # Mostrar los autores más cercanos al autor semilla
    # Ordenar los resultados por relevancia y mapear los índices de vuelta a los códigos originales
    sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
    top_authors = [(indice_a_codigo[i], r[i]) for i in sorted_indices[:n]]  # Mostrar los n autores más cercanos

    print(f"Autor aleatorio índice: {autor_aleatorio}")
    print(f"Autor aleatorio código original: {indice_a_codigo[autor_aleatorio]}")
    print("Autores más cercanos al autor semilla:")
    for autor, score in top_authors:
        print(f"Autor: {autor}, Score: {score}")

    return autor_aleatorio_codigo, top_authors

def FC(file_path_coauthors, autor, k=5):
    # Cargar el archivo CSV de coautores
    df_coauthors = pd.read_csv(file_path_coauthors)

    # Crear un DataFrame pivote donde las filas son los autores y las columnas son los coautores con el valor de coautorías normalizadas
    pivot_df = df_coauthors.pivot(index='codigo_autor', columns='codigo_coautor', values='coautorías_normalizadas').fillna(0)

    # Inicializar la lista para almacenar los vecinos más cercanos y sus pesos
    vecinos_mas_cercanos = []

    # Verificar si el autor está en el DataFrame pivote
    if autor in pivot_df.index:
        # Ordenar los coautores por el valor de coautorías normalizadas en orden descendente
        sorted_coauthors = pivot_df.loc[autor].sort_values(ascending=False)
        
        # Seleccionar los k vecinos más cercanos (excluyendo el propio autor si aparece en la lista)
        vecinos_cercanos = sorted_coauthors.head(k).items()
        
        vecinos_mas_cercanos = [(coautor, peso) for coautor, peso in vecinos_cercanos]
    else:
        print(f"El autor {autor} no tiene coautores en el conjunto de entrenamiento.")

    return vecinos_mas_cercanos

def calcular_ranking_revistas(file_path_journals, vecinos_mas_cercanos):
    # Cargar el archivo CSV de publicaciones normalizadas en revistas
    df_journals = pd.read_csv(file_path_journals)

    # Inicializar un diccionario para almacenar el ranking de revistas
    ranking_revistas = {}

    # Recorrer cada vecino y sus pesos
    for vecino, peso in vecinos_mas_cercanos:
        # Filtrar las publicaciones del vecino
        publicaciones_vecino = df_journals[df_journals['codigo_autor'] == vecino]
        
        # Recorrer las publicaciones y actualizar el ranking de revistas
        for _, row in publicaciones_vecino.iterrows():
            revista = row['codigo_revista']
            publicaciones_normalizadas = row['publicaciones_normalizadas'] * peso
            
            if revista in ranking_revistas:
                ranking_revistas[revista] += publicaciones_normalizadas
            else:
                ranking_revistas[revista] = publicaciones_normalizadas

    # Ordenar el ranking de revistas y limitar a las 10 mejores
    ranking_revistas_ordenado = sorted(ranking_revistas.items(), key=lambda x: x[1], reverse=True)[:10]

    return [revista for revista, _ in ranking_revistas_ordenado]

def calcular_metricas(file_path_test, revistas_recomendadas, autor):
    # Cargar el archivo de test
    df_test = pd.read_csv(file_path_test)
    
    # Filtrar las publicaciones del autor en el conjunto de test
    publicaciones_test = df_test[df_test['codigo_autor'] == autor]['codigo_revista'].tolist()

    print(f"Publicaciones test: \t{publicaciones_test}")
    # Calcular verdaderos positivos, falsos positivos y falsos negativos
    tp = len([revista for revista in revistas_recomendadas if revista in publicaciones_test])
    fp = len([revista for revista in revistas_recomendadas if revista not in publicaciones_test])
    fn = len([revista for revista in publicaciones_test if revista not in revistas_recomendadas])

    # Calcular precisión, recall y F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Cargar el archivo CSV
file_path_authors = './procesados/training/autor-numpublic_training_filtrado.csv'
file_path_coauthors = './procesados/training/autor-coautor-normalizado_filtrado.csv'
file_path_journals = './procesados/training/autor-revista-normalizado.csv'
file_path_test = './procesados/test/autor-revista-numpublic_test_filtrado.csv'

df_authors = pd.read_csv(file_path_authors)
autor_aleatorio = df_authors['codigo_autor'].sample(n=1).iloc[0]
k = 5

# Inicializar listas para almacenar las métricas de todos los autores
precision_list = []
recall_list = []
f1_list = []

# Iterar sobre todos los autores
for autor in df_authors['codigo_autor']:
    # Obtener vecinos más cercanos
    vecinos = FC(file_path_coauthors, autor, k)
    
    # Calcular el ranking de revistas
    ranking_revistas = calcular_ranking_revistas(file_path_journals, vecinos)
    
    # Calcular las métricas de precisión, recall y F1 score
    precision, recall, f1 = calcular_metricas(file_path_test, ranking_revistas, autor)
    
    # Almacenar las métricas en las listas correspondientes
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# Calcular las medias de las métricas
mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)
mean_f1 = np.mean(f1_list)

# Crear un DataFrame con las métricas medias
df_metrics = pd.DataFrame({
    'Métrica': ['Precisión', 'Recall', 'F1 Score'],
    'Valor': [mean_precision, mean_recall, mean_f1]
})

# Guardar el DataFrame en un archivo CSV
df_metrics.to_csv('./resultados/FC/resultsFC_k5.csv', index=False)

print(df_metrics)