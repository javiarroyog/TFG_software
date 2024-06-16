import pandas as pd
from pyrwr.rwr import RWR


def RandomWalkRestartCompleto(df_coauthors, df_journals, autor, n, c=0.15, epsilon=1e-5, max_iters=100):
    # Renombrar columnas de df_journals para que coincidan con df_coauthors
    df_journals = df_journals.rename(columns={'codigo_revista': 'codigo_item', 'publicaciones_normalizadas': 'peso'})
    df_coauthors = df_coauthors.rename(columns={'codigo_coautor': 'codigo_item', 'coautorías_normalizadas': 'peso'})
    
    # Concatenar df_coauthors y df_journals
    df_combined = pd.concat([df_coauthors[['codigo_autor', 'codigo_item', 'peso']], df_journals[['codigo_autor', 'codigo_item', 'peso']]], ignore_index=True)

    # Crear diccionarios para mapear códigos a índices y viceversa
    codigo_a_indice = {}
    indice_a_codigo = {}

    # Asignar un índice único a cada autor, item y revista
    indice = 0
    for codigo in pd.concat([df_combined['codigo_autor'], df_combined['codigo_item']]).unique():
        codigo_a_indice[codigo] = indice
        indice_a_codigo[indice] = codigo
        indice += 1

    # Reemplazar los códigos de autor y item/revista por sus índices
    df_combined['id_autor'] = df_combined['codigo_autor'].map(codigo_a_indice)
    df_combined['id_item'] = df_combined['codigo_item'].map(codigo_a_indice)

    # Seleccionar solo las columnas con índices y pesos
    df_indices = df_combined[['id_autor', 'id_item', 'peso']]

    # Guardar el DataFrame formateado en un nuevo archivo CSV usando espacios como separadores y sin encabezados de columna
    ruta = './src/RWR/combined.csv'
    df_indices.to_csv(ruta, index=False, sep=' ', header=False)

    # Seleccionar el índice del autor
    autor_indice = codigo_a_indice[autor]

    # Crear un objeto RWR
    rwr = RWR()

    # Leer el grafo combinado
    graph_type = 'directed'
    rwr.read_graph(ruta, graph_type)

    # Parámetros:
    # -> autor_indice: es la semilla de la que parte el algoritmo
    # -> c: la probabilidad de reiniciar el random walk (por defecto 0.15)
    # -> epsilon: la tolerancia para la convergencia del algoritmo (por defecto 1e-5)
    # -> max_iters: el número máximo de iteraciones (por defecto 100)
    r = rwr.compute(autor_indice, c=c, epsilon=epsilon, max_iters=max_iters)
    # Salida: un vector con los puntuajes de cada nodo

    # Excluir el propio autor de los resultados
    r[autor_indice] = 0

    # Ordenar los resultados por relevancia y mapear los índices de vuelta a los códigos originales
    sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
    top_items = [(indice_a_codigo[i], r[i]) for i in sorted_indices]

    # Filtrar solo los items que son revistas
    revistas = set(df_journals['codigo_item'])
    top_revistas = [(item) for item, peso in top_items if item in revistas]

    return top_revistas[:n]


def calcular_metricas(file_path_test, revistas_recomendadas, autor):
    # Cargar el archivo de test
    df_test = pd.read_csv(file_path_test)
    
    # Filtrar las publicaciones del autor en el conjunto de test
    publicaciones_test = df_test[df_test['codigo_autor'] == autor]['codigo_revista'].tolist()

    # Calcular verdaderos positivos, falsos positivos y falsos negativos
    tp = len([revista for revista in revistas_recomendadas if revista in publicaciones_test])
    fp = len([revista for revista in revistas_recomendadas if revista not in publicaciones_test])
    fn = len([revista for revista in publicaciones_test if revista not in revistas_recomendadas])

    # Calcular precisión, recall y F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Cargar los archivo CSV
# Archivo para saber el número de publicaciones de cada autor
file_path_authors = './procesados/training/autor-numpublic_training_filtrado.csv'
# Archivo para saber los coautores de cada autor
file_path_coauthors = './procesados/training/autor-coautor-normalizado_filtrado.csv'
# Archivo para saber las revistas en las que ha publicado cada autor
file_path_journals = './procesados/training/autor-revista-normalizado.csv'
# Archivo de test para evaluar las recomendaciones
file_path_test = './procesados/test/autor-revista-numpublic_test_filtrado.csv'

# Pasamos los archivos a dataframes
df_authors = pd.read_csv(file_path_authors)
df_coauthors = pd.read_csv(file_path_coauthors)
df_journals = pd.read_csv(file_path_journals)
df_test = pd.read_csv(file_path_test)

# Inicializar una lista para almacenar las métricas
metricas_totales = []

# Lista de valores para el parámetro c
valores_e= [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

for e in valores_e:
    metricas = []
    autores_aleatorios = []

    while len(autores_aleatorios) < 50:
        autor_aleatorio = df_authors['codigo_autor'].sample(n=1).iloc[0]
        if autor_aleatorio in df_coauthors['codigo_autor'].values:
            autores_aleatorios.append(autor_aleatorio)

    num_vecinos = 5
    num_revistas = 20

    for autor in autores_aleatorios:
        revistas = RandomWalkRestartCompleto(df_coauthors, df_journals, autor, num_vecinos, 0.15, e)
        precision, recall, f1 = calcular_metricas(file_path_test, revistas, autor)
        metricas.append((precision, recall, f1))

    # Crear un DataFrame para las métricas y guardarlas en un archivo CSV
    df_metricas = pd.DataFrame(metricas, columns=['precision', 'recall', 'f1'])
    df_metricas.to_csv(f'./src/RWR/metricas_epsilon_{e}.csv', index=False)
    metricas_totales.append((e, df_metricas['precision'].mean(), df_metricas['recall'].mean(), df_metricas['f1'].mean()))

# Guardar las métricas promedio en un archivo CSV
df_medias_totales = pd.DataFrame(metricas_totales, columns=['epsilon', 'precision_media', 'recall_media', 'f1_media'])
df_medias_totales.to_csv('./src/RWR/medias_totales_epsilon.csv', index=False)

print("Métricas calculadas y guardadas en archivos CSV.")
