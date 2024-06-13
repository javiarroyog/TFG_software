import pandas as pd
from pyrwr.rwr import RWR


def obtener_autores_mas_cercanos(file_path_coauthors, file_path_authors, n=10, c=0.15, epsilon=1e-5, max_iters=1000):
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
    print("Hola me llamo" + autor_aleatorio_codigo)
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
# Uso de la función
file_path_coauthors = './procesados/training/autor-coautor-normalizado_filtrado.csv'
file_path_authors = './procesados/training/autor-numpublic_training_filtrado.csv'
file_path_test = './procesados/test/autor-revista-numpublic_test_filtrado.csv'
file_path_journals = './procesados/training/autor-revista-normalizado.csv'

autor_principal, top_authors = obtener_autores_mas_cercanos(file_path_coauthors, file_path_authors)

revistas_recomendadas = calcular_ranking_revistas(file_path_journals, top_authors)
print(f"Revistas recomendadas: {revistas_recomendadas}")

precision, recall, f1 = calcular_metricas(file_path_test, revistas_recomendadas, autor_principal)

print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")



