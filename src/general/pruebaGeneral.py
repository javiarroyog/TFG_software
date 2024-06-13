import numpy as np
import pandas as pd
from pyrwr.rwr import RWR
from surprise import Dataset, KNNBasic, Reader


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
    #df_indices.to_csv(ruta, index=False, sep=' ', header=False)

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

def RandomWalkRestartCoautores(df_coauthors, autor, n, c=0.15, epsilon=1e-5, max_iters=100):

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
    #df_indices.to_csv(ruta, index=False, sep=' ', header=False)

    # Convertir el DataFrame a una lista de tuplas
    #coauthor_list = df_indices.to_records(index=False).tolist()

    # Seleccionar un autor aleatorio que esté en el conjunto de coautores
    autor_indice = codigo_a_indice[autor]

    # Crear un objeto RWR
    rwr = RWR()

    # Leer el grafo de coautores
    graph_type = 'directed'
    rwr.read_graph(ruta, graph_type)

    #Parámetros:
    # -> autor_inidice: es la semilla de la que parte el algoritmo
    # -> c: la probabilidad de reiniciar el random walk (por defecto 0.15)
    # -> epsilon: la tolerancia para la convergencia del algoritmo (por defecto 1e-5)
    # -> max_iters: el número máximo de iteraciones (por defecto 100)
    r = rwr.compute(autor_indice, c=c, epsilon=epsilon, max_iters=max_iters)
    # Salida: un vector con los puntuajes de cada nodo

    # Excluir el propio autor de los resultados
    r[autor_indice] = 0

    # Ordenar los resultados por relevancia y mapear los índices de vuelta a los códigos originales
    sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
    # Mostrar los n autores más cercanos al autor semilla
    top_authors = [(indice_a_codigo[i], r[i]) for i in sorted_indices[:n]]  

    return top_authors

def getRevistas(df_journals, autor):
    return df_journals[df_journals['codigo_autor'] == autor]['codigo_revista'].tolist()

def getRevistasPeso(df_journals, autor):
    return df_journals[df_journals['codigo_autor'] == autor][['codigo_revista', 'publicaciones_normalizadas']]

def FCSurprise(df_coauthors, autor, k=5):
    #inicializar el reader
    reader = Reader(rating_scale=(df_coauthors['publicaciones_normalizadas'].min(), df_coauthors['publicaciones_normalizadas'].max()))
    print (df_coauthors['publicaciones_normalizadas'].min(), df_coauthors['publicaciones_normalizadas'].max())
    data_train = Dataset.load_from_df(df_coauthors[['codigo_autor', 'codigo_revista', 'publicaciones_normalizadas']], reader)

    trainset = data_train.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': True}
    algoritmo = KNNBasic(sim_options=sim_options)
    algoritmo.fit(trainset)

    # Convertir el autor específico a su id interno en Surprise
    autor_id = algoritmo.trainset.to_inner_uid(autor)
    
    # Obtener los k vecinos más cercanos (usuarios similares)
    vecinos = algoritmo.get_neighbors(autor_id, k=k)
    
    # Convertir los ids internos de Surprise de vuelta a los ids originales de los autores
    vecinos_ids = [algoritmo.trainset.to_raw_uid(vid) for vid in vecinos]

    # Ranking de revistas
    revistas_recomendadas = []
    for vecino in vecinos_ids:
        revistas_vecino = df_coauthors[df_coauthors['codigo_autor'] == vecino]['codigo_revista'].tolist()
        revistas_recomendadas.extend(revistas_vecino)


    # Eliminar revistas que ya ha publicado el autor
    #for revista in revistas_recomendadas:
    #    if revista in getRevistas(df_journals, autor):
    #        revistas_recomendadas.remove(revista)
    #revistas_recomendadas = list(set(revistas_recomendadas) - set(df_coauthors[df_coauthors['codigo_autor'] == autor]['codigo_revista'].tolist()))

    #Sumamos puntuacion de revistas repetidas
    revistas_recomendadas = [(revista, revistas_recomendadas.count(revista)) for revista in revistas_recomendadas]
    revistas_recomendadas = list(set(revistas_recomendadas))
    revistas_recomendadas.sort(key=lambda x: x[1], reverse=True)
    revistas_recomendadas = [revista for revista, _ in revistas_recomendadas][:10]
    
    return vecinos_ids, revistas_recomendadas
    
def FC(df_coauthors, autor, k=5):

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

def calcular_ranking_revistas (df_journals, vecinos_mas_cercanos):
    ranking_revistas = {}

    # Recorrer cada vecino y sus pesos
    for vecino, peso in vecinos_mas_cercanos:
        publicaciones_vecino = getRevistasPeso(df_journals, vecino)
    
        for _, row in publicaciones_vecino.iterrows():
            revista = row['codigo_revista']
            puntuacion_original = row['publicaciones_normalizadas']
            # La puntuación de la revista es la puntuación del vecino multiplicada por el peso del vecino
            puntuacion = peso * puntuacion_original
            #print(f"Peso Vecino: {vecino, peso}, Puntuacion: {puntuacion_original}, Puntuacion Total: {puntuacion}, Revista: {revista}")
            if revista in ranking_revistas:
                ranking_revistas[revista] += puntuacion
            else:
                ranking_revistas[revista] = puntuacion

    # Ordenar el ranking de revistas y limitar a las 10 mejores
    ranking_revistas_ordenado = sorted(ranking_revistas.items(), key=lambda x: x[1], reverse=True)[:10]

    #devolvemos tanto la revista como la puntuación
    return ranking_revistas_ordenado

    #return [revista for revista, _ in ranking_revistas_ordenado]
        
def calcular_metricas(df_test, revistas_recomendadas, autor):
    # Filtrar las publicaciones del autor en el conjunto de test
    publicaciones_test = getRevistas(df_test, autor)

    print(f"Publicaciones test: \t{publicaciones_test}")
    # Calcular verdaderos positivos, falsos positivos y falsos negativos
    tp = len([revista for revista in revistas_recomendadas if revista in publicaciones_test])
    fp = len([revista for revista in revistas_recomendadas if revista not in publicaciones_test])
    fn = len([revista for revista in publicaciones_test if revista not in revistas_recomendadas])
    
    print("Aciertos: ", tp)
    print("Errores: ", fp)
    print("Faltantes: ", fn)

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

# Seleccionar un autor aleatorio que esté en el conjunto de coautores
autor_aleatorio = df_authors['codigo_autor'].sample(n=1).iloc[0]
while autor_aleatorio not in pd.read_csv(file_path_coauthors)['codigo_autor'].values:
    autor_aleatorio = df_authors['codigo_autor'].sample(n=1).iloc[0]
    
if autor_aleatorio not in pd.read_csv(file_path_coauthors)['codigo_autor'].values:
    print(f"El autor {autor_aleatorio} no tiene coautores en el conjunto de entrenamiento.")
    exit()
k = 5
#autor_aleatorio = '0000-0002-4699-5627'

print(f"Autor aleatorio: {autor_aleatorio}")

print('\n\nFILTRADO COLABORATIVO PURO')
# Calcular los autores más cercanos FC
vecinos = FC(df_coauthors, autor_aleatorio, k)
for vecino, peso in vecinos:
    print(f"Vecino: {vecino}, Peso: {peso}")

# Calcular el ranking de revistas
ranking_revistas = calcular_ranking_revistas(df_journals, vecinos)
for revista in ranking_revistas:
    print(f"Revista: {revista}")

## Calcular las métricas de precisión, recall y F1 score
precision, recall, f1 = calcular_metricas(df_test, ranking_revistas, autor_aleatorio)
print(f"Precisión: {precision}\nRecall: {recall}\nF1 Score: {f1}")

print('\n\nRANDOM WALK WITH RESTART + FC')
vecinos2 = RandomWalkRestartCoautores(df_coauthors, autor_aleatorio,k)
for vecino, peso in vecinos2:
    print(f"Vecino: {vecino}, Peso: {peso}")

ranking_revistas2 = calcular_ranking_revistas(df_journals, vecinos2)
for revista in ranking_revistas2:
    print(f"Revista: {revista}")

precision2, recall2, f12 = calcular_metricas(df_test, ranking_revistas2, autor_aleatorio)
print(f"Precisión: {precision2}\nRecall: {recall2}\nF1 Score: {f12}")
print('\n\nRANDOM WALK WITH RESTART COMPLETO')
ranking_revistas3 = RandomWalkRestartCompleto(df_coauthors, df_journals, autor_aleatorio, 10)
for revista in ranking_revistas3:
    print(f"Revista: {revista}")


precision3, recall3, f13 = calcular_metricas(df_test, ranking_revistas3, autor_aleatorio)
print(f"Precisión: {precision3}\nRecall: {recall3}\nF1 Score: {f13}")

print('\n\nFILTRADO COLABORATIVO SURPRISE')
vecinos4, revistas4 = FCSurprise(df_journals, autor_aleatorio, k)
for vecino in vecinos4:
    print(f"Vecino: {vecino}")

for revista in revistas4:
    print(f"Revista: {revista}")

precision4, recall4, f14 = calcular_metricas(df_test, revistas4, autor_aleatorio)
print(f"Precisión: {precision4}\nRecall: {recall4}\nF1 Score: {f14}")