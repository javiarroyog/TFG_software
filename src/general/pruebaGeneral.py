import numpy as np
import pandas as pd
from pyrwr.rwr import RWR
from surprise import Dataset, KNNBasic, Reader


#Función para añadir los arcos Revista->Autor y formatear el grafo según PYRWR
def formateaPYRWRCompleto (df_coauthors, df_journals, df_journals2, r):
    # Renombrar columnas de df_journals para que coincidan con df_coauthors
    df_journals = df_journals.rename(columns={'codigo_revista': 'codigo_item', 'publicaciones_normalizadas': 'peso'})
    df_journals2 = df_journals2.rename(columns={'codigo_revista': 'codigo_autor','codigo_autor': 'codigo_item', 'publicaciones_normalizadas_revista': 'peso'})
    df_coauthors = df_coauthors.rename(columns={'codigo_coautor': 'codigo_item', 'coautorías_normalizadas': 'peso'})
    
    #se multiplica por un factor r (beta) los pesos de las revistas y por 1-r los pesos de los coautores
    if (r != 0):
        df_journals['peso'] = df_journals['peso'] * r
        df_coauthors['peso'] = df_coauthors['peso'] * (1-r)

    # Concatenar df_coauthors y df_journals
    df_combined = pd.concat([df_coauthors[['codigo_autor', 'codigo_item', 'peso']], df_journals[['codigo_autor', 'codigo_item', 'peso']], df_journals2[['codigo_autor', 'codigo_item', 'peso']]], ignore_index=True)

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

    return indice_a_codigo, codigo_a_indice, ruta

#Función para formatear el grafo de coautorías según PYRWR
def formateaPYRWRSimple (df_coauthors):

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

    # Seleccionar solo las columnas con índices y pesos
    df_indices = df_formatted[['id_autor', 'id_coautor', 'coautorías_normalizadas']]

    # Guardar el DataFrame formateado en un nuevo archivo CSV usando espacios como separadores y sin encabezados de columna
    ruta = './src/RWR/combined2.csv'
    df_indices.to_csv(ruta, index=False, sep=' ', header=False)

    return indice_a_codigo, codigo_a_indice, ruta

#Función para ejecutar el algoritmo Random Walk with Restart en el grafo completo (Coautorías y revistas)
def RandomWalkRestartCompleto(codigo_a_indice, indice_a_codigo, ruta, df_journals, autor, n, c=0.15, epsilon=1e-9, max_iters=100):
    # Seleccionar el índice del autor
    autor_indice = codigo_a_indice[autor]

    # Crear un objeto RWR
    rwr = RWR()

    # Leer el grafo combinado
    graph_type = 'directed'
    rwr.read_graph(ruta, graph_type)

    r = rwr.compute(autor_indice, c=c, epsilon=epsilon, max_iters=max_iters)

    # Excluir el propio autor de los resultados
    r[autor_indice] = 0

    # Ordenar los resultados por relevancia y mapear los índices de vuelta a los códigos originales
    sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
    top_items = [(indice_a_codigo[i]) for i in sorted_indices]

    # Filtrar solo los items que son revistas
    revistas = df_journals['codigo_revista'].unique()
    top_revistas = [(item) for item in top_items if item in revistas]

    return top_revistas[:n]

#Función para ejecutar el algoritmo Random Walk with Restart en el grafo de coautorías
def RandomWalkRestartCoautores(codigo_a_indice,indice_a_codigo, ruta, autor, n, c=0.15, epsilon=1e-9, max_iters=100):
    
    if autor not in codigo_a_indice:
        print(f"El autor {autor} no tiene coautores en el conjunto de entrenamiento.")
        return []
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
    #r[autor_indice] = 0

    #Incluir el propio autor en los resultados
    r[autor_indice] = 1

    # Ordenar los resultados por relevancia y mapear los índices de vuelta a los códigos originales
    sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
    # Mostrar los n autores más cercanos al autor semilla
    top_authors = [(indice_a_codigo[i], r[i]) for i in sorted_indices[:n]]  

    return top_authors

#Función para obtener las revistas en las que ha publicado un autor
def getRevistas(df_journals, autor):
    return df_journals[df_journals['codigo_autor'] == autor]['codigo_revista'].tolist()

#Función para obtener las revistas en las que ha publicado un autor con su puntuación
def getRevistasPeso(df_journals, autor):
    return df_journals[df_journals['codigo_autor'] == autor][['codigo_revista', 'publicaciones_normalizadas']]

# Actualmente no lo usaremos (Para trabajo futuro)
def FCSurprise(df_coauthors, autor, k=5, num_revistas=10):
    #inicializar el reader
    reader = Reader(rating_scale=(df_coauthors['publicaciones_normalizadas'].min(), df_coauthors['publicaciones_normalizadas'].max()))
    #print (df_coauthors['publicaciones_normalizadas'].min(), df_coauthors['publicaciones_normalizadas'].max())
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
    revistas_recomendadas = [revista for revista, _ in revistas_recomendadas]
    
    return vecinos_ids,revistas_recomendadas[:num_revistas]

#Función para obtener los vecinos más cercanos de un autor en el grafo de coautorías
def FC(df_coauthors, autor, k=30):

    # Crear un DataFrame pivote (auxiliar) donde las filas son los autores y las columnas son los coautores con el valor de coautorías normalizadas
    pivot_df = df_coauthors.pivot(index='codigo_autor', columns='codigo_coautor', values='coautorías_normalizadas').fillna(0)

    # Inicializar la lista para almacenar los vecinos más cercanos y sus pesos
    vecinos_mas_cercanos = []

    # Verificar si el autor al que vamos a recomendar está en el DataFrame pivote
    if autor in pivot_df.index:
        # Ordenar los coautores por el valor de coautorías normalizadas en orden descendente
        sorted_coauthors = pivot_df.loc[autor].sort_values(ascending=False)
        
        vecinos_cercanos = sorted_coauthors.head(k).items()

        # Añadimos al autor como su vecino con peso 1
        vecinos_cercanos = [(autor, 1)] + list(vecinos_cercanos)

        # Almacenar los vecinos más cercanos y sus pesos en la lista
        vecinos_mas_cercanos = [(coautor, peso) for coautor, peso in vecinos_cercanos]
    else:
        print(f"El autor {autor} no tiene coautores en el conjunto de entrenamiento.")

    return vecinos_mas_cercanos

#Función para calcular el ranking de revistas recomendadas para un autor
def calcular_ranking_revistas (df_journals, vecinos_mas_cercanos, num_revistas=10):
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
    ranking_revistas_ordenado = sorted(ranking_revistas.items(), key=lambda x: x[1], reverse=True)

    return [revista for revista, _ in ranking_revistas_ordenado][:num_revistas]

#Función para calcular las métricas de evaluación (precisión, recall y F1 score)
def calcular_metricas(df_test, revistas_recomendadas, autor):
    # Filtrar las publicaciones del autor en el conjunto de test
    publicaciones_test = getRevistas(df_test, autor)

    #print(f"Publicaciones test: \t{publicaciones_test}")
    #print(f"Publicaciones recomendadas: \t{revistas_recomendadas}")
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

#Función para ejecutar las pruebas finales de los algoritmos de recomendación
def prueba_final (modelo, df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios, num_vecinos, num_revistas ,path_resultados ,c=0.15 ,e=1e-9, r=0):
    # Inicializar listas para almacenar las métricas de cada algoritmo
    metricas_fc = []
    metricas_rwr_fc = []
    metricas_rwr_completo = []
    

    if (num_revistas == 0):
        dinamico = True
    else:
        dinamico = False
    
    indice_a_codigo1, codigo_a_indice1, ruta1 = formateaPYRWRCompleto(df_coauthors, df_journals,df_journals2,r)
    indice_a_codigo2, codigo_a_indice2, ruta2 = formateaPYRWRSimple(df_coauthors)

    for autor in autores_aleatorios:
        print(f"Calculando métricas para el autor: {autor}")

        # Calcular el número de revistas a recomendar si es dinámico
        if (dinamico == True):
            num_revistas = len(getRevistas(df_journals, autor))
            print(f"Numero de revistas: {num_revistas}")
        
        # Filtrado Colaborativo Puro
        if modelo == 'FC':
            
            vecinos_fc = FC(df_coauthors, autor, num_vecinos)

            k = num_revistas
            ranking_revistas_fc = calcular_ranking_revistas(df_journals, vecinos_fc, num_revistas)
            precision_fc, recall_fc, f1_fc = calcular_metricas(df_test, ranking_revistas_fc, autor)
            metricas_fc.append((k, precision_fc, recall_fc, f1_fc))

            k=3
            ranking_revistas_fc3 = ranking_revistas_fc[:k]
            precision_fc3, recall_fc3, f1_fc3 = calcular_metricas(df_test, ranking_revistas_fc3, autor)
            metricas_fc.append((k, precision_fc3, recall_fc3, f1_fc3))

            k=5
            ranking_revistas_fc5 = ranking_revistas_fc[:k]
            precision_fc5, recall_fc5, f1_fc5 = calcular_metricas(df_test, ranking_revistas_fc5, autor)
            metricas_fc.append((k, precision_fc5, recall_fc5, f1_fc5))

            k=10
            ranking_revistas_fc10 = ranking_revistas_fc[:k]
            precision_fc10, recall_fc10, f1_fc10 = calcular_metricas(df_test, ranking_revistas_fc10, autor)
            metricas_fc.append((k, precision_fc10, recall_fc10, f1_fc10))

            k=15
            ranking_revistas_fc15 = ranking_revistas_fc[:k]
            precision_fc15, recall_fc15, f1_fc15 = calcular_metricas(df_test, ranking_revistas_fc15, autor)
            metricas_fc.append((k, precision_fc15, recall_fc15, f1_fc15))

            k=20
            ranking_revistas_fc20 = ranking_revistas_fc[:k]
            precision_fc20, recall_fc20, f1_fc20 = calcular_metricas(df_test, ranking_revistas_fc20, autor)
            metricas_fc.append((k, precision_fc20, recall_fc20, f1_fc20))

            k=25
            ranking_revistas_fc25 = ranking_revistas_fc[:k]
            precision_fc25, recall_fc25, f1_fc25 = calcular_metricas(df_test, ranking_revistas_fc25, autor)
            metricas_fc.append((k, precision_fc25, recall_fc25, f1_fc25))


        ### Random Walk with Restart + FC

        # Random Walk with Restart Completo
        if modelo == 'RWR_COMPLETO':
            k = num_revistas
            ranking_revistas_rwr_completo = RandomWalkRestartCompleto(codigo_a_indice1, indice_a_codigo1, ruta1, df_journals, autor, num_revistas,c,e)
            precision_rwr_completo, recall_rwr_completo, f1_rwr_completo = calcular_metricas(df_test, ranking_revistas_rwr_completo, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo, recall_rwr_completo, f1_rwr_completo))

            k=3
            ranking_revistas_rwr_completo3 = ranking_revistas_rwr_completo[:k]
            precision_rwr_completo3, recall_rwr_completo3, f1_rwr_completo3 = calcular_metricas(df_test, ranking_revistas_rwr_completo3, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo3, recall_rwr_completo3, f1_rwr_completo3))

            k=5
            ranking_revistas_rwr_completo5 = ranking_revistas_rwr_completo[:k]
            precision_rwr_completo5, recall_rwr_completo5, f1_rwr_completo5 = calcular_metricas(df_test, ranking_revistas_rwr_completo5, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo5, recall_rwr_completo5, f1_rwr_completo5))

            k=10
            ranking_revistas_rwr_completo10 = ranking_revistas_rwr_completo[:k]
            precision_rwr_completo10, recall_rwr_completo10, f1_rwr_completo10 = calcular_metricas(df_test, ranking_revistas_rwr_completo10, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo10, recall_rwr_completo10, f1_rwr_completo10))

            k=15
            ranking_revistas_rwr_completo15 = ranking_revistas_rwr_completo[:k]
            precision_rwr_completo15, recall_rwr_completo15, f1_rwr_completo15 = calcular_metricas(df_test, ranking_revistas_rwr_completo15, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo15, recall_rwr_completo15, f1_rwr_completo15))

            k=20
            ranking_revistas_rwr_completo20 = ranking_revistas_rwr_completo[:k]
            precision_rwr_completo20, recall_rwr_completo20, f1_rwr_completo20 = calcular_metricas(df_test, ranking_revistas_rwr_completo20, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo20, recall_rwr_completo20, f1_rwr_completo20))

            k=25
            ranking_revistas_rwr_completo25 = ranking_revistas_rwr_completo[:k]
            precision_rwr_completo25, recall_rwr_completo25, f1_rwr_completo25 = calcular_metricas(df_test, ranking_revistas_rwr_completo25, autor)
            metricas_rwr_completo.append((k,precision_rwr_completo25, recall_rwr_completo25, f1_rwr_completo25))

        if modelo == 'RWR_FC':
            vecinos_rwr_fc = RandomWalkRestartCoautores(codigo_a_indice2, indice_a_codigo2, ruta2, autor, num_vecinos)
            
            k = num_revistas
            ranking_revistas_rwr_fc = calcular_ranking_revistas(df_journals, vecinos_rwr_fc, num_revistas)
            precision_rwr_fc, recall_rwr_fc, f1_rwr_fc = calcular_metricas(df_test, ranking_revistas_rwr_fc, autor)
            metricas_rwr_fc.append((1, precision_rwr_fc, recall_rwr_fc, f1_rwr_fc))
    
            k=3
            ranking_revistas_rwr_fc3 = ranking_revistas_rwr_fc[:k]
            precision_rwr_fc3, recall_rwr_fc3, f1_rwr_fc3 = calcular_metricas(df_test, ranking_revistas_rwr_fc3, autor)
            metricas_rwr_fc.append((k, precision_rwr_fc3, recall_rwr_fc3, f1_rwr_fc3))
            
            k=5
            ranking_revistas_rwr_fc5 = ranking_revistas_rwr_fc[:k]
            precision_rwr_fc5, recall_rwr_fc5, f1_rwr_fc5 = calcular_metricas(df_test, ranking_revistas_rwr_fc5, autor)
            metricas_rwr_fc.append((k, precision_rwr_fc5, recall_rwr_fc5, f1_rwr_fc5))
            k=10
            ranking_revistas_rwr_fc10 = ranking_revistas_rwr_fc[:k]
            precision_rwr_fc10, recall_rwr_fc10, f1_rwr_fc10 = calcular_metricas(df_test, ranking_revistas_rwr_fc10, autor)
            metricas_rwr_fc.append((k, precision_rwr_fc10, recall_rwr_fc10, f1_rwr_fc10))
            k=15
            ranking_revistas_rwr_fc15 = ranking_revistas_rwr_fc[:k]
            precision_rwr_fc15, recall_rwr_fc15, f1_rwr_fc15 = calcular_metricas(df_test, ranking_revistas_rwr_fc15, autor)
            metricas_rwr_fc.append((k, precision_rwr_fc15, recall_rwr_fc15, f1_rwr_fc15))

            k=20
            ranking_revistas_rwr_fc20 = ranking_revistas_rwr_fc[:k]
            precision_rwr_fc20, recall_rwr_fc20, f1_rwr_fc20 = calcular_metricas(df_test, ranking_revistas_rwr_fc20, autor)
            metricas_rwr_fc.append((k, precision_rwr_fc20, recall_rwr_fc20, f1_rwr_fc20))
            
            k=25
            ranking_revistas_rwr_fc25 = ranking_revistas_rwr_fc[:k]
            precision_rwr_fc25, recall_rwr_fc25, f1_rwr_fc25 = calcular_metricas(df_test, ranking_revistas_rwr_fc25, autor)
            metricas_rwr_fc.append((k, precision_rwr_fc25, recall_rwr_fc25, f1_rwr_fc25))
        
        # Para enfoque dinámico
        if (dinamico == True):
            num_revistas = 0
        

    if modelo == "FC":
        df_metricas_fc = pd.DataFrame(metricas_fc, columns=['K', 'Precision', 'Recall', 'F1'])
        
        # Calcular las medias de las métricas para cada K y guardarlas en un archivo CSV
        # k = 3 por separado
        df_metricas_fc_aux = df_metricas_fc[df_metricas_fc['K'] == 3]
        medias_metricas = {
            'K': 3,
            'Precision': [df_metricas_fc_aux['Precision'].mean()],
            'Recall': [df_metricas_fc_aux['Recall'].mean()],
            'F1': [df_metricas_fc_aux['F1'].mean()]
        }
        #
        ## Convertir a DataFrame
        df_medias_metricas = pd.DataFrame(medias_metricas)
        #
        ## Guardar las medias en un archivo CSV
        path_resultados2 = path_resultados + '3.csv'
        df_medias_metricas.to_csv(path_resultados2, index=False, mode='a', header=False)

        for i in range(5,31,5):
            df_metricas_fc_aux = df_metricas_fc[df_metricas_fc['K'] == i]
            medias_metricas = {
                'K': [i],
                'Precision': [df_metricas_fc_aux['Precision'].mean()],
                'Recall': [df_metricas_fc_aux['Recall'].mean()],
                'F1': [df_metricas_fc_aux['F1'].mean()]
            }
            #
            ## Convertir a DataFrame
            df_medias_metricas = pd.DataFrame(medias_metricas)
            #
            ## Guardar las medias en un archivo CSV
            path_resultados2 = path_resultados + str(i) + '.csv'
            df_medias_metricas.to_csv(path_resultados2, index=False, mode='a', header=False)
        
    if modelo == "RWR_COMPLETO":
        # Convertir las métricas a DataFrames
        df_metricas_rwr_completo = pd.DataFrame(metricas_rwr_completo, columns=['K', 'Precision', 'Recall', 'F1'])
        
        # Calcular las medias de las métricas para cada K y guardarlas en un archivo CSV
        # k = 3 por separado
        df_metricas_rwr_completo_aux = df_metricas_rwr_completo[df_metricas_rwr_completo['K'] == 3]
        medias_metricas = {
            'K': 3,
            'Precision': [df_metricas_rwr_completo_aux['Precision'].mean()],
            'Recall': [df_metricas_rwr_completo_aux['Recall'].mean()],
            'F1': [df_metricas_rwr_completo_aux['F1'].mean()]
        }
        #
        ## Convertir a DataFrame
        df_medias_metricas = pd.DataFrame(medias_metricas)
        #
        ## Guardar las medias en un archivo CSV
        path_resultados2 = path_resultados + '3.csv'
        df_medias_metricas.to_csv(path_resultados2, index=False, mode='a', header=False)

        for i in range(5,31,5):
            df_metricas_rwr_completo_aux = df_metricas_rwr_completo[df_metricas_rwr_completo['K'] == i]
            medias_metricas = {
                'K': [i],
                'Precision': [df_metricas_rwr_completo_aux['Precision'].mean()],
                'Recall': [df_metricas_rwr_completo_aux['Recall'].mean()],
                'F1': [df_metricas_rwr_completo_aux['F1'].mean()]
            }
            #
            ## Convertir a DataFrame
            df_medias_metricas = pd.DataFrame(medias_metricas)
            #
            ## Guardar las medias en un archivo CSV
            path_resultados2 = path_resultados + str(i) + '.csv'
            df_medias_metricas.to_csv(path_resultados2, index=False, mode='a', header=False)

    if modelo == "RWR_FC":
        df_metricas_rwr_fc = pd.DataFrame(metricas_rwr_fc, columns=['K', 'Precision', 'Recall', 'F1'])
        # Calcular las medias de las métricas para cada K y guardarlas en un archivo CSV
        # k = 3 por separado
        df_metricas_rwr_fc_aux = df_metricas_rwr_fc[df_metricas_rwr_fc['K'] == 3]
        medias_metricas = {
            'K': 3,
            'Precision': [df_metricas_rwr_fc_aux['Precision'].mean()],
            'Recall': [df_metricas_rwr_fc_aux['Recall'].mean()],
            'F1': [df_metricas_rwr_fc_aux['F1'].mean()]
        }
        #
        ## Convertir a DataFrame
        df_medias_metricas = pd.DataFrame(medias_metricas)
        #
        ## Guardar las medias en un archivo CSV
        path_resultados2 = path_resultados + '3.csv'
        df_medias_metricas.to_csv(path_resultados2, index=False, mode='a', header=False)

        for i in range(5,31,5):
            df_metricas_rwr_fc_aux = df_metricas_rwr_fc[df_metricas_rwr_fc['K'] == i]
            medias_metricas = {
                'K': [i],
                'Precision': [df_metricas_rwr_fc_aux['Precision'].mean()],
                'Recall': [df_metricas_rwr_fc_aux['Recall'].mean()],
                'F1': [df_metricas_rwr_fc_aux['F1'].mean()]
            }
            #
            ## Convertir a DataFrame
            df_medias_metricas = pd.DataFrame(medias_metricas)
            #
            ## Guardar las medias en un archivo CSV
            path_resultados2 = path_resultados + str(i) + '.csv'
            df_medias_metricas.to_csv(path_resultados2, index=False, mode='a', header=False)
        
    print("FIN")

################################################################################
## MAIN
################################################################################
# Cargar los archivo CSV
# Archivo para saber el número de publicaciones de cada autor
file_path_authors = './procesados/training/autor-numpublic_training_filtrado.csv'
# Archivo para saber los coautores de cada autor
file_path_coauthors = './procesados/training/autor-coautor-normalizado_filtrado.csv'
# Archivo para saber las revistas en las que ha publicado cada autor
file_path_journals = './procesados/training/autor-revista-normalizado.csv'
# Archivo para saber las revistas en las que ha publicado cada autor + ratings Revisa->Autor
file_path_journals2 = './procesados/training/autor-revista-biNormalizado.csv'
# Archivo de test para evaluar las recomendaciones
file_path_test = './procesados/test/autor-revista-numpublic_test_filtrado.csv'

# Pasamos los archivos a dataframes
df_authors = pd.read_csv(file_path_authors)
df_coauthors = pd.read_csv(file_path_coauthors)
df_journals = pd.read_csv(file_path_journals)
df_journals2 = pd.read_csv(file_path_journals2)
df_test = pd.read_csv(file_path_test)

autores_aleatorios = []
# Iterar sobre todos los autores aleatorios
#while len(autores_aleatorios) < 1000:
#    autor_aleatorio = df_authors['codigo_autor'].sample(n=1).iloc[0]
#    if autor_aleatorio in df_coauthors['codigo_autor'].values:
#        autores_aleatorios.append(autor_aleatorio)
#
###Guardamos los autores en un archivo
#ruta = './src/general/autores_aleatorios.csv'
#df_autores = pd.DataFrame(autores_aleatorios, columns=['codigo_autor'])
#df_autores.to_csv(ruta, index=False)

autores_aleatorios = pd.read_csv('./src/general/autores_aleatorios.csv')['codigo_autor']

autores = df_authors['codigo_autor'].tolist()

################################################################################
## PRUEBAS PARA FIJAR EL NÚMERO K DE REVISTAS A RECOMENDAR
################################################################################

## ENFOQUE ESTÁTICO:
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/k/results_k3.csv', 30, 3,0.15, 1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/k/results_k5.csv', 30, 5,0.15, 1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/k/results_k10.csv', 30, 10,0.15, 1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/k/results_k15.csv', 30, 15,0.15, 1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/k/results_k20.csv', 30, 20,0.15, 1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/k/results_k25.csv', 30, 25,0.15, 1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/k/results_k30.csv', 30, 30,0.15, 1e-9)

## ENFOQUE DINÁMICO:
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/k/results_kdinamico.csv', 30,0 ,0.15, 1e-9)

################################################################################
## FIN PRUEBAS PARA FIJAR EL NÚMERO K DE REVISTAS A RECOMENDAR
################################################################################


################################################################################
## PRUEBAS PARA FIJAR LA TASA C DE RANDOM WALK WITH RESTART
################################################################################
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c005.csv', 30, 10, 0.05,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c010.csv', 30, 10, 0.10,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c015.csv', 30, 10, 0.15,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c020.csv', 30, 10, 0.20,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c025.csv', 30, 10, 0.25,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c030.csv', 30, 10, 0.30,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c035.csv', 30, 10, 0.35,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c040.csv', 30, 10, 0.40,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c045.csv', 30, 10, 0.45,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c050.csv', 30, 10, 0.50,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c055.csv', 30, 10, 0.55,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c060.csv', 30, 10, 0.60,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c065.csv', 30, 10, 0.65,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c070.csv', 30, 10, 0.70,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c075.csv', 30, 10, 0.75,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c080.csv', 30, 10, 0.80,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c085.csv', 30, 10, 0.85,1e-9)
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios, './src/resultados/c/results_c090.csv', 30, 10, 0.90,1e-9)

# Hemos visto que la mejor es 0.15:
#prueba_final(df_journals, df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/c/FINAL_C015.csv', 30, 5, 0.15, 1e-9)
################################################################################
# FIN PRUEBAS PARA FIJAR LA TASA C DE RANDOM WALK WITH RESTART
################################################################################

################################################################################
## PRUEBAS PARA FIJAR LA TASA E DE RANDOM WALK WITH RESTART
################################################################################
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e9.csv', 30, 10, 0.15, 1e-9)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e8.csv', 30, 10, 0.15, 1e-8)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e7.csv', 30, 10, 0.15, 1e-7)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e6.csv', 30, 10, 0.15, 1e-6)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e5.csv', 30, 10, 0.15, 1e-5)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e4.csv', 30, 10, 0.15, 1e-4)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e3.csv', 30, 10, 0.15, 1e-3)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_e1e2.csv', 30, 10, 0.15, 1e-2)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/e/results_1.csv', 30, 10, 0.15, 1)

################################################################################
# FIN PRUEBAS PARA FIJAR LA TASA E DE RANDOM WALK WITH RESTART
################################################################################

################################################################################
## PRUEBAS AÑADIENDO ARCOS REVISTA->AUTOR
################################################################################
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/pruebarwrCONRevistas.csv', 30, 10, 0)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/pruebarwrSINRevistas.csv', 30, 10, 1)

################################################################################
## PRUEBAS PARA FACTOR R
################################################################################
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/r/pruebaFactorR07.csv', 30, 5, 0.15, 1e-9, 0.7)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/r/pruebaFactorR09.csv', 30, 5, 0.15, 1e-9, 0.9)
#prueba_final(df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios,'./src/resultados/r/pruebaFactorR099.csv', 30, 5, 0.15, 1e-9, 0.99)

################################################################################
## FIN PRUEBAS PARA FACTOR R
################################################################################

################################################################################
## PRUEBAS PARA FACTOR N DE FC
################################################################################
#prueba_final("FC",df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios, 20,10,'./src/resultados/n_autorVecino/results_n')

################################################################################
## FIN PRUEBAS PARA FACTOR N DE FC
################################################################################

################################################################################
## PRUEBAS PARA FACTOR N DE RWR_FC
################################################################################
#prueba_final("RWR_FC",df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios, 35,10,'./src/resultados/n_hibrido/results_n')

################################################################################
## FIN PRUEBAS PARA FACTOR N DE RWR_FC
################################################################################

################################################################################
## PRUEBAS SOBRE EL VECINDARIO DE RWR_FC
################################################################################
#prueba_final("RWR_FC",df_journals,df_journals2, df_coauthors, df_test, autores_aleatorios, 35,10,'./src/resultados/vecindario_hibrido/results_vecindario')

################################################################################
## EJECUCIÓN FINAL RWR
################################################################################
#prueba_final("RWR_COMPLETO",df_journals,df_journals2, df_coauthors, df_test, autores, 5,30,'./src/resultados/FINAL_RWR/results_rwr_completo_k',0.15,1e-9,0.7)

################################################################################
## EJECUCIÓN FINAL FC
################################################################################
#prueba_final("FC",df_journals,df_journals2, df_coauthors, df_test, autores, 20,30,'./src/resultados/FINAL_FC/results_fc_k')

################################################################################
## EJECUCIÓN FINAL RWR_FC
################################################################################
#prueba_final("RWR_FC",df_journals,df_journals2, df_coauthors, df_test, autores, 10,30,'./src/resultados/FINAL_RWR_FC/results_rwr_fc_k')