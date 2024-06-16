import pandas as pd


def getRevistas(df_journals, autor):
    return df_journals[df_journals['codigo_autor'] == autor]['codigo_revista'].tolist()

def getRevistasPeso(df_journals, autor):
    return df_journals[df_journals['codigo_autor'] == autor][['codigo_revista', 'publicaciones_normalizadas']]

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

def calcular_ranking_revistas (df_journals, vecinos_mas_cercanos, num_revistas=20):
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

    #devolvemos tanto la revista como la puntuación
    #return ranking_revistas_ordenado

    return [revista for revista, _ in ranking_revistas_ordenado][:num_revistas]

def calcular_metricas(df_test, revistas_recomendadas, autor):
    # Filtrar las publicaciones del autor en el conjunto de test
    publicaciones_test = getRevistas(df_test, autor)

    print(f"Publicaciones test: \t{publicaciones_test}")
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
valores_k= [5,10,15,20,25,30,35,40,45,50]

for k in valores_k:
    metricas = []
    autores_aleatorios = []

    while len(autores_aleatorios) < 50:
        autor_aleatorio = df_authors['codigo_autor'].sample(n=1).iloc[0]
        if autor_aleatorio in df_coauthors['codigo_autor'].values:
            autores_aleatorios.append(autor_aleatorio)

    num_vecinos = 5
    num_revistas = 10

    for autor in autores_aleatorios:
        vecinos = FC(df_coauthors, autor, k)
        revistas = calcular_ranking_revistas(df_journals, vecinos, num_revistas)
        precision, recall, f1 = calcular_metricas(df_test, revistas, autor)
        metricas.append((precision, recall, f1))

    # Crear un DataFrame para las métricas y guardarlas en un archivo CSV
    df_metricas = pd.DataFrame(metricas, columns=['precision', 'recall', 'f1'])
    df_metricas.to_csv(f'./src/FC/metricas_k_{k}.csv', index=False)
    metricas_totales.append((k, df_metricas['precision'].mean(), df_metricas['recall'].mean(), df_metricas['f1'].mean()))

# Guardar las métricas promedio en un archivo CSV
df_medias_totales = pd.DataFrame(metricas_totales, columns=['epsilon', 'precision_media', 'recall_media', 'f1_media'])
df_medias_totales.to_csv('./src/FC/medias_totales_epsilon.csv', index=False)

print("Métricas calculadas y guardadas en archivos CSV.")
