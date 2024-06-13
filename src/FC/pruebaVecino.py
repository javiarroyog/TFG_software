import pandas as pd
from surprise import Dataset, KNNBasic, Reader

# Paso 1: Cargar los Datos
df_test = pd.read_csv('./procesados/test/autor-revista-numpublic_test_filtrado.csv', converters={'codigo_revista': str, 'codigo_autor': str})
df_train = pd.read_csv('./procesados/training/autor-coautor-normalizado.csv', converters={'codigo_autor': str, 'codigo_coautor': str})
df_revistas = pd.read_csv('./procesados/training/autor-revista-normalizado.csv', converters={'codigo_revista': str, 'codigo_autor': str})

# Paso 2: Preparar los Datos para Surprise
reader = Reader(rating_scale=(df_train['coautorías_normalizadas'].min(), df_train['coautorías_normalizadas'].max()))
print("mínimo -->", df_train['coautorías_normalizadas'].min(), "máximo --> ", df_train['coautorías_normalizadas'].max())

data_train = Dataset.load_from_df(df_train[['codigo_autor', 'codigo_coautor', 'coautorías_normalizadas']], reader)

# Paso 3: Entrenar el Modelo
trainset = data_train.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': True}
algoritmo = KNNBasic(sim_options=sim_options)
algoritmo.fit(trainset)

# Obtener y Mostrar los k Vecinos Más Cercanos
def obtener_vecinos_mas_cercanos(algoritmo, autor_inicial, k):
    # Convertir el autor específico a su id interno en Surprise
    autor_id = algoritmo.trainset.to_inner_uid(autor_inicial)
    
    # Obtener los k vecinos más cercanos (usuarios similares)
    vecinos = algoritmo.get_neighbors(autor_id, k=k)
    
    # Convertir los ids internos de Surprise de vuelta a los ids originales de los autores
    vecinos_ids = [algoritmo.trainset.to_raw_uid(vid) for vid in vecinos]
    
    return vecinos_ids

# Por cada vecino, se obtienen las revistas en las que ha publicado
def obtener_revistas_vecinos(vecinos):
    revistas = []
    for vecino in vecinos:
        revistas.append(df_revistas[df_revistas['codigo_autor'] == vecino])
    
    # Crear un DataFrame ranking de las revistas más repetidas
    revistas = pd.concat(revistas)
    revistas = revistas.groupby('codigo_revista').size().reset_index(name='count')
    revistas = revistas.sort_values(by='count', ascending=False, ignore_index=True)
    
    print(revistas)
    revistas.to_csv('./prueba.csv', index=False)
    return revistas['codigo_revista'].tolist()

def calculaPrecision(revistas_vecinos, autor_inicial):
    revistas_autor_inicial = df_test[df_test['codigo_autor'] == autor_inicial]['codigo_revista'].unique()
    print("Revistas autor inicial -->", revistas_autor_inicial)

    revistas_relevantes = 0
    for revista in revistas_vecinos:
        if revista in revistas_autor_inicial:
            revistas_relevantes += 1

    print("Revistas relevantes -->", revistas_relevantes)
    precision = revistas_relevantes / len(revistas_vecinos) if len(revistas_vecinos) > 0 else 0
    recall = revistas_relevantes / len(revistas_autor_inicial) if len(revistas_autor_inicial) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# Cogemos un autor de ejemplo de test
autor_inicial = df_test['codigo_autor'].sample(1).iloc[0]

# Obtenemos los vecinos más cercanos
vecinos = obtener_vecinos_mas_cercanos(algoritmo, autor_inicial, 5)
print(f"Obtenidos {len(vecinos)} vecinos del autor {autor_inicial}:")
for vecino in vecinos:
    print(vecino)

# Obtenemos las revistas en las que han publicado los vecinos
revistas_vecinos = obtener_revistas_vecinos(vecinos)

# Calculamos la precisión, el recall y el F1
precision,recall,f1 = calculaPrecision(revistas_vecinos, autor_inicial)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
