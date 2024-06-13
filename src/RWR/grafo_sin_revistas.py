import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Reemplaza 'autor-coautor-numpublic_training.csv' con la ruta a tu archivo CSV
file_path = './procesados/training/autor-coautor-numpublic_training.csv'

# Leer el archivo .csv usando pandas
df = pd.read_csv(file_path)

# Crear un grafo vacío
G = nx.Graph()

# Agregar las aristas al grafo con pesos
for index, row in df.iterrows():
    G.add_edge(row['codigo_autor'], row['codigo_coautor'], weight=row['num_publicaciones'])

# Mostrar información del grafo
print(f"Número de nodos: {G.number_of_nodes()}")
print(f"Número de aristas: {G.number_of_edges()}")

# Dibujar el grafo mostrando los pesos
pos = nx.spring_layout(G)  # posiciones para todos los nodos
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)

# Asegúrate de que las etiquetas de los pesos estén en el formato correcto
edge_labels = {(row['codigo_autor'], row['codigo_coautor']): row['num_publicaciones'] for index, row in df.iterrows()}

# Dibujar las etiquetas de las aristas
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.show()
