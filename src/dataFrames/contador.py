import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('./procesados/training/autor-revista-normalizado.csv')

# Contar códigos únicos de revista
codigos_revista_unicos = df['codigo_revista'].nunique()

# Contar códigos únicos de autor
codigos_autor_unicos = df['codigo_autor'].nunique()

print(f"Total de códigos de revista únicos: {codigos_revista_unicos}")
print(f"Total de códigos de autor únicos: {codigos_autor_unicos}")
