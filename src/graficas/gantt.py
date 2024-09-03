import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Crear un DataFrame con los datos del proyecto
data = {
    'Tarea': ['Revisión de la literatura', 'Estudio de las tecnologías', 'Estudio del conjunto de datos', 'Preprocesamiento de los datos',
              'Obtención de ratings', 'Implementación del modelo RWR', 'Implementación del modelo FC', 'Implementación del modelo híbrido',
              'Estudio de los resultados'],
    'Inicio': ['2023-03-01', '2023-03-15', '2023-04-01', '2023-04-15', '2023-05-01', '2023-05-15', '2023-07-01', '2023-08-15', '2023-07-15'],
    'Fin': ['2023-03-15', '2023-04-01', '2023-04-15', '2023-05-01', '2023-05-15', '2023-08-01', '2023-07-15', '2023-09-15', '2023-09-01']
}

df = pd.DataFrame(data)

# Convertir las fechas de inicio y fin a formato datetime
df['Inicio'] = pd.to_datetime(df['Inicio'])
df['Fin'] = pd.to_datetime(df['Fin'])

# Calcular la duración de cada tarea
df['Duración'] = df['Fin'] - df['Inicio']

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(12, 6))

# Crear el diagrama de Gantt
for idx, row in df.iterrows():
    ax.barh(row['Tarea'], row['Duración'].days, left=row['Inicio'])

# Formatear el eje x para mostrar las fechas
ax.set_xlabel('Fecha')
ax.set_ylabel('Tarea')
ax.set_title('Diagrama de Gantt del Proyecto')

# Establecer el formato de las fechas en el eje x
ax.xaxis.set_major_locator(plt.MultipleLocator(30))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: datetime.datetime.fromordinal(int(x)).strftime('%b %Y')))

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
