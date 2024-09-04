# TFG_software

**Recomendación de revistas científicas usando redes de coautoría**  
_Uso de algoritmos RWR y filtrado colaborativo para la recomendación de revistas científicas_

## Descripción del proyecto

Este proyecto tiene como objetivo implementar un sistema de recomendación de revistas científicas utilizando redes de coautoría. Los algoritmos principales que se utilizan en el proyecto son **Random Walk with Restart (RWR)** y **Filtrado Colaborativo (FC)**. También se incluye una implementación híbrida que combina ambos enfoques para mejorar los resultados de la recomendación.

## Estructura del proyecto

- **`src/`**: Carpeta que contiene el código fuente del proyecto.
  - **general/**: Implementación de los algoritmos RWR, FC y su versión híbrida.
  - **dataframes/**: Scripts para la preparación y normalización de datos.
  - **graficas/**: Scripts para la visualización de resultados y gráficos.
  - **resultados/**: Almacenaje de los resultados finales y de los experimentos
  - **RWR/**: Conjuntos de datos necesarios para el algoritmo RWR

- **`procesados/`**: Carpeta que contiene los conjuntos de datos utilizados en el proyecto.

- **`pyrwr-master/`**: Carpeta de la librería usada para el algoritmo de RWR

## Requisitos

- **Python3.**
- **Pandas**
- **Surprise**
- **pyrwr**

## Uso

Para la ejecución de los 3 modelos se usará el siguiente comando

`python src/general/pruebaGeneral.py`

## Contacto

Para más información acerca del proyecto:

- Javier Arroyo García
- arroyojavi@correo.ugr.es

