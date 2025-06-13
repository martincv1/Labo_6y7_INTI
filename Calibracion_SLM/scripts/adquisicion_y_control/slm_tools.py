import numpy as np


# Función que crea el patron escalon
def crear_patron(resolucion, orientacion, mitad, intensidad):
    grayscale_array = np.zeros(resolucion, dtype=np.uint8)
    if orientacion == "horizontal":
        half_height = resolucion[0] // 2  # altura mitad
        if mitad == "sup":
            grayscale_array[:half_height, :] = intensidad  # llena las filas hasta half_height con intensidad
        elif mitad == "inf":
            grayscale_array[half_height:, :] = intensidad  # llena las filas desde half_height con intensidad
    if orientacion == "vertical":
        half_width = resolucion[1] // 2
        if mitad == "izq":
            grayscale_array[:, :half_width] = intensidad
        elif mitad == "der":
            grayscale_array[:, half_width:] = intensidad
    return grayscale_array
