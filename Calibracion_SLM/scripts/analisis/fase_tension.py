import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re
from datetime import datetime

def extraer_datos(nombre_archivo):
    # Expresión regular para capturar la hora y el valor después de la I
    patron = r"_(\d{2})-(\d{2})-(\d{2})_I(\d+)_"
    coincidencia = re.search(patron, nombre_archivo)
    if coincidencia:
        hora, minuto, segundo, valor_I = coincidencia.groups()
        dt = datetime.strptime(f"{hora}:{minuto}:{segundo}", "%H:%M:%S")
        return dt, int(valor_I)
    else:
        return None, None


dir_path = r'Calibracion_SLM\data\fase_tiempo'
paths = os.listdir(dir_path)
file_path = r'Calibracion_SLM\data\fase_tiempo\1206_13-54-43_I40_T21.pkl'
dts = []
intensidades = []
for path in paths:
    dt, inten = extraer_datos(path)
    if dt is not None:
        dts.append(dt)
        intensidades.append(inten)



with open(file_path, mode = 'rb') as f:
    imagen = pickle.load(f)

plt.imshow(imagen)
plt.show()