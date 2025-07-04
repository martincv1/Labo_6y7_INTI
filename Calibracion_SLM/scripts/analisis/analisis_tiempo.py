import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re
from datetime import datetime
import pandas as pd
from collections import defaultdict
import cv2
from scipy.optimize import curve_fit
from scipy.signal import butter, firwin, filtfilt, hilbert
from collections import defaultdict
import funciones

def lineal(x, m, c):
    return m*x + c

altura_rec = 20
ancho_rec = 200
x_rec1 = 535
y_rec1 = 435
x_rec2 = 535
y_rec2 = 510

#altura_rec = 20
#ancho_rec = 200
#x_rec1 = 635
#y_rec1 = 450
#x_rec2 = 635
#y_rec2 = 530

def extraer_datos(nombre_archivo):
    # Expresión regular para capturar la hora y el valor después de la I
    patron = r"_(\d{2})-(\d{2})-(\d{2})_I(\d+)_"
    coincidencia = re.search(patron, nombre_archivo)
    if coincidencia:
        hora, minuto, segundo, valor_I = coincidencia.groups()
        dt = datetime.strptime(f"{hora}:{minuto}:{segundo}", "%H:%M:%S")
            
        try:
            with open(nombre_archivo, 'rb') as f:
                imag = pickle.load(f)
                
                # roto la imagen para alinear las franjas
                imag_rot = funciones.rotate_bound(imag, 1.7)
                # selecciono las zonas de interés
                recorte1 = imag_rot[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
                recorte2 = imag_rot[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]

                # Obtengo fase a partir de ordenada al origen de  hilbert
                fase_mod, fase_nomod  = funciones.fase_hilbert(recorte1, recorte2, 0.09, 0.03)  
                fasor_mod = np.exp(1j*fase_mod)
                fasor_nomod = np.exp(1j*fase_nomod)
                return dt.time(), fase_mod, fase_nomod, int(valor_I)
        except Exception as e:
            print(f"Error al procesar {nombre_archivo}: {e}")
            return None, None, None, None
    return None, None, None, None  


dir_path = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo'
paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pkl')]
#paths = []
#for archivo in os.listdir(dir_path):
 #   paths.append(os.path.join(dir_path, archivo))
#data_imagen = extraer_datos('/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo/1206_13-55-13_I40_T21.pkl')
#print(data_imagen)

data = {
    40: {"dts": [], "fase_nm": [], "fase_m": []},
    120: {"dts": [], "fase_nm": [], "fase_m": []},
    200: {"dts": [], "fase_nm": [], "fase_m": []}
    }

for p in paths:
    dt, fase_nm, fase_m, intensidad = extraer_datos(p)
    if intensidad in data:
        data[intensidad]["dts"].append(dt)
        data[intensidad]["fase_nm"].append(fase_nm)
        data[intensidad]["fase_m"].append(fase_m)

#dfs = {}
#for intensidad, valores in data.items():
#    if valores["dts"]:  # Si hay datos para esta intensidad
#        dfs[intensidad] = pd.DataFrame({
#            "Hora": valores["dts"],
#            "Fase_no_modulada": valores["fase_nm"],
#            "Fase_modulada": valores["fase_m"]
#        })

dfs = {}
for intensidad, valores in data.items():
    if valores["dts"]:  # Si hay datos para esta intensidad
        # Crear DataFrame temporal
        temp_df = pd.DataFrame({
            "Hora": valores["dts"],
            "Fase_no_modulada": valores["fase_nm"],
            "Fase_modulada": valores["fase_m"]
        })
        # Ordenar por hora
        temp_df = temp_df.sort_values("Hora")
        # Eliminar duplicados (manteniendo el primero que aparece)
        #temp_df = temp_df.drop_duplicates(subset="Hora", keep='first')
        # Resetear índice
        dfs[intensidad] = temp_df.reset_index(drop=True)

complejos_40 = np.exp(1j*dfs[40]["Fase_no_modulada"])
complejos_40_mod = np.exp(1j*dfs[40]["Fase_modulada"])
div_40 = complejos_40 / complejos_40_mod
fase_40 = np.angle(div_40)


mask = fase_40 > 0
fase_40_limpia = fase_40[mask]



complejos_120 = np.exp(1j*dfs[120]["Fase_no_modulada"])
complejos_120_mod = np.exp(1j*dfs[120]["Fase_modulada"])
div_120 = complejos_120/complejos_120_mod
fase_120 = np.angle(div_120)

complejos_200 = np.exp(1j*dfs[200]["Fase_no_modulada"])
complejos_200_mod = np.exp(1j*dfs[200]["Fase_modulada"])
div_200 = complejos_200/complejos_200_mod
fase_200 = np.angle(div_200)

#defino mi array de tiempos a partir de data.datatime
horas_segundos_40 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[40]["Hora"]]
horas_segundos_120 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[120]["Hora"]]
horas_segundos_200 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[200]["Hora"]]

horas_segundos_200 = np.array(horas_segundos_200)
tiempo_centrado_200 = horas_segundos_200 - horas_segundos_200.min()
horas_segundos_40 = np.array(horas_segundos_40)
tiempo_centrado_40 = horas_segundos_40 - horas_segundos_40.min()
horas_segundos_120 = np.array(horas_segundos_120)
tiempo_centrado_120 = horas_segundos_120 - horas_segundos_120.min()
horas_segundos_40_filtrado = np.array(horas_segundos_40)[mask]

plt.scatter(tiempo_centrado_40, fase_40)
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.title("Intensidad 40")
plt.show()
#plt.scatter(horas_segundos_40_filtrado, fase_40_limpia)
#plt.axhline(np.mean(fase_40_limpia), color='red', linestyle='--', label='Media')
#plt.xlabel("Tiempo [s]")
#plt.ylabel("Diferencia de Fase [rad]")
#plt.title("Intensidad 40")
#plt.legend()
#plt.show()
plt.scatter(tiempo_centrado_120, fase_120)
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.title("Intensidad 120")
plt.show()
plt.scatter(tiempo_centrado_200, fase_200)
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.title("Intensidad 200")
plt.show()

print(dfs[40]["Fase_no_modulada"], type(dfs[40]["Fase_no_modulada"]))
print(dfs[40]["Hora"])