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
    patron = r"_(\d{2})-(\d{2})-(\d{2})-(\d{3})_I(\d+)_"  #ahora busca ms tmb
    coincidencia = re.search(patron, nombre_archivo)
    if coincidencia:
        hora, minuto, segundo, milisegundo, valor_I = coincidencia.groups()
        dt = datetime.strptime(f"{hora}:{minuto}:{segundo}.{milisegundo}", "%H:%M:%S.%f")
            
        try:
            with open(nombre_archivo, 'rb') as f:
                imag = pickle.load(f)
                
                # roto la imagen para alinear las franjas
                imag_rot = funciones.rotate_bound(imag, 1.7)
                # selecciono las zonas de interés
                recorte1 = imag_rot[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
                recorte2 = imag_rot[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]

                # Obtengo fase a partir de ordenada al origen de  hilbert
                fase_mod, fase_nomod  = funciones.fase_hilbert(recorte1, recorte2, 0.12, 0.02)  
                fasor_mod = np.exp(1j*fase_mod)
                fasor_nomod = np.exp(1j*fase_nomod)
                return dt.time(), fase_mod, fase_nomod, int(valor_I)
        except Exception as e:
            print(f"Error al procesar {nombre_archivo}: {e}")
            return None, None, None, None
    return None, None, None, None  


dir_path = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_largo0702'
paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pkl')]
#paths = []
#for archivo in os.listdir(dir_path):
 #   paths.append(os.path.join(dir_path, archivo))
#data_imagen = extraer_datos('/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo/1206_13-55-13_I40_T21.pkl')
#print(data_imagen)

data = {
    40: {"dts": [], "fase_nm": [], "fase_m": []},
    128: {"dts": [], "fase_nm": [], "fase_m": []},
    230: {"dts": [], "fase_nm": [], "fase_m": []},
    60: {"dts": [], "fase_nm": [], "fase_m": []}
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


mask = fase_40 < 2
fase_40_limpia = fase_40[mask]



complejos_128 = np.exp(1j*dfs[128]["Fase_no_modulada"])
complejos_128_mod = np.exp(1j*dfs[128]["Fase_modulada"])
div_128 = complejos_128/complejos_128_mod
fase_128 = np.angle(div_128)

# Bloque para 230 (modificado desde 200)
complejos_230 = np.exp(1j*dfs[230]["Fase_no_modulada"])
complejos_230_mod = np.exp(1j*dfs[230]["Fase_modulada"])
div_230 = complejos_230/complejos_230_mod
fase_230 = np.angle(div_230)

# Nuevo bloque para 60 (agregado)
complejos_60 = np.exp(1j*dfs[60]["Fase_no_modulada"])
complejos_60_mod = np.exp(1j*dfs[60]["Fase_modulada"])
div_60 = complejos_60/complejos_60_mod
fase_60 = np.angle(div_60)

mask_60 = fase_60 >0
fase_60_limpia = fase_60[mask_60]

# Alternativa: Calcular fase_mod - fase_nomod (asegurando que sea positiva)
#fase_40 = np.angle(complejos_40_mod * np.conj(complejos_40))  # = fase_mod - fase_nomod
#fase_128 = np.angle(complejos_128_mod * np.conj(complejos_128))
#fase_230 = np.angle(complejos_230_mod * np.conj(complejos_230))
#fase_60 = np.angle(complejos_60_mod * np.conj(complejos_60))

#defino mi array de tiempos a partir de data.datatime
horas_segundos_40 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[40]["Hora"]]
horas_segundos_128 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[128]["Hora"]]
horas_segundos_230 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[230]["Hora"]]
horas_segundos_60 = [h.hour * 3600 + h.minute * 60 + h.second for h in dfs[60]["Hora"]]


horas_segundos_230 = np.array(horas_segundos_230)
tiempo_centrado_230 = horas_segundos_230 - horas_segundos_230.min()
horas_segundos_40 = np.array(horas_segundos_40)
tiempo_centrado_40 = horas_segundos_40 - horas_segundos_40.min()
horas_segundos_128 = np.array(horas_segundos_128)
tiempo_centrado_128 = horas_segundos_128 - horas_segundos_128.min()
horas_segundos_60 = np.array(horas_segundos_60)
tiempo_centrado_60 = horas_segundos_60 - horas_segundos_60.min()

horas_segundos_40_filtrado = np.array(horas_segundos_40)[mask]
tiempo_cen_40 = horas_segundos_40_filtrado - horas_segundos_40_filtrado.min()
horas_segundos_60_filtrado = np.array(horas_segundos_60)[mask_60]
tiempo_cen_60 = horas_segundos_60_filtrado - horas_segundos_60_filtrado.min()

plt.scatter(tiempo_cen_40, fase_40_limpia, label =f"Stdv ={np.round(np.std(fase_40_limpia), decimals = 3)}")
plt.axhline(np.mean(fase_40_limpia), color='red', linestyle='--', label=f'Media = {np.round(np.mean(fase_40_limpia), decimals = 2)}')
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.legend()
#plt.title("Intensidad 40")
plt.show()
plt.scatter(tiempo_cen_60, fase_60_limpia, label =f"Stdv ={np.round(np.std(fase_60_limpia), decimals = 3)}")
plt.axhline(np.mean(fase_60_limpia), color='red', linestyle='--', label=f'Media = {np.round(np.mean(fase_60_limpia), decimals = 2)}')
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.legend()
#plt.title("Intensidad 60")
plt.show()
#plt.scatter(horas_segundos_40_filtrado, fase_40_limpia)
#plt.axhline(np.mean(fase_40_limpia), color='red', linestyle='--', label='Media')
#plt.xlabel("Tiempo [s]")
#plt.ylabel("Diferencia de Fase [rad]")
#plt.title("Intensidad 40")
#plt.legend()
#plt.show()
plt.scatter(tiempo_centrado_128, fase_128)
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.title("Intensidad 128")
plt.show()
plt.scatter(tiempo_centrado_230, fase_230)
plt.xlabel("Tiempo [s]")
plt.ylabel("Diferencia de Fase [rad]")
plt.title("Intensidad 230")
plt.show()


print(dfs[40]["Fase_no_modulada"], type(dfs[40]["Fase_no_modulada"]))
print(dfs[40]["Hora"])