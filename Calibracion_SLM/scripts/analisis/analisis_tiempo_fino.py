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


dir_path = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702'
paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pkl')]
#paths = []
#for archivo in os.listdir(dir_path):
 #   paths.append(os.path.join(dir_path, archivo))
primer_data_imagen = extraer_datos('/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702/0207_12-01-51-004_I40_0_T20.pkl')
ultima_data_imagen = extraer_datos('/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702/0207_12-02-40-505_I40_239_T20.pkl')
print(primer_data_imagen)
print(ultima_data_imagen)

data = {
    40: {"dts": [], "fase_nm": [], "fase_m": []},
    128: {"dts": [], "fase_nm": [], "fase_m": []},
    180: {"dts": [], "fase_nm": [], "fase_m": []},
    230: {"dts": [], "fase_nm": [], "fase_m": []},
    250: {"dts": [], "fase_nm": [], "fase_m": []}
    }
graficar = True
if graficar:
        
    for p in paths:
        dt, fase_nm, fase_m, intensidad = extraer_datos(p)
        if intensidad in data:
            data[intensidad]["dts"].append(dt)
            data[intensidad]["fase_nm"].append(fase_nm)
            data[intensidad]["fase_m"].append(fase_m)


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



    complejos_128 = np.exp(1j*dfs[128]["Fase_no_modulada"])
    complejos_128_mod = np.exp(1j*dfs[128]["Fase_modulada"])
    div_128 = complejos_128/complejos_128_mod
    fase_128 = np.angle(div_128)

    complejos_180 = np.exp(1j*dfs[180]["Fase_no_modulada"])
    complejos_180_mod = np.exp(1j*dfs[180]["Fase_modulada"])
    div_180 = complejos_180/complejos_180_mod
    fase_180 = np.angle(div_180)

    complejos_230 = np.exp(1j*dfs[230]["Fase_no_modulada"])
    complejos_230_mod = np.exp(1j*dfs[230]["Fase_modulada"])
    div_230 = complejos_230/complejos_230_mod
    fase_230 = np.angle(div_230)

    complejos_250 = np.exp(1j*dfs[250]["Fase_no_modulada"])
    complejos_250_mod = np.exp(1j*dfs[250]["Fase_modulada"])
    div_250 = complejos_250/complejos_250_mod
    fase_250 = np.angle(div_250)

    mask_2 = fase_250 > -1.8
    fase_250_clean = fase_250[mask_2]

    #defino mi array de tiempos a partir de data.datatime,
    # aca convierte todo a segundos pero dado que tenemos precision de milisegundos no es lo ideal
    horas_segundos_40 = [h.hour * 3600 + h.minute * 60 + h.second + h.microsecond / 1_000_000 for h in dfs[40]["Hora"]]
    horas_segundos_128 = [h.hour * 3600 + h.minute * 60 + h.second + h.microsecond / 1_000_000 for h in dfs[128]["Hora"]]
    horas_segundos_180 = [h.hour * 3600 + h.minute * 60 + h.second + h.microsecond / 1_000_000 for h in dfs[180]["Hora"]]
    horas_segundos_230 = [h.hour * 3600 + h.minute * 60 + h.second + h.microsecond / 1_000_000 for h in dfs[230]["Hora"]]
    horas_segundos_250 = [h.hour * 3600 + h.minute * 60 + h.second + h.microsecond / 1_000_000 for h in dfs[250]["Hora"]]

    # aca convierto todo a milisegundos
    # Convertir a arrays de numpy y centrar en cero
    horas_milisegundos_40 = np.array([h.hour * 3_600_000 + h.minute * 60_000 + h.second * 1_000 + h.microsecond // 1_000 for h in dfs[40]["Hora"]])
    tiempo_centrado_ms_40 = horas_milisegundos_40 - horas_milisegundos_40.min()


    horas_segundos_40 = np.array(horas_segundos_40)
    tiempo_centrado_40 = horas_segundos_40 - horas_segundos_40.min()
    horas_segundos_128 = np.array(horas_segundos_128)
    tiempo_centrado_128 = horas_segundos_128 - horas_segundos_128.min()
    horas_segundos_180 = np.array(horas_segundos_180)
    tiempo_centrado_180 = horas_segundos_180 - horas_segundos_180.min()
    horas_segundos_230 = np.array(horas_segundos_230)
    tiempo_centrado_230 = horas_segundos_230 - horas_segundos_230.min()
    horas_segundos_250 = np.array(horas_segundos_250)
    tiempo_centrado_250 = horas_segundos_250 - horas_segundos_250.min()

    horas_segundos_40_filtrado = np.array(horas_segundos_40)[mask]
    horas_segundos_250_filtrado = np.array(horas_segundos_250)[mask_2]
    tiempo_centr_250 = horas_segundos_250_filtrado - horas_segundos_250_filtrado.min()

    plt.scatter(tiempo_centrado_40, fase_40, label =f"Stdv ={np.round(np.std(fase_40), decimals = 3)}")
    plt.axhline(np.mean(fase_40), color='red', linestyle='--', label=f'Media = {np.round(np.mean(fase_40), decimals = 2)}')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Diferencia de Fase [rad]")
    #plt.title("Valor de gris 40")
    plt.legend()
    plt.show()
    #plt.scatter(tiempo_centrado_ms_40, fase_40)
    #plt.xlabel("Tiempo [ms]")
    #plt.ylabel("Diferencia de Fase [rad]")
    #plt.title("Intensidad 40")
    #plt.show()
    #plt.scatter(horas_segundos_40_filtrado, fase_40_limpia)
    #plt.axhline(np.mean(fase_40_limpia), color='red', linestyle='--', label=f'Media = {np.round(np.mean(fase_40_limpia), decimals = 2)}')
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Diferencia de Fase [rad]")
    #plt.title("Intensidad 40")
    #plt.legend()
    #plt.show()
    #plt.scatter(tiempo_centrado_128, np.abs(fase_128), label =f"Stdv ={np.round(np.std(np.abs(fase_128)), decimals = 3)}")
    #plt.axhline(np.mean(np.abs(fase_128)), color='red', linestyle='--', label=f'Media = {np.round(np.mean(np.abs(fase_128)), decimals = 2)}')
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Diferencia de Fase [rad]")
    #plt.title("Intensidad 128")
    #plt.legend()
    #plt.show()
    #plt.scatter(tiempo_centrado_180, fase_180)
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Diferencia de Fase [rad]")
    #plt.title("Intensidad 180")
    #plt.show()
    #plt.scatter(tiempo_centrado_230, fase_230)
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Diferencia de Fase [rad]")
    #plt.title("Intensidad 230")
    #plt.show()
    #plt.scatter(tiempo_centr_250, np.abs(fase_250_clean), label =f"Stdv ={np.round(np.std(np.abs(fase_250_clean)), decimals = 3)}")
    #plt.axhline(np.mean(np.abs(fase_250_clean)), color='red', linestyle='--', label=f'Media = {np.round(np.mean(np.abs(fase_250_clean)), decimals = 2)}')
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Diferencia de Fase [rad]")
    #plt.legend()
    #plt.title("Valor de gris 250")
    plt.show()

    print(dfs[40]["Fase_no_modulada"], type(dfs[40]["Fase_no_modulada"]))
    print(dfs[40]["Hora"])

# Elegí uno de los archivos .pkl que quieras visualizar
archivo_imagen = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702/0207_12-02-40-093_I40_237_T20.pkl'
archivo_imagen_1 = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702/0207_12-02-40-302_I40_238_T20.pkl'
archivo_imagen_2 = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702/0207_12-02-40-505_I40_239_T20.pkl'

# Abrir el archivo pickle
with open(archivo_imagen, 'rb') as f:
    imagen = pickle.load(f)
with open(archivo_imagen_1, 'rb') as f:
    imagen_1 = pickle.load(f)
with open(archivo_imagen_2, 'rb') as f:
    imagen_2 = pickle.load(f)

# Rotar y recortar (como en tu código)
imagen_rotada = funciones.rotate_bound(imagen, 1.7)
imagen_rotada_1 = funciones.rotate_bound(imagen_1, 3.5)
imagen_rotada_2 = funciones.rotate_bound(imagen_2, 2)
recorte1 = imagen_rotada_1[160:820, 214:1290]

#plt.imshow(imagen_rotada, cmap='gray')
#plt.axis('off')
#plt.show()
plt.imshow(imagen_rotada_1, cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(recorte1, cmap='gray')
plt.axis('off')
#plt.savefig("/home/lorenzo/Labo_6/recorte1_sin_marco.png", bbox_inches='tight', pad_inches=0)
plt.show()
