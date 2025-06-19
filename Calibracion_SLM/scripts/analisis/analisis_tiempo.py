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

def lineal(x, m, c):
    return m*x + c

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

altura_rec = 20
ancho_rec = 100
x_rec1 = 635
y_rec1 = 470
x_rec2 = 635
y_rec2 = 540

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
                imag_rot = rotate_bound(imag, 4)
                # selecciono las zonas de interés
                recorte1 = imag_rot[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
                recorte2 = imag_rot[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]

                # Señales promedio por columna
                signal1 = np.sum(recorte1, axis=0)
                signal2 = np.sum(recorte2, axis=0)

                # FFT para encontrar la frecuencia dominante
                f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
                f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
                n = len(f_signal1)
                freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
                freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
                frecuencias = np.fft.fftfreq(len(signal1), d=1)
                freq1 = frecuencias[freq1_ind]
                freq2 = frecuencias[freq2_ind]

                # Parámetros del filtro
                fs = 1.0  # muestras por píxel
                bandwidth = 0.005  # ancho de banda en ciclos/píxel
                lowcut1 = freq1 - bandwidth
                highcut1 = freq1 + bandwidth
                lowcut2 = freq2 - bandwidth
                highcut2 = freq2 + bandwidth

                # Diseño del filtro FIR
                order = 5
                nyq = 0.5 * fs
                low1 = lowcut1 / nyq
                high1 = highcut1 / nyq
                low2 = lowcut2 / nyq
                high2 = highcut2 / nyq

                numtaps = 20
                b1 = firwin(numtaps, [low1, high1], pass_zero=False)
                b2 = firwin(numtaps, [low2, high2], pass_zero=False)

                # Filtrado de las señales
                filtra1 = filtfilt(b1, [1.0], signal1)
                filtra2 = filtfilt(b2, [1.0], signal2)

                # Transformada de Hilbert
                hil1 = hilbert(filtra1)
                hil2 = hilbert(filtra2)
                lin1 = np.unwrap(np.angle(hil1))
                lin2 = np.unwrap(np.angle(hil2))

                # Ajuste lineal
                inf = 30
                sup = 70
                lin1_ajust = lin1[inf:sup]
                lin2_ajust = lin2[inf:sup]
                popt1, pcov1 = curve_fit(lineal, np.arange(inf, sup), lin1_ajust)
                popt2, pcov2 = curve_fit(lineal, np.arange(inf, sup), lin2_ajust)
                diferencia_fase = popt1[1] - popt2[1] 
                
                return dt.time(), float(popt1[1]), float(popt2[1]), int(valor_I)
        except Exception as e:
            print(f"Error al procesar {nombre_archivo}: {e}")
            return None, None, None, None
    return None, None, None, None  


dir_path = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo'
paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pkl')]
#paths = []
#for archivo in os.listdir(dir_path):
 #   paths.append(os.path.join(dir_path, archivo))
data_imagen = extraer_datos('/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo/1206_13-55-13_I40_T21.pkl')
print(data_imagen)

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

dfs = {}
for intensidad, valores in data.items():
    if valores["dts"]:  # Si hay datos para esta intensidad
        dfs[intensidad] = pd.DataFrame({
            "Hora": valores["dts"],
            "Fase_no_modulada": valores["fase_nm"],
            "Fase_modulada": valores["fase_m"]
        })



print(dfs[40]["Hora"])



