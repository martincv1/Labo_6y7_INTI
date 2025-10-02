# Importo librerias
import numpy as np
import HEDS
from hedslib.heds_types import *
import time
import pickle
import os
import cv2

from acquisition_tools import jai_camera, holoeye_SLM

camera = jai_camera()

# Defino array de intensidades e inicializo el SLM

intensidades_array = np.arange(256)
# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
SLM = holoeye_SLM()

# Defino variables para realizar la medición

resol_SLM = (1080, 1920)
tiempo_espera_inicial = 60
print(f"Espero {tiempo_espera_inicial} s antes de empezar")
# time.sleep(tiempo_espera_inicial)
fecha = time.strftime("%d%m")
T_dia = 21
cant_promedio = 1

cant_pruebas_retrieve = 4
tiempo_prueba = 0.05
for ronda in [9, 10, 11, 12]:
    # time.sleep(tiempo_espera_inicial)
    save_dir = rf"data\repetibilidad\{ronda} 0.69-1.42\imgs"
    os.makedirs(save_dir, exist_ok=True)
    time.sleep(tiempo_espera_inicial)

    # Escribo el bucle de la medición: barro en intensidades, para cada intensidad saco varias imgs
    # que promedio. Todo esto para un valor de tensión que cambio desde el PCM

    for i in intensidades_array:
        print(f"Mostrando patrón con intensidad: {i}")
        patron = SLM.crear_patron(resol_SLM, "horizontal", "sup", i)

        SLM.mostrar_patron(patron)
        time.sleep(0.5)
        for j in range(10):
            camera.reset_queue()

            file_name = os.path.join(save_dir, f"{fecha}_I{i}_T{T_dia}")
            if cant_promedio == 1:
                frame = camera.get_frame()
                cv2.imwrite(f"{file_name}_{j}.png", frame)
            else:
                frames = camera.get_multiple_frame(cant_promedio)
                for h in range(cant_promedio):
                    cv2.imwrite(f"{file_name}_{h}.png", frames[h])

camera.close()
