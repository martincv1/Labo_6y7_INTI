# Importo librerias
import numpy as np
import HEDS
from hedslib.heds_types import *
import time
import pickle
import os
import cv2

from Calibracion_SLM.scripts.adquisicion_y_control.acquisition_tools import jai_camera
import Calibracion_SLM.scripts.adquisicion_y_control.slm_tools as slm_tools

camera = jai_camera()

# Defino array de intensidades e inicializo el SLM

intensidades_array = np.arange(0, 255, 1)
# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Defino variables para realizar la medición

resol_SLM = (1080, 1920)
tiempo_espera_inicial = 120
print(f'Espero {tiempo_espera_inicial} s antes de empezar')
time.sleep(tiempo_espera_inicial)
fecha = '1306'
T_dia = 21
cant_promedio = 5
save_dir = r"data\fase_tension"
cant_pruebas_retrieve = 4
tiempo_prueba = 0.05

os.makedirs(save_dir, exist_ok=True)

# Escribo el bucle de la medición: barro en intensidades, para cada intensidad saco varias imgs
# que promedio. Todo esto para un valor de tensión que cambio desde el PCM

for i in intensidades_array:
    print(f"Mostrando patrón con intensidad: {i}")    
    patron = slm_tools.crear_patron(resol_SLM, "horizontal", "sup", i)
    # va a guardar cada patron como archivo.npy en la carpeta patrones
    # np.save(f"patrones/patron_{idx:03d}.npy", patron)
    err, dataHandle = slm.loadImageData(patron)  # carga la data (array) a la video memory del display SLM
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    err = dataHandle.show()  # Show the returned data handle on the SLM
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    time.sleep(0.5)

    camera.reset_queue()

    file_name = os.path.join(save_dir, f'{fecha}_I{i}_T{T_dia}')
    if cant_promedio == 1:
        frame = jai_camera.get_frame(camera)        
        cv2.imwrite(f'{file_name}.png', frame)
    else:
        frame = jai_camera.get_average_frame(camera, cant_promedio)
        with open(f'{file_name}.pkl', 'wb') as f:
            pickle.dump(frame, f)

camera.close()
