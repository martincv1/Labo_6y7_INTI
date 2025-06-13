# Importo librerias
import numpy as np
import time
import pickle
import os
import cv2

# En la misma ruta que donde esta el codigo tiene que estar la carpeta HEDS, hedslib y lib.
# Tmb hay que instalar con pip install el archivo .whl del ebus
import HEDS
from hedslib.heds_types import *

from Calibracion_SLM.scripts.adquisicion_y_control.acquisition_tools import jai_camera
import Calibracion_SLM.scripts.adquisicion_y_control.slm_tools as slm_tools


camera = jai_camera()

intensidades_array = np.array([40, 120, 200])
print(intensidades_array)
# Inicializo el SLM
# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Este es el bucle de medición
resol_SLM = (1080, 1920)
tiempo_espera_inicial = 180
print(f'Espero {tiempo_espera_inicial} s antes de empezar')
time.sleep(tiempo_espera_inicial)
fecha = '1206'
T_dia = 21
cant_promedio = 5
save_dir = r"data\fase_tiempo"
muestras = 240
intervalo = 5  # para el sleep entre mediciones
cant_pruebas_retrieve = 4
tiempo_prueba = 0.05

os.makedirs(save_dir, exist_ok=True)
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

    for h in range(muestras):
        print(f"midiendo muestra {h}/{muestras}")

        t_ini = time.time()

        hora_actual = time.strftime("%H-%M-%S", time.localtime())
        file_name = os.path.join(save_dir, f'{fecha}_{hora_actual}_I{i}_T{T_dia}')
        if cant_promedio == 1:
            frame = jai_camera.get_frame(camera)
            cv2.imwrite(f'{file_name}.png', frame)
        else:
            frame = jai_camera.get_average_frame(camera, cant_promedio)
            with open(f'{file_name}.pkl', 'wb') as f:
                pickle.dump(frame, f)

        t_final = time.time()
        time.sleep(intervalo - (t_final - t_ini))

camera.close()
