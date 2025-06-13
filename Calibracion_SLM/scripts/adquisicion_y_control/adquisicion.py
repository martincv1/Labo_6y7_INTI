# Importo librerias
import numpy as np
import eBUS as eb
import lib.PvSampleUtils as psu
import HEDS
from hedslib.heds_types import *
import time
import pickle
import cv2

from Calibracion_SLM.scripts.adquisicion_y_control.acquisition_tools import jai_camera
import Calibracion_SLM.scripts.adquisicion_y_control.slm_tools as slm_tools


camera = jai_camera()

intensidades_array = np.arange(0, 255, 25)
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
segundos = 120
print(f'Espero {segundos} antes de empezar')
time.sleep(segundos)
fecha = '0506'
T_dia = 22
cant_muestras_promediadas = 3
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

    if cant_muestras_promediadas == 1:
        frame = jai_camera.get_frame(camera)
        cv2.imwrite(f'fotos/{fecha}_I{i}_T{T_dia}.png', frame)
    else:
        frame = jai_camera.get_average_frame(camera, cant_muestras_promediadas)        
        with open(f'fotos/pik/{fecha}_I{i}_T{T_dia}.pkl', 'wb') as f:
            pickle.dump(frame, f)

jai_camera.close()
