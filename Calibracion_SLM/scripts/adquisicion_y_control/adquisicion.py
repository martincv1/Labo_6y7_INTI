# Importo librerias
import numpy as np
import HEDS
from hedslib.heds_types import *
import time
import pickle
import cv2
import os

from acquisition_tools import jai_camera
import slm_tools


camera = jai_camera()

intensidades_array = [0]
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
segundos = 10
print(f'Espero {segundos} antes de empezar')
time.sleep(segundos)
fecha = '2608'
T_dia = 22
cant_muestras_promediadas = 1
cant_ims_por_intensidad = 10
save_dir = 'data\Topografia_en0_2_pruebaclase'

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

    for kim in range(cant_ims_por_intensidad):
        filename = f"{fecha}_I{i}_{kim}_T{T_dia}"
        file_path = os.path.join(save_dir, filename)
        if cant_muestras_promediadas == 1:
            frame = camera.get_frame()
            cv2.imwrite(f'{file_path}.png', frame)
        else:
            frame = camera.get_average_frame(cant_muestras_promediadas)        
            with open(f'{file_path}.pkl', 'wb') as f:
                pickle.dump(frame, f)

camera.close()
