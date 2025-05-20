# Importo las librerias asociadas al control del SLM: HOLOEYE SLM Display SDK
import HEDS
from hedslib.heds_types import *

# Importo numpy y time
import numpy as np
import matplotlib.pyplot as plt
import time

def crear_patron(resolucion, orientacion, mitad, intensidad):
    grayscale_array = np.zeros(resolucion, dtype=np.uint8)
    if orientacion == "horizontal":
        half_height = resolucion[0]//2  #altura mitad
        if mitad == "sup":
            grayscale_array[:half_height, :] = intensidad #llena las filas hasta half_height con intensidad
        elif mitad == "inf":
            grayscale_array[half_height:, :] = intensidad #llena las filas desde half_height con intensidad
    if orientacion == "vertical":
        half_width = resolucion[1]//2
        if mitad == "izq":
            grayscale_array[:, :half_width] = intensidad
        elif mitad =="der":
            grayscale_array[:, half_width:] = intensidad
    return grayscale_array

#print(crear_patron(resolucion = (6,4), intensidad=255, orientacion="vertical", mitad="der"))
#plt.imshow(crear_patron((1080, 1920), "vertical", "izq", 100 ))
#plt.show()
# Primero defino el array de intensidades, el cual vamos a utilizar para pasarle los distintos valores al SLM
intensidades = [0, 100, 150, 255]
#for i in range(0, 255+1):
#    intensidades.append(i)
intensidades_array = np.array(intensidades)

#Inicializo el SLM
# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm  = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())
bucle = True
#Este es el bucle de medición
resol_SLM = (1080, 1920)
if bucle:
    for i in intensidades_array:
        print(f"Mostrando patrón con intensidad: {i}")
        patron = crear_patron(resol_SLM, "horizontal", "sup", i )
        #np.save(f"patrones/patron_{idx:03d}.npy", patron)    # va a guardar cada patron como archivo.npy en la carpeta patrones
        err, dataHandle = slm.loadImageData(patron)  #carga la data (array) a la video memory del display SLM
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        err = dataHandle.show() # Show the returned data handle on the SLM
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        time.sleep(3)
else:
    patron = crear_patron(resol_SLM, "horizontal", "sup", 255 )
    err, dataHandle = slm.loadImageData(patron)  #carga la data (array) a la video memory del display SLM
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    err = dataHandle.show() # Show the returned data handle on the SLM
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()
