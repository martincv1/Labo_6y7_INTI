from acquisition_tools import holoeye_SLM
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

SLM = holoeye_SLM()
SLM.__init__

resol_SLM = (1080, 1920)
lista_patrones = [0]
image_filepath = r"C:\Users\INTI\Pictures\SLM\2025_10_02 topografia\logo_INTI_50.png"
tiempo_espera = 120
prueba = "image_file"
if prueba == "patrones":
    for i in lista_patrones:
        patron = SLM.crear_patron(resol_SLM, "horizontal", "sup", i)
        SLM.mostrar_patron(patron)
        time.sleep(tiempo_espera)
elif prueba == "image_file":
    im = np.array(Image.open(image_filepath))[:, :, 0]
    if np.all(im.shape == resol_SLM):
        SLM.mostrar_patron(im)
    else:
        raise ValueError("La imagen cargada no tiene las dimensiones adecuadas.")
    time.sleep(tiempo_espera)

# SLM.close   # no funciona

# patron = SLM.crear_patron(resol_SLM, "horizontal", "sup", 250)
