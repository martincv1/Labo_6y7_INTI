from acquisition_tools import holoeye_SLM
import time

SLM = holoeye_SLM()
SLM.__init__

resol_SLM = (1080, 1920)
lista = [50, 100, 180, 250]
prueba = True
if prueba:
    for i in lista:
        patron = SLM.crear_patron(resol_SLM, "horizontal", "sup", i)
        SLM.mostrar_patron(patron)
        time.sleep(5)

# SLM.close   # no funciona

patron = SLM.crear_patron(resol_SLM, "horizontal", "sup", 210)
SLM.mostrar_patron(patron)
time.sleep(180)

#patron = SLM.crear_patron(resol_SLM, "horizontal", "sup", 250)
