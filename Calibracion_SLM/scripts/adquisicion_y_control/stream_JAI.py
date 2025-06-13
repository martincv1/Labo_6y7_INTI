# Importo librerias
import cv2

from Calibracion_SLM.scripts.adquisicion_y_control.acquisition_tools import jai_camera

"""----------------------------- Defino funciones de configuración -----------------------"""
BUFFER_COUNT = 16

camera = jai_camera(buffers=BUFFER_COUNT)

flag = True
while flag:
    frame = camera.get_frame()

    if frame:
        flag = False

    print(f" {camera.frame_rate_val:.1f} FPS  {camera.bandwidth_val / 1000000.0:.1f} Mb/s     ", end='\r')

camera.close()

cv2.imshow("foto", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
