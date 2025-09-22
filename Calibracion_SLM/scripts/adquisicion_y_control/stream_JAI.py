# Importo librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt

from acquisition_tools import jai_camera

"""----------------------------- Defino funciones de configuración -----------------------"""
BUFFER_COUNT = 16

camera = jai_camera(buffers=BUFFER_COUNT)

print("Camera ready")
flag = True
while flag:
    frame1, frame2 = camera.get_multiple_frame(2)

    if isinstance(frame1, np.ndarray):
        flag = False
    if isinstance(frame2, np.ndarray):
        flag = False

    print(
        f" {camera.frame_rate_val:.1f} FPS  {camera.bandwidth_val / 1000000.0:.1f} Mb/s     "
    )

frame1 = frame1.astype(np.float32)
frame2 = frame2.astype(np.float32)
mean_diff = np.mean(np.abs(frame1 - frame2))
print(mean_diff)
plt.imshow(frame1 - frame2)
plt.show()
camera.close()

# cv2.imshow("foto1", frame1)
# cv2.imshow("foto2", frame2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
