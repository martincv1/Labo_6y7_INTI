# Importo librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from acquisition_tools import jai_camera

"""----------------------------- Defino funciones de configuración -----------------------"""
BUFFER_COUNT = 16

camera = jai_camera(buffers=BUFFER_COUNT)


while True:
    camera.reset_queue()
    frame = camera.get_frame()
    cv2.imshow("frame", frame)
    time.sleep(0.5)


print("Camera ready")
frames = camera.get_multiple_frame(10)
plt.imshow(frames[9])
plt.show()
print(frames[0])
# # mean_diff = np.mean(np.abs(frame1 - frame2))
# print(mean_diff)
# plt.imshow(frame1 - frame2)
# plt.show()
camera.close()

# cv2.imshow("foto1", frame1)
# cv2.imshow("foto2", frame2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
